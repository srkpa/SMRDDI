import glob
import json
import pickle
import random
from subprocess import call

from pytorch_lightning.callbacks import LearningRateMonitor
from typing import NamedTuple
from torch.utils.data import TensorDataset
from tqdm import tqdm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from baddi.datasets.chem import get_smiles_alphabet
from baddi.datasets.data import *
from baddi.models.models import *
from baddi.models.trainer import *
from baddi.utils import wrapped_partial
import logging
from argparse import ArgumentParser
from baddi.models.callbacks import CustomWriter


def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)


def train_model(hparams, *args):
    set_all_seeds(seed=42)

    Experiment = NamedTuple(
        "Experiment", name=str, save_dir=str, autosave=bool, hpc_exp_number=int
    )
    exp = Experiment(
        name=hparams.job_name,
        save_dir=hparams.log_path,
        autosave=False,
        hpc_exp_number=hparams.hpc_exp_number,
    )

    print("Exp name :", hparams.job_name)
    print("Hparams :", hparams)
    print("test tube log path :", hparams.log_path)

    checkpoint_path = (
        f"{hparams.log_path}/{hparams.job_name}/version_{exp.hpc_exp_number}"
    )
    prediction_path = f"{checkpoint_path}/pred_path"

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(prediction_path, exist_ok=True)

    checkpoint_file = glob.glob(
        f"{checkpoint_path}/lightning_logs/version_*/checkpoints/*.ckpt"
    )
    print("Resume training from :", checkpoint_file)

    dataset_params = dict(
        data_file=hparams.data_file,
        dataset_name=hparams.dataset_name,
        debug=hparams.debug,
        drugs_file=hparams.drugs_file,
        min_count=hparams.min_count,
        num_rounds=hparams.num_rounds,
        transform=hparams.transform,
        use_randomize_smiles=hparams.use_randomize_smiles,
        use_contrastive_transform=hparams.use_contrastive_transform,
        target_transform=None if hparams.use_contrastive_transform else LabelEncoder(),
        embedding=[hparams.embedding_smiles, hparams.embedding]
        if hparams.embedding is not None
        else None,
    )

    print("Dataset params :", dataset_params)

    train_params = dict(
        batch_size=hparams.batch_size,
        max_epochs=hparams.max_epochs,
        test_size=hparams.test_size,
        valid_size=hparams.test_size,
        splitter=hparams.splitter,
        balance=hparams.balance,
    )

    print("Train params : ", train_params)
    data_module = DrugDataModule(
        dataset_params=dataset_params,
        train_params=train_params,
        seed=42,
    )
    data_module.setup()

    vocab = get_smiles_alphabet()
    if hparams.base_network.lower() == "conv1d":
        base_network = get_module_fn(network_name="conv1d")(
            vocab_size=len(vocab),
            embedding_size=hparams.embedding_size,
            cnn_sizes=hparams.cnn,
            kernel_size=hparams.kernel_size,
            pooling_len=hparams.pooling_len,
            pooling=hparams.pooling,
            dilatation_rate=hparams.dilatation_rate,
            activation="ReLU",
            normalize_features=hparams.normalize_features,
            b_norm=hparams.b_norm,
            use_self_attention=False,
            dropout=hparams.dropout,
        )

    else:
        base_network = nn.Identity()

    if (
        not hparams.use_contrastive_transform
    ):  # data_module.train_dataset.num_classes > 0
        input_size = (
            base_network.output_dim * 2
            if hasattr(base_network, "output_dim")
            else data_module.train_dataset.n_features * 2
        )
        output_dim = data_module.train_dataset.num_classes
        model = wrapped_partial(
            SiamNet,
            base_network=base_network,
            lr=hparams.lr,
            weight_decay=1e-4,  # 1e-4
            max_epochs=hparams.max_epochs,
            num_classes=data_module.train_dataset.num_classes,
        )
        opt_metric, modul = "val_acc", SiamNet
    else:
        input_size = base_network.output_dim
        output_dim = hparams.output_dim

        model = wrapped_partial(
            SimCLR,
            base_network=base_network,
            lr=hparams.lr,
            temperature=0.07,
            weight_decay=1e-4,
            max_epochs=hparams.max_epochs,
            save_dir=checkpoint_path,
        )
        modul = SimCLR

    head = get_module_fn(network_name="feedforward")(
        input_size=input_size,  # [128, 256, 512]
        fc_layer_dims=hparams.fc,  # [128]
        output_dim=output_dim,
        last_layer_activation=None,
        dropout=hparams.dropout,
    )

    model = model(projection_head=head)
    print("Model :", model)

    trainer = pl.Trainer(
        default_root_dir=checkpoint_path,
        deterministic=True,
        logger=True,
        auto_scale_batch_size=True,
        enable_checkpointing=True,  # checkpoint_callback=True, pl 1.7.7
        max_epochs=hparams.max_epochs,
        # gpus=-1 if torch.cuda.is_available() else None,
        accelerator="gpu",
        devices=[0, 1],  # [1, 2]
        strategy="ddp",
        precision="16",
        # -num_nodes=hparams.nb_nodes,
        # auto_select_gpus=True,
        callbacks=[
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor="val_loss", patience=50, min_delta=0.0001),
            RichProgressBar(),
            CustomWriter(output_dir=prediction_path, write_interval="epoch"),
        ],
        # resume_from_checkpoint="/home/srkpanou/experiments/LC-CURVE/version_2000000/lightning_logs/version_6/checkpoints/epoch=38-step=12675.ckpt",
        # if len(checkpoint_file) > 0
        # else None,
        # replace_sampler_ddp=False,  # because of weighted random sampler
        limit_train_batches=20000,  # hparams.limit_train_batches,
        # limit_val_batches=8,
    )
    trainer.logger._default_hp_metric = None
    trainer.fit(model, data_module)
    final_model = modul.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    trainer.predict(
        final_model, dataloaders=data_module.test_dataloader(), return_predictions=False
    )
    # print("len predictions", len(preds))
    # preds = torch.cat(preds, dim=0)
    # print("Predictions", preds.shape)
    # torch.save(preds, f"{checkpoint_path}/predictions.pt")

    with open(f"{checkpoint_path}/{hparams.base_network.lower()}.pkl", "wb") as f:
        pickle.dump(final_model, f)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--job_name", default="SIMCLR")
    parser.add_argument("--log_path", default="/home/srkpanou/experiments")
    parser.add_argument("--nb_gpus", help="gpus", default=3, type=int)
    parser.add_argument(
        "--accelerator", help="accelerator", default="ddp_spawn", type=str
    )
    parser.add_argument("--nb_cpus", help="cpus", default=3, type=int)
    parser.add_argument("--hpc_exp_number", type=int, default=1)

    parser.add_argument(
        "--drugs_file",
        default="/home/srkpanou/BADDI/data/chembl_32_smiles.csv",
        type=str,
    )
    parser.add_argument("--data_file", default=None, type=str)
    parser.add_argument("--dataset_name", default="CHEMBL32", type=str)
    parser.add_argument("--min_count", help="mincount", default=0, type=int)
    parser.add_argument("--num_rounds", help="num rounds", default=1, type=int)
    parser.add_argument(
        "--pred_file", type=str, default="/home/srkpanou/BADDI/data/drugbank_smiles.csv"
    )
    parser.add_argument(
        "--transform", default="SmilesToSeq", help="feature transform", type=str
    )
    parser.add_argument(
        "--use_randomize_smiles", default=False, type=bool, help="Randomize smiles"
    )
    parser.add_argument(
        "--use_contrastive_transform",
        default=True,
        type=bool,
        help="Self-supervised mode or not?",
    )
    parser.add_argument("--debug", default=False, type=bool, help="use a subset or not")
    parser.add_argument(
        "--embedding", default=None, type=str, help="drugs embedded smiles"
    )
    parser.add_argument(
        "--embedding_smiles",
        default=None,
        type=str,
        help="learned drugs embedding .pt file",
    )

    parser.add_argument("--batch_size", help="batch size", default=64, type=int)
    parser.add_argument("--max_epochs", help="max epochs", default=1, type=int)
    parser.add_argument("--test_size", help="test size", default=0.2, type=float)
    parser.add_argument("--valid_size", help="valid size", default=0.2, type=float)
    parser.add_argument(
        "--splitter",
        type=str,
        default=None,
        help="split to use for constrative learning",
    )

    parser.add_argument(
        "-b",
        "--balance",
        type=str,
        choices=["equal", "class_weight", "auto", "x", "sample_weight"],
        default="x",
        help="techniques to handle imbalanced dataset",
    )

    parser.add_argument(
        "--base_network", help="base network", default="conv1d", type=str
    )

    parser.add_argument(
        "--output_dim",
        default=128,
        type=int,
        help="constrastive learning output dim",
    )

    parser.add_argument(
        "--embedding_size",
        default=116,
        type=int,
    )

    parser.add_argument("--cnn", type=int, nargs="*")

    parser.add_argument("--fc", type=int, nargs="*")

    parser.add_argument("--kernel_size", type=int, default=7, nargs="*")
    parser.add_argument(
        "--pooling_len",
        default=1,
        type=int,
    )
    parser.add_argument("--pooling", default="max", type=str)

    parser.add_argument("--dilatation_rate", default=3, type=int, nargs="*")
    parser.add_argument("--normalize_features", default=True, type=bool)
    parser.add_argument("--b_norm", default=True, type=bool)
    parser.add_argument(
        "--dropout",
        default=0.006,
        type=float,
    )
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)

    hyperparams = parser.parse_args()
    train_model(hyperparams)

# python train.py --cnn 64 128 256 512 --fc 256 --kernel_size 7 5 3 3 --pooling_len 4 --batch_size 1024 --max_epochs 100 --hpc_exp_number 1 --drugs_file /home/srkpanou/BADDI/data/0/chembl_32_smiles.csv --job_name NCURVE
# python train.py --cnn 64 128 256 512 --fc 256 --kernel_size 7 5 3 3 --pooling_len 4 --batch_size 1024 --max_epochs 38 --hpc_exp_number 2000000 --drugs_file /home/srkpanou/BADDI/data/FS/chembl_32_smiles.csv --job_name LC-CURVE
# python train.py --cnn 64 128 256 512 --fc 256 --kernel_size 7 5 3 3 --pooling_len 4 --batch_size 1024 --max_epochs 50 --hpc_exp_number 32 --drugs_file /home/srkpanou/BADDI/data/3_2/chembl_32_smiles.csv --job_name LC-CURVE
