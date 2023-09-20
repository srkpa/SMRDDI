import glob
import json
import pickle
import random
from subprocess import call

from pytorch_lightning.callbacks import LearningRateMonitor
from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster
from torch.utils.data import TensorDataset
from tqdm import tqdm

from baddi.datasets.chem import get_smiles_alphabet
from baddi.datasets.data import *
from baddi.models.models import *
from baddi.models.trainer import *
from baddi.utils import wrapped_partial


def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)


def requeue(script):
    result = call(f"sbatch {script}", shell=True)
    print(f"\nrequeing job {script}... via sbatch")
    if result == 0:
        print("requeued exp ", script)
    else:
        print("requeue failed...")


def train_model(hparams, cluster=None, *args):
    set_all_seeds(seed=42)

    hpc_exp_number = hparams.hpc_exp_number

    exp = Experiment(
        name=hparams.job_name,
        save_dir=hparams.log_path,
        autosave=False,
    )
    exp.argparse(hparams)

    if cluster is not None:
        cluster.set_checkpoint_save_function(
            fx=requeue, kwargs={"script": hparams.test_tube_slurm_cmd_path}
        )
        cluster.set_checkpoint_load_function(fx=lambda **kargs: None, kwargs={})

    print("Cluster :", cluster)
    print("Exp name :", hparams.job_name)
    print("Hparams :", hparams)
    print("test tube log path :", hparams.log_path)

    checkpoint_path = f"{hparams.log_path}/{hparams.job_name}/version_{hpc_exp_number}"
    # checkpoint_file = []
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
        b_args = {
            arg: getattr(hparams, arg)
            for arg in vars(hparams)
            if arg.startswith("cnn_l")
        }
        cnn_sizes = [
            b_args[f"cnn_l{n}"] for n in range(len(b_args)) if b_args[f"cnn_l{n}"] > 0
        ]
        base_network = get_module_fn(network_name="conv1d")(
            vocab_size=len(vocab),
            embedding_size=hparams.embedding_size,
            cnn_sizes=cnn_sizes,
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
    elif hparams.base_network.lower() == "lstm":
        b_args = [
            arg
            for arg in vars(hparams)
            if arg.startswith("lstm_l")
            if getattr(hparams, arg) > 0
        ]
        base_network = get_module_fn(network_name="lstm")(
            vocab_size=len(vocab),
            embedding_size=hparams.embedding_size,
            lstm_hidden_size=hparams.lstm_hidden_size,
            nb_lstm_layers=len(b_args),
            bidirectional=hparams.bidirectional,
            normalize_features=hparams.normalize_features,
            dropout=hparams.dropout,
        )
    else:
        base_network = nn.Identity()

    if data_module.train_dataset.num_classes > 0:
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
        pred_loader = []

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
        opt_metric, modul = "val_acc_top5", SimCLR

        smiles = pd.read_csv(hparams.pred_file, index_col=0)

        t_moles = torch.Tensor(
            [
                np.squeeze(data_module.train_dataset.transform([mol]))
                for mol in smiles["SMILES"].values.tolist()
            ]
        )
        y_moles = torch.ones(len(t_moles))
        print("Pred Input: ", t_moles.shape, type(t_moles), t_moles[0])
        pred_loader = DataLoader(
            TensorDataset(t_moles.long(), y_moles), batch_size=64, shuffle=False
        )

    fc_layers_args = {
        arg: getattr(hparams, arg)
        for arg in vars(hparams)
        if arg.startswith("n_units_l")
    }
    fc_layers_dims = [
        fc_layers_args[f"n_units_l{n}"]
        for n in range(len(fc_layers_args))
        if fc_layers_args[f"n_units_l{n}"] > 0
    ]
    head = get_module_fn(network_name="feedforward")(
        input_size=input_size,
        fc_layer_dims=fc_layers_dims,
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
        enable_checkpointing=True,
        max_epochs=hparams.max_epochs,
        gpus=-1 if torch.cuda.is_available() else None,
        num_nodes=hparams.nb_nodes,
        auto_select_gpus=True,
        callbacks=[
            LearningRateMonitor("epoch"),
        ],
        resume_from_checkpoint=checkpoint_file[-1]
        if checkpoint_file
        else None,
        replace_sampler_ddp=False,
    )
    trainer.logger._default_hp_metric = None
    trainer.fit(model, data_module)
    final_model = modul.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    val_acc = trainer.callback_metrics[opt_metric].item()
    preds = trainer.predict(final_model, dataloaders=data_module.test_dataloader())
    torch.save(preds, f"{checkpoint_path}/predictions.pt")

    train_result = trainer.test(final_model, data_module.train_dataloader())
    test_result = trainer.test(final_model, dataloaders=data_module.test_dataloader())

    result = train_result[0] if len(train_result) else {}
    for i in test_result:
        result.update(i)

    print("Test:", result)
    exp_log = {
        "trial": hpc_exp_number,
        "value": val_acc,
        **result,
    }
    print("Exp log : ", exp_log)

    with open(f"{checkpoint_path}/report_v{hpc_exp_number}.json", "w") as f:
        json.dump(exp_log, f, indent=4)

    outs = []
    with torch.no_grad():
        for batch_x, batch_labels in tqdm(pred_loader):
            output = final_model.base_network(batch_x)
            outs.append(output.detach().cpu())

        if outs:
            preds = torch.cat(outs, dim=0)
            print("Final Prediction :", preds.shape)
            torch.save(preds, f"{checkpoint_path}/predictions.pt")

    with open(f"{checkpoint_path}/{hparams.base_network.lower()}.pkl", "wb") as f:
        pickle.dump(final_model, f)

    exp.save()


def run(hparams) -> None:
    cluster = SlurmCluster(
        hyperparam_optimizer=hparams,
        log_path=hparams.log_path,
        python_cmd="python3",
    )

    cluster.per_experiment_nb_nodes = hparams.nb_nodes
    cluster.per_experiment_nb_cpus = hparams.nb_cpus
    cluster.job_time = hparams.job_time
    cluster.minutes_to_checkpoint_before_walltime = 5
    cluster.memory_mb_per_node = hparams.mem
    cluster.add_slurm_cmd(cmd="account", value=hparams.account, comment="alloc")
    # cluster.add_slurm_cmd(cmd='array', value='1-{}%1'.format(hparams.array), comment='array job')
    cluster.notify_job_status(
        email="sewagnouin-rogia.kpanou.1@ulaval.ca", on_done=True, on_fail=True
    )

    if hparams.nb_gpus > 0:
        cluster.per_experiment_nb_gpus = hparams.nb_gpus
        cluster.optimize_parallel_cluster_gpu(
            train_model,
            nb_trials=hparams.nb_trials,
            job_name=hparams.job_name,
            # enable_auto_resubmit=False
        )
    else:
        cluster.per_experiment_nb_cpus = hparams.nb_cpus
        cluster.optimize_parallel_cluster_cpu(
            train_model,
            nb_trials=hparams.nb_trials,
            job_name=hparams.job_name,
            # enable_auto_resubmit=False
        )


if __name__ == "__main__":
    # Set up our argparser and make the y_val tunable.
    parser = HyperOptArgumentParser(strategy="random_search")

    parser.add_argument("--job_name", default="srkpa")
    parser.add_argument("--log_path", default="/some/path/to/log")
    parser.add_argument("--nb_gpus", help="gpus", default=0, type=int)
    parser.add_argument(
        "--accelerator", help="accelerator", default="ddp_spawn", type=str
    )
    parser.add_argument("--nb_trials", help="trials", default=10, type=int)
    parser.add_argument("--nb_nodes", help="nodes", default=1, type=int)
    parser.add_argument("--nb_cpus", help="cpus", default=3, type=int)
    parser.add_argument("--job_time", help="nb gpus", default="48:00:00", type=str)
    parser.add_argument("--mem", help="mem", default=74000, type=int)
    parser.add_argument("--account", default="def-eroussea")
    # parser.add_argument('--array', default=2)

    parser.add_argument("--drugs_file", default=None, type=str)
    parser.add_argument("--data_file", default="/some/path/to/data", type=str)
    parser.add_argument("--dataset_name", default="data", type=str)
    parser.add_argument("--min_count", help="mincount", default=0, type=int)
    parser.add_argument("--num_rounds", help="num rounds", default=1, type=int)
    parser.add_argument(
        "--transform", default="SmilesToSeq", help="feature transform", type=str
    )
    parser.add_argument(
        "--use_randomize_smiles", default=False, type=bool, help="Randomize smiles"
    )
    parser.add_argument(
        "--use_contrastive_transform",
        default=False,
        type=bool,
        help="constrative smiles",
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

    parser.add_argument("--batch_size", help="batch size", default=256, type=int)
    parser.add_argument("--max_epochs", help="max epochs", default=100, type=int)
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
        "--pred_file", default=None, type=str, help="pred file for constrastive model"
    )

    parser.add_argument(
        "--base_network", help="base network", default="conv1d", type=str
    )

    parser.json_config("--config", default="example.json")

    # parser.opt_list('--base_network', default='conv1d', options=['conv1d', 'lstm'], tunable=True)

    parser.opt_range(
        "--output_dim",
        default=32,
        type=int,
        tunable=True,
        low=32,
        high=512,
        nb_samples=10,
        help="constrastive learning output dim",
    )

    parser.opt_range(
        "--embedding_size",
        default=32,
        type=int,
        tunable=True,
        low=32,
        high=300,
        nb_samples=10,
    )
    for i in range(2):
        parser.opt_range(
            f"--cnn_l{i}",
            default=0,
            type=int,
            tunable=True,
            low=32,
            high=512,
            nb_samples=10,
        )
        parser.opt_range(
            f"--lstm_l{i}",
            default=0,
            type=int,
            tunable=True,
            low=32,
            high=512,
            nb_samples=10,
        )

    parser.opt_range(
        "--lstm_hidden_size",
        default=32,
        type=int,
        tunable=True,
        low=32,
        high=512,
        nb_samples=10,
    )

    for i in range(3):  # nb units
        parser.opt_range(
            f"--n_units_l{i}",
            default=0,
            type=int,
            tunable=True,
            low=16,
            high=256,  # 1024
            nb_samples=10,
        )

    parser.opt_range(
        "--kernel_size",
        default=3,
        type=int,
        tunable=True,
        low=1,
        high=30,
        nb_samples=10,
    )
    parser.opt_range(
        "--pooling_len", default=1, type=int, tunable=True, low=1, high=5, nb_samples=3
    )
    parser.opt_list("--pooling", default="avg", options=["avg", "max"], tunable=True)
    parser.opt_range(
        "--dilatation_rate",
        default=1,
        type=int,
        tunable=True,
        low=1,
        high=5,
        nb_samples=3,
    )
    parser.opt_list(
        "--normalize_features", default=True, options=[True, False], tunable=True
    )
    parser.opt_list("--b_norm", default=True, options=[True, False], tunable=True)
    parser.opt_range(
        "--dropout",
        default=0.00001,
        type=float,
        tunable=True,
        low=0.00001,
        high=0.5,
        nb_samples=10,
        log_base=10,
    )
    parser.opt_range(
        "--lr",
        default=0.00001,
        type=float,
        tunable=True,
        low=0.00001,
        high=0.1,
        nb_samples=10,
        log_base=10,
    )

    parser.opt_list(
        "--bidirectional", default=True, options=[True, False], tunable=True
    )

    hyperparams = parser.parse_args()
    run(hyperparams)
