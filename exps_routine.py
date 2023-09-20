import glob
import json
import pickle
import random
from copy import deepcopy
from dataclasses import field, dataclass
from datetime import datetime

import joblib
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import TensorDataset
from tqdm import tqdm

from baddi.datasets.chem import get_smiles_alphabet
from baddi.datasets.data import *
from baddi.models.models import *
from baddi.models.trainer import *
from baddi.utils import wrapped_partial

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")


def create_timestamp_string(fmt="%Y-%m-%d.%H.%M.%S.%f"):
    now = datetime.now()
    return now.strftime(fmt)


def get_best_params(filepath: str, column: str = "value") -> Tuple:
    config = pd.read_csv(filepath_or_buffer=filepath, index_col=0)
    config = config.iloc[config[column].idxmax()]
    trial = config['number']
    params = {col.replace('params_', ''): value for col, value in config.to_dict().items() if col.startswith('params_')}
    return trial, params


def save(obj, filename, output_path):
    with open(os.path.join(output_path, filename), 'w') as CNF:
        json.dump(obj, CNF, indent=2)


def set_all_seeds(seed):
    """See https://pytorch.org/docs/stable/notes/randomness.html
    We activate the PyTorch options for best reproducibility. Note that this may be detrimental
    to processing speed, as per the above documentation:
      ...the processing speed (e.g. the number of batches trained per second) may be lower
      than when the model functions nondeterministically.
      However, even though single-run speed may be slower, depending on your application
      determinism may save time by facilitating experimentation, debugging,
      and regression testing.
    Args:
        seed (int): the seed which will be used for all random generators.
    """
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # numpy RNG
    np.random.seed(seed)
    # PL
    pl.seed_everything(seed, workers=True)


def do_compute_metrics(probas_pred, target):
    probas_pred = torch.exp(probas_pred)
    _, num_classes = probas_pred.shape
    pred = probas_pred.argmax(dim=1, keepdim=True).squeeze().numpy()
    probas_pred = probas_pred.numpy()
    target = target.numpy()
    binary_target = MultiLabelBinarizer().fit_transform(np.expand_dims(target, axis=1).tolist())
    return dict(
        acc=metrics.accuracy_score(target, pred),
        balanced_acc=metrics.balanced_accuracy_score(target, pred),
        auroc_ovr=metrics.roc_auc_score(target, probas_pred, multi_class='ovr'),
        auroc_ovo=metrics.roc_auc_score(target, probas_pred, multi_class='ovo'),
        macro_f1_score=metrics.f1_score(target, pred, average='macro'),
        micro_f1_score=metrics.f1_score(target, pred, average='micro'),
        weighted_f1_score=metrics.f1_score(target, pred, average='weighted'),
        macro_precision=metrics.precision_score(target, pred, average='macro'),
        micro_precision=metrics.f1_score(target, pred, average='micro'),
        weighted_precision=metrics.precision_score(target, pred, average='weighted'),
        macro_recall=metrics.recall_score(target, pred, average='macro'),
        micro_recall=metrics.recall_score(target, pred, average='micro'),
        weighted_recall=metrics.recall_score(target, pred, average='weighted'),
        cohen_kappa_score=metrics.cohen_kappa_score(target, pred),
        matthews_corrcoef=metrics.matthews_corrcoef(target, pred),
        log_loss=metrics.log_loss(target, probas_pred),
        macro_ap=metrics.average_precision_score(binary_target, probas_pred, average='macro'),
        micro_ap=metrics.average_precision_score(binary_target, probas_pred, average='micro'),
        weighted_ap=metrics.average_precision_score(binary_target, probas_pred, average='weighted'))


@dataclass
class Objective:
    save_dir: str
    vocab: field(default_factory=list)
    data: DrugDataModule
    base_network: str
    projection_head: str = 'feedforward'
    max_epochs: int = 500
    metadata: list = field(default_factory=list)
    kwargs: dict = None

    def __call__(self, trial: optuna.trial.Trial) -> float:
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        embedding_size = trial.suggest_int("embedding_size", 32, 300, log=True)
        normalize_features = trial.suggest_categorical("normalize_features", [True, False])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        checkpoint_path = f"{self.save_dir}/trial_{trial.number}/"
        os.makedirs(checkpoint_path, exist_ok=True)

        if self.base_network.lower() == 'conv1d':
            b_norm = trial.suggest_categorical("b_norm", [True, False])
            pooling = trial.suggest_categorical("pooling", ['avg', 'max'])
            n_layers = trial.suggest_int("cnn_sizes", 1, 2)
            cnn_sizes = [
                trial.suggest_int(f"cnn_sizes_l{i}", 32, 512, log=True)
                for i in range(n_layers)
            ]
            kernel_size = [
                trial.suggest_int(f"kernel_size_l{i}", 1, 30, log=True)
                for i in range(n_layers)
            ]
            pooling_len = [
                trial.suggest_int(f"pooling_len_l{i}", 1, 5, log=True)
                for i in range(n_layers)
            ]
            dilatation_rate = trial.suggest_int("dilatation_rate", 1, 5, log=True)
            base_network = get_module_fn(network_name='conv1d')(vocab_size=len(self.vocab),
                                                                embedding_size=embedding_size,
                                                                cnn_sizes=cnn_sizes,
                                                                kernel_size=kernel_size,
                                                                pooling_len=pooling_len,
                                                                pooling=pooling,
                                                                dilatation_rate=dilatation_rate,
                                                                activation='ReLU',
                                                                normalize_features=normalize_features,
                                                                b_norm=b_norm,
                                                                use_self_attention=False,
                                                                dropout=dropout
                                                                )
        elif self.base_network.lower() == 'lstm':
            nb_lstm_layers = trial.suggest_int("nb_lstm_layers", 1, 3)
            lstm_hidden_size = trial.suggest_int("lstm_hidden_size", 16, 512, log=True)
            bidirectional = trial.suggest_categorical("bidirectional", [True, False])
            base_network = get_module_fn(network_name='lstm')(
                vocab_size=len(self.vocab),
                embedding_size=embedding_size,
                lstm_hidden_size=lstm_hidden_size,
                nb_lstm_layers=nb_lstm_layers,
                bidirectional=bidirectional,
                normalize_features=normalize_features,
                dropout=dropout
            )
        else:
            # if not os.path.isdir(self.base_network):
            #     raise ValueError('Mauvaise valeur du réseau de base')
            # base_network = test_model(pretrained_model_path=self.base_network, return_model=True, attr='base_network')
            base_network = nn.Identity()

        fc_layers = trial.suggest_int("fc_layers", 1, 3)
        fc_layers_dims = [
            trial.suggest_int(f"n_units_l{i}", 16, 1024, log=True)
            for i in range(fc_layers)
        ]
        if self.data.train_dataset.num_classes > 0:
            input_size = base_network.output_dim * 2 if hasattr(base_network,
                                                                'output_dim') else self.data.train_dataset.n_features * 2
            output_dim = self.data.train_dataset.num_classes
            model = wrapped_partial(SiamNet, base_network=base_network, lr=lr, weight_decay=1e-4,
                                    max_epochs=self.max_epochs, num_classes=output_dim)
            opt_metric = "val_acc"
            modul = SiamNet
            test_loaders = self.data.test_dataloader()
        else:
            input_size = base_network.output_dim
            output_dim = trial.suggest_int("output_dim", 32, 512, log=True)
            model = wrapped_partial(SimCLR, base_network=base_network, lr=lr, temperature=0.07, weight_decay=1e-4,
                                    max_epochs=self.max_epochs, save_dir=checkpoint_path)
            opt_metric = "val_acc_top5"
            modul = SimCLR
            test_loaders = [self.kwargs]

        head = get_module_fn(network_name='feedforward')(input_size=input_size,
                                                         fc_layer_dims=fc_layers_dims,
                                                         output_dim=output_dim,
                                                         last_layer_activation=None)
        model = model(projection_head=head)
        print(model)
        trainer = pl.Trainer(
            default_root_dir=checkpoint_path,
            deterministic=True,
            logger=True,
            checkpoint_callback=True,
            max_epochs=self.max_epochs,
            gpus=-1 if torch.cuda.is_available() else None,
            auto_select_gpus=True,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor=opt_metric),
                       ModelCheckpoint(save_weights_only=True, mode="max", monitor=opt_metric),
                       LearningRateMonitor("epoch"),
                       ],
        )
        trainer.logger._default_hp_metric = None
        try:
            trainer.fit(model, self.data)
            final_model = modul.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

            # Test best model on train and validation set
            train_result = trainer.test(final_model, self.data.train_dataloader())
            test_result = trainer.test(final_model, dataloaders=self.data.test_dataloader())
            result = train_result[0] if len(train_result) else {}
            result = {'trial': trial.number, **result}
            for i in test_result:
                result |= i

            print('#Test step:', result)
            self.metadata.append(result)

            for cpt, test_loader in enumerate(test_loaders):
                trainer.predict(final_model, dataloaders=test_loader)
                try:
                    pred = np.vstack(
                        deepcopy(trainer).predict(deepcopy(final_model), dataloaders=self.kwargs['pred_on']))
                    torch.save(pred, f'{checkpoint_path}/predictions_dataloader_{cpt}.pt')
                    print('#Pred step - shape :', pred.shape)
                except Exception:
                    print(f"#Pred step {cpt} : Failed!")
            with open(f"{os.path.join(checkpoint_path, str(trial.number))}.pkl", "wb") as f:
                pickle.dump(final_model, f)
        except RuntimeError as err:
            print(f"Trial {trial.number}; {err}, {type(err)}")
            return 0.0

        return trainer.callback_metrics[opt_metric].item()


def train_model(experiment_name: str, dataset_params: Dict, train_params: Dict, model_params: Dict, seed: int = 42,
                pruning: bool = True,
                n_trials: int = 100, output_path: str = None, config_id: str = None, **kwargs):

    output_path += f"/{config_id}"

    os.makedirs(output_path, exist_ok=True)
    set_all_seeds(seed=seed)

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()
    )
    os.makedirs(output_path, exist_ok=True)
    max_epochs = train_params.pop('max_epochs')

    data = DrugDataModule(
        dataset_params=dataset_params,
        train_params=train_params,
        seed=seed,
    )
    data.setup()
    study = optuna.create_study(study_name=experiment_name, direction="maximize", pruner=pruner)
    study_obj_params = dict(
        save_dir=output_path,
        data=data,
        vocab=get_smiles_alphabet(),
        max_epochs=max_epochs,
        **model_params
    )

    if kwargs.get('pred_file', None) is not None:
        smiles = pd.read_csv(kwargs.get('pred_file'), index_col=0)
        t_moles = torch.Tensor(
            [np.squeeze(data.train_dataset.transform([mol])) for mol in smiles['SMILES'].values.tolist()])
        y_moles = torch.ones(len(t_moles))
        print('# Pred Step - Input: ', t_moles.shape, type(t_moles), t_moles[0])
        pred_loader = DataLoader(
            TensorDataset(t_moles.long(), y_moles),
            batch_size=64,
            shuffle=False
        )
        study_obj_params['kwargs'] = pred_loader

    objective = Objective(**study_obj_params)
    study.optimize(objective, n_trials=n_trials)
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print(f"Seed {seed} / Best trial: ")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    joblib.dump(study, f"{output_path}/{experiment_name}_study.pkl")
    reports = study.trials_dataframe()
    reports.to_csv(path_or_buf=f"{output_path}/{experiment_name}.csv")
    results = pd.DataFrame(objective.metadata)
    results.to_csv(path_or_buf=f"{output_path}/metrics.csv")


def test_model(dataset_params: Dict = None, batch_size: int = 64, pretrained_model_path: str = None,
               return_pred: bool = False,
               attr: str = None, return_model: bool = False,
               **kwargs) -> Any:
    set_all_seeds(seed=42)
    experiment_file = glob.glob(f"{pretrained_model_path}/*.csv")[-1]
    best_trial, best_config = get_best_params(filepath=experiment_file, column="value")
    checkpoint_dir = os.path.join(pretrained_model_path, f'trial_{best_trial}', 'lightning_logs/version_*/checkpoints/')
    checkpoint_path = glob.glob(f"{checkpoint_dir}/*.ckpt")
    checkpoint_path = checkpoint_path[-1] if len(checkpoint_path) else None
    module = SimCLR if dataset_params is None else SiamNet

    print('Experiment file : ', experiment_file)
    print('Module loaded : ', module)
    print('Best trial id: ', best_trial)
    if os.path.isfile(checkpoint_path):
        print(f"Found pretrained model at {checkpoint_path}, loading...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device :', device)
        with open(f"{os.path.join(pretrained_model_path, f'trial_{best_trial}', f'{best_trial}.pkl')}", "rb") as f:
            model = pickle.load(f)
        # model = module.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=device)
        model.eval()
        if attr is not None:
            model.freeze()
            model = getattr(model, attr)
        print(model)
        if return_model:
            return model
        if dataset_params is not None:
            add_args = args_setup(dataset_params=dataset_params, train_params=None)
            test_dataset = DrugDataset(**dataset_params, split='test', **add_args)
            loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
            )
            outs, labels = [], []
            with torch.no_grad():
                for batch_x, batch_labels in tqdm(loader):
                    output = model(batch_x)
                    outs.append(output.detach().cpu())  #
                    labels.append(batch_labels)
                outs = torch.cat(outs, dim=0)
                labels = torch.cat(labels, dim=0)

            if return_pred:
                return data.TensorDataset(outs, labels)
            scores = do_compute_metrics(probas_pred=outs, target=labels)
            scores_df = pd.DataFrame(data=scores.items())
            print(scores)
            scores_df.to_csv(f'{pretrained_model_path}/best_trial_test.csv')
            return scores


def run_experiment(**kwargs):
    # use_pretrained_model = kwargs.get('pretrained_model_path', None)
    # if use_pretrained_model is None:
    train_model(**kwargs)
    # else:
    #     test_model(**kwargs)
