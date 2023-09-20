import torch
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
import os
from torch import tensor, load, stack, Tensor
from itertools import chain
from pathlib import Path


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions,
            os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices,
            os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"),
        )


# or you can set `writer_interval="batch"` and override `write_on_batch_end` to save
# predictions at batch level
def _load_batches_indices_predictions(path: Path, regex: str = "batch_indices"):
    results = []
    for file in path.glob(f"{regex}*"):
        results += list(chain.from_iterable(load(file)[0]))

    return stack(results) if isinstance(results[0], (list, Tensor)) else tensor(results)


def gather_all_predictions(filepath: str):
    """_summary_

    Args:
        filepath (str): _description_
    """
    path = Path(filepath)
    indices = _load_batches_indices_predictions(path, regex="batch")
    predictions = _load_batches_indices_predictions(path, regex="predictions")
    print(predictions.shape, indices.shape)
    torch.save(predictions[indices.argsort()], path.parent / "predictions.pt")
