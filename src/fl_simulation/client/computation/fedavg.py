"""A worker for Federated Averaging algorithm."""

from torch.optim import Optimizer, SGD

from .fixed_epochs_worker import FixedEpochsWorker


class FedAvgWorker(FixedEpochsWorker):
    """Federated Averaging worker."""

    def get_opt(self, *opt_arg, **opt_kwarg) -> Optimizer:
        """Get the local optimizer.

        Returns:
            Optimizer: optimizer to be used.
        """
        return SGD((p for p in self.local_model.parameters() if p.requires_grad), *opt_arg, **opt_kwarg)
