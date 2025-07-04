from abc import ABC, abstractmethod
import torch
from robustness_experiment_box.database.verification_context import VerificationContext
from autoverify.verifier.verification_result import CompleteVerificationData

class BaseCertifier(ABC):
    def __init__(
        self,
        sigma: float = 0.25,
        alpha: float = 0.01,
        n0: int = 100,
        n: int = 100000,
        batch_size: int = 1024
    ):
        self.sigma = sigma
        self.alpha = alpha
        self.n0 = n0
        self.n = n
        self.batch_size = batch_size

    @abstractmethod
    def certify(self, context: VerificationContext) -> CompleteVerificationData:
        pass

    def _predict(self, model: torch.nn.Module, x: torch.Tensor) -> int:
        """Cohen's PREDICT algorithm"""
        counts = self._sample_counts(model, x, self.n0)
        return counts.argmax().item()

    def _sample_counts(self, model: torch.nn.Module, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Batch-processed version of Cohen's sampling"""
        counts = torch.zeros(model.num_classes)
        x = x.to(next(model.parameters()).device)
        
        for _ in range(0, num_samples, self.batch_size):
            batch = x.repeat(min(self.batch_size, num_samples), 1, 1, 1)
            noise = torch.randn_like(batch) * self.sigma
            outputs = model(batch + noise).argmax(1)
            counts += torch.bincount(outputs, minlength=model.num_classes)
        
        return counts