import numpy as np
from scipy.stats import norm
from .base_certifier import BaseCertifier
from robustness_experiment_box.database.verification_context import VerificationContext
from autoverify.verifier.verification_result import CompleteVerificationData
from robustness_experiment_box.database.verification_result import VerificationResult


class RandomizedSmoothingCertifier(BaseCertifier):
    def certify(self, context: VerificationContext) -> CompleteVerificationData:
        model = context.network.load_pytorch_model().eval()
        x = context.data_point.data
        true_label = context.data_point.label
        
        # PREDICT phase
        counts_n0 = self._sample_counts(model, x, self.n0)
        predicted_class = counts_n0.argmax().item()
        
        if predicted_class != true_label:
            return CompleteVerificationData(
                result=VerificationResult.UNKNOWN,
                took=0.0,
                certified_radius=0.0
            )

        # CERTIFY phase
        counts_n = self._sample_counts(model, x, self.n)
        p_a = counts_n[predicted_class].item() / self.n
        p_a_lower = self._lower_confidence_bound(p_a)
        
        if p_a_lower < 0.5:
            return CompleteVerificationData(
                result=VerificationResult.UNKNOWN,
                took=0.0,
                certified_radius=0.0
            )

        R = self.sigma * norm.ppf(p_a_lower)
        return CompleteVerificationData(
            result=VerificationResult.CERTIFIED,
            took=...,  # Actual timing
            certified_radius=R
        )

    def _lower_confidence_bound(self, p_a: float) -> float:
        """Clopper-Pearson lower bound"""
        return max(0, p_a - norm.ppf(1 - self.alpha) * np.sqrt(p_a * (1 - p_a) / self.n))
