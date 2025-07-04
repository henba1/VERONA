import torch
from .randomized_smoothing import RandomizedSmoothingCertifier
from diffusion_denoiser import PretrainedDiffusionDenoiser  #TODO

class DenoisedSmoothingCertifier(RandomizedSmoothingCertifier):
    def __init__(self, denoiser_path: str, **kwargs):
        super().__init__(**kwargs)
        self.denoiser = PretrainedDiffusionDenoiser.load(denoiser_path).eval()
        
    def _sample_counts(self, model: torch.nn.Module, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        def denoise_and_classify(batch: torch.Tensor):
            denoised = self.denoiser(batch)
            return model(denoised)
            
        # Override base sampling with denoising
        counts = torch.zeros(model.num_classes)
        x = x.to(next(model.parameters()).device)
        
        for _ in range(0, num_samples, self.batch_size):
            batch = x.repeat(min(self.batch_size, num_samples), 1, 1, 1)
            noise = torch.randn_like(batch) * self.sigma
            outputs = denoise_and_classify(batch + noise).argmax(1)
            counts += torch.bincount(outputs, minlength=model.num_classes)
            
        return counts
