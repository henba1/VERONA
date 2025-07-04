import torch
import numpy as np
from improved_diffusion.script_util import create_model_and_diffusion  #TODO : https://github.com/openai/improved-diffusion?tab=readme-ov-file

class PretrainedDiffusionDenoiser:
    def __init__(self, model, diffusion, device="cuda"):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.model.eval().to(device)
        
    @classmethod
    def load(cls, config_path="mnist_uncond_50M_500K.pt"):
        # Load MNIST-adapted diffusion model
        model_config = {
            'image_size': 28,
            'num_channels': 128,
            'num_res_blocks': 3,
            'num_classes': None,
            'learn_sigma': False,
            'class_cond': False,
            'use_checkpoint': False,
            'attention_resolutions': '16,8',
            'num_heads': 4,
            'num_heads_upsample': -1,
            'use_scale_shift_norm': True,
            'in_channels': 1  # MNIST is grayscale
        }
        
        model, diffusion = create_model_and_diffusion(
            **model_config,
            use_fp16=False  # MNIST doesn't need fp16
        )
        
        # Load MNIST-specific checkpoint
        state_dict = torch.load(config_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return cls(model, diffusion)

    def __call__(self, x):
        # x: [B, 1, 28, 28] in [0,1]
        x = x * 2 - 1  # Convert to [-1,1] range
        
        with torch.no_grad():
            t = torch.tensor([999], device=self.device)  # Reverse final step
            noise = self.diffusion.q_sample(x, t)
            denoised = self.diffusion.p_mean_variance(
                self.model, x, t, clip_denoised=True
            )["pred_xstart"]
            
        return (denoised + 1) / 2  # Convert back to [0,1]
