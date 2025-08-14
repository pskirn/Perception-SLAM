# lib/multi_depth_model_woauxi.py
import torch
import torch.nn as nn
import numpy as np

class RelDepthModel(nn.Module):
    def __init__(self, backbone='resnext101'):
        super().__init__()
        self.backbone = backbone
        
    def inference(self, x):
        # Generate synthetic depth for testing
        b, c, h, w = x.shape
        
        # Create a simple depth pattern (e.g., gradient)
        depth = torch.zeros(b, 1, h, w)
        for i in range(h):
            depth[:, :, i, :] = i / h * 10  # Depth increases with y
            
        # Add some noise
        noise = torch.randn_like(depth) * 0.1
        depth = depth + noise
        
        return depth