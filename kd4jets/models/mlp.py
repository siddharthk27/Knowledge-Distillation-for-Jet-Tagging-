# system imports
import sys

# 3rd party imports
from pytorch_lightning import LightningModule
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
from .lgatr_wrapper import LGATrWrapper
from ..knowledge_distillation_base import KnowledgeDistillationBase
from .lorentz_net import LorentzNetWrapper

class MLPKD(KnowledgeDistillationBase):
    def __init__(self, hparams):
        super().__init__(hparams)        
    
    def get_student(self, hparams):
        """
        return a torch.nn.Module model that is the student model. The forward()
        function should return logits and penultimate layer representation
        """
        return MLPTagger(**hparams)
        
    def get_teacher(self, hparams):
        """
        Initializes the LGATr teacher using the checkpoint defined in config.
        """
        # Get path from the yaml config
        ckpt_path = hparams.get("teacher_checkpoint")
        if not ckpt_path:
            raise ValueError("Error: 'teacher_checkpoint' is missing in your yaml config!")

        print(f"Loading Teacher from: {ckpt_path}")
        model = LGATrWrapper(checkpoint_path=ckpt_path)

        # Freeze the teacher
        for param in model.parameters():
            param.requires_grad = False
        return model

class MLPTagger(nn.Module):
    def __init__(
            self, 
            d_input = 5, 
            d_ff = 72, 
            d_output = 2, 
            dropout = 0., 
            depth = 2, 
            **kwargs
        ):
        super().__init__()
        
        mlp = []
        d = d_input * 128
        for _ in range(depth - 1):
            mlp.extend([
                nn.Linear(d, d_ff, bias=True),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(d_ff),
            ])
            d = d_ff
        
        self.mlp = nn.Sequential(*mlp)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d, d_output, bias=True)
        )
        
    def preprocess(self, batch):
        mask = batch["label"].float()
        Pjet = (batch["Pmu"][:, 2:] * mask[:, :, None]).sum(1)
        rel_pT = norm(batch["Pmu"][:, 2:, 1:3])/norm(Pjet[:, None, 1:3])
        deta = torch.atanh(
            batch["Pmu"][:, 2:, [3]] / norm(batch["Pmu"][:, 2:, 1:4])
        ) - torch.atanh(
            Pjet[:, [3]] / norm(Pjet[:, 1:4])
        ).view(-1, 1, 1)
        dphi = torch.atan2(
            batch["Pmu"][:, 2:, [2]], batch["Pmu"][:, 2:, [1]]
        ) - torch.atan2(
            Pjet[:, [2]], Pjet[:, [1]]
        ).view(-1, 1, 1)
        dphi = torch.remainder(dphi + torch.pi, 2 * torch.pi) - torch.pi
        features = torch.cat(
            [rel_pT, deta, dphi, batch["nodes"][:, 2:, :1]],
            dim = -1
        ).float()
        features.masked_fill_(
            ( ~ batch["label"][:, :, None]) | (features != features), 0
        )
        features = features[:, :128]
        features = torch.cat(
            [
                features, 
                torch.zeros((
                    features.shape[0],
                    128 - features.shape[1],
                    features.shape[2]
                )).to(features)
            ], dim = 1
        )
        return features.view(features.shape[0], -1)
        
    def forward(self, batch):
        features = self.preprocess(batch)   
        z = self.mlp(features)
        output = self.output_layer(z)
        
        return output, z
    
def norm(x):
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)