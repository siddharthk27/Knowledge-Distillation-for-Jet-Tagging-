import torch
import torch.nn as nn
import sys

# 1. Imports
from lgatr import LGATr, MLPConfig, SelfAttentionConfig

# Try to import LGATRConfig (fallback to None if missing)
try:
    from lgatr.gatr_config import LGATRConfig
except ImportError:
    try:
        from lgatr.gatr_config import GATrConfig as LGATRConfig
    except ImportError:
        LGATRConfig = None

# Fallback for embed_point
try:
    from lgatr.interface import embed_point
except ImportError:
    from lgatr import embed_vector, embed_scalar
    def embed_point(p4):
        # Fallback: Embed (E, px, py, pz) -> Vector(px,py,pz) + Scalar(E)
        spatial = p4[..., 1:] 
        energy = p4[..., 0:1] 
        return embed_vector(spatial) + embed_scalar(energy)

class LGATrWrapper(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()

        # 2. Configure Model
        # CORRECT CONFIG from checkpoint diagnostic:
        # Based on actual checkpoint weight analysis
        
        if LGATRConfig:
            print("Initializing using LGATRConfig object...")
            cfg = LGATRConfig(
                in_mv_channels=16,
                out_mv_channels=16,
                hidden_mv_channels=16,
                in_s_channels=1,
                out_s_channels=None,
                hidden_s_channels=64,
                num_blocks=12,
                num_heads=4,
                dropout_prob=0.0
            )
            self.encoder = LGATr(cfg)
        else:
            print("Initializing using direct kwargs...")
            self.encoder = LGATr(
                in_mv_channels=16,
                out_mv_channels=16,
                hidden_mv_channels=16,
                in_s_channels=1,
                out_s_channels=None,
                hidden_s_channels=64,
                num_blocks=12,
                attention=SelfAttentionConfig(
                    num_heads=4,
                    dropout_prob=0.0
                ),
                mlp=MLPConfig(
                    dropout_prob=0.0
                ) 
            )

        # 3. Classifier Head (matches hidden_s_channels=64)
        self.classifier = nn.Linear(64, 2) 

        # 4. Load Weights
        if checkpoint_path:
            self._load_weights(checkpoint_path)

    def _load_weights(self, path):
        """
        Load weights from checkpoint that has structure:
        {
            'model': state_dict with 'net.' prefix,
            'optimizer': ...,
            'scheduler': ...,
            'ema': ...
        }
        """
        print(f"Loading weights from {path}...")
        
        # Monkey-patch 'gatr' module if needed
        import lgatr
        if 'gatr' not in sys.modules:
            sys.modules['gatr'] = lgatr
        
        # Load checkpoint
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        
        # Navigate to model weights
        # Based on your output, the structure is checkpoint["model"] with "net." prefix
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("✓ Found 'model' key in checkpoint")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("✓ Found 'model_state_dict' key in checkpoint")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("✓ Found 'state_dict' key in checkpoint")
        else:
            state_dict = checkpoint
            print("⚠ Using root level as state_dict")
        
        # Clean keys: Remove "net." prefix
        # Keys in checkpoint: "net.linear_in.weight", "net.blocks.0.attention..."
        # Keys expected by encoder: "linear_in.weight", "blocks.0.attention..."
        clean_state_dict = {}
        for k, v in state_dict.items():
            # Remove "net." prefix
            if k.startswith("net."):
                new_k = k[4:]  # Remove "net." (4 characters)
            else:
                new_k = k
            clean_state_dict[new_k] = v
        
        # Load into encoder
        missing, unexpected = self.encoder.load_state_dict(clean_state_dict, strict=False)
        
        # Report results
        total_params = len(self.encoder.state_dict())
        loaded_params = total_params - len(missing)
        
        print(f"\n{'='*60}")
        print(f"Weight Loading Summary:")
        print(f"{'='*60}")
        print(f"✓ Successfully loaded: {loaded_params}/{total_params} parameters")
        print(f"✗ Missing keys: {len(missing)}")
        print(f"⚠ Unexpected keys: {len(unexpected)}")
        
        if len(missing) > 0:
            print(f"\nFirst 5 missing keys:")
            for k in list(missing)[:5]:
                print(f"  - {k}")
        
        if len(unexpected) > 0:
            print(f"\nFirst 5 unexpected keys:")
            for k in list(unexpected)[:5]:
                print(f"  - {k}")
        
        print(f"{'='*60}\n")
        
        # Warn if too many missing keys
        if len(missing) > 10:
            print("⚠ WARNING: Many keys are missing! The model may not work correctly.")
            print("   This might indicate a mismatch between checkpoint and model architecture.")

    def forward(self, batch_dict):
        # A. Inputs
        p4 = batch_dict["Pmu"]         # [Batch, N, 4]
        mask = batch_dict["atom_mask"] # [Batch, N]
        batch_size = p4.shape[0]
        device = p4.device

        # B. Beam Injection (1,0,0,1 particle)
        beam_p4 = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device).view(1, 1, 4)
        beam_p4 = beam_p4.expand(batch_size, -1, -1)
        
        p4_in = torch.cat([beam_p4, p4], dim=1)        
        beam_mask = torch.ones((batch_size, 1), device=device, dtype=mask.dtype)
        mask_in = torch.cat([beam_mask, mask], dim=1) 

        # C. Embedding
        # Checkpoint expects 16 MV input channels, not 1!
        # We embed the 4-momentum and repeat/expand to 16 channels
        x_mv_single = embed_point(p4_in)  # [B, N, 1, 16] - single MV per particle
        
        # Expand to 16 input channels by repeating the same embedding
        # This is a simple strategy - you may want to use different embeddings
        x_mv = x_mv_single.repeat(1, 1, 16, 1)  # [B, N, 16, 16]
        
        # Scalar input: Use energy as the scalar feature
        x_s = p4_in[..., 0:1]  # [B, N, 1] - energy channel

        # D. Forward
        out_mv, out_s = self.encoder(x_mv, x_s, attention_mask=mask_in)
        
        # E. Readout (Beam Token at index 0)
        global_feature = out_s[:, 0, :] 
        logits = self.classifier(global_feature)
        
        return logits, global_feature