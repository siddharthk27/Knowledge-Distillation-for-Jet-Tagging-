import torch
import torch.nn as nn
import sys # <--- Needed for the patch

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
        if LGATRConfig:
            print("Initializing using LGATRConfig object...")
            cfg = LGATRConfig(
                in_mv_channels=1,
                out_mv_channels=1,
                hidden_mv_channels=16,
                in_s_channels=0,
                out_s_channels=None,
                hidden_s_channels=32,
                num_blocks=12,
                num_heads=8,
                dropout_prob=0.0
            )
            self.encoder = LGATr(cfg)
        else:
            print("Initializing using direct kwargs...")
            self.encoder = LGATr(
                in_mv_channels=1,
                out_mv_channels=1,
                hidden_mv_channels=16,
                in_s_channels=0,
                out_s_channels=None,
                hidden_s_channels=32,
                num_blocks=12,
                attention=SelfAttentionConfig(
                    num_heads=8, 
                    dropout_prob=0.0
                ),
                mlp=MLPConfig(
                    dropout_prob=0.0
                ) 
            )

        # 3. Classifier Head
        self.classifier = nn.Linear(32, 2) 

        # 4. Load Weights
        if checkpoint_path:
            self._load_weights(checkpoint_path)

    def _load_weights(self, path):
            print(f"Loading weights from {path}...")
            
            # Monkey-patch 'gatr' module if needed
            import lgatr
            if 'gatr' not in sys.modules:
                sys.modules['gatr'] = lgatr
            
            try:
                state_dict = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                state_dict = torch.load(path, map_location="cpu")
            
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Clean keys
            clean_state_dict = {}
            for k, v in state_dict.items():
                # Strip common prefixes from the training wrapper
                new_k = k.replace("model.net.", "").replace("module.", "").replace("net.", "")
                clean_state_dict[new_k] = v
                
            # --- THE FIX IS HERE ---
            # Instead of loading into 'self' (which expects 'encoder.layers...'),
            # we load directly into 'self.encoder' (which expects 'layers...').
            missing, unexpected = self.encoder.load_state_dict(clean_state_dict, strict=False)
            
            print(f"Weights Loaded into Encoder. Missing keys: {len(missing)}")
            if len(missing) > 0:
                print(f"Example missing: {missing[0]}")

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
        x_mv = embed_point(p4_in).unsqueeze(2) 
        x_s = torch.zeros(batch_size, p4_in.shape[1], 0, device=device)

        # D. Forward
        out_mv, out_s = self.encoder(x_mv, x_s, attention_mask=mask_in)
        
        # E. Readout (Beam Token at index 0)
        global_feature = out_s[:, 0, :] 
        logits = self.classifier(global_feature)
        
        return logits, global_feature