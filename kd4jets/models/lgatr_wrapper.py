import torch
import torch.nn as nn
from lgatr import LGATr, LGATRConfig
from lgatr.interface import embed_point

class LGATrWrapper(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()

        # 1. Exact Config from config.yaml
        cfg = LGATRConfig(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=16,  #
            hidden_s_channels=32,   #
            num_blocks=12,          #
            num_heads=8,            #
            dropout=0.0,
        )

        self.encoder = LGATr(cfg)

        # 2. Classifier Head
        # Config says 'mean_aggregation: false', so it uses the global token.
        self.classifier = nn.Linear(cfg.hidden_s_channels, 2)

        # 3. Load Weights
        if checkpoint_path:
            self._load_weights(checkpoint_path)

    def _load_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Clean prefix 'model.net.' -> matches lgatr package structure
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("model.net.", "").replace("module.", "")
            clean_state_dict[new_k] = v

        # strict=False handles the classifier head naming mismatch
        self.load_state_dict(clean_state_dict, strict=False)
        print("LGATr Teacher weights loaded successfully.")

    def forward(self, batch_dict):
        p4 = batch_dict["Pmu"]       # [Batch, N, 4]
        mask = batch_dict["atom_mask"] # [Batch, N]
        batch_size = p4.shape[0]
        device = p4.device

        # --- STEP 1: Beam Token Injection (Critical for 'beam_token: true') ---
        # Create a beam particle (Energy=1, px=0, py=0, pz=1)
        # This matches 'beam_reference: xyplane' standard behavior
        beam_p4 = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device).view(1, 1, 4)
        beam_p4 = beam_p4.expand(batch_size, -1, -1) # [Batch, 1, 4]

        # Append beam to input
        p4_in = torch.cat([beam_p4, p4], dim=1)      # [Batch, N+1, 4]

        # Update Mask (Beam is always real, so mask=True)
        beam_mask = torch.ones((batch_size, 1), device=device, dtype=mask.dtype)
        mask_in = torch.cat([beam_mask, mask], dim=1) # [Batch, N+1]

        # --- STEP 2: Embedding ---
        # embed_point expects (t, x, y, z).
        # KD4Jets provides (E, px, py, pz) which is equivalent.
        x_mv = embed_point(p4_in).unsqueeze(2) # [Batch, N+1, 1, 16]

        # Zero scalars
        x_s = torch.zeros(batch_size, p4_in.shape[1], 1, device=device)

        # --- STEP 3: Forward & Readout ---
        out_mv, out_s = self.encoder(x_mv, x_s, attention_mask=mask_in)

        # Readout: The config uses 'mean_aggregation: false' and 'num_global_tokens: 1'.
        # The LGATr package usually appends the global token at the END.
        # We take the last token as the graph representation.
        global_token_feature = out_s[:, -1, :]

        logits = self.classifier(global_token_feature)

        return logits, global_token_feature
