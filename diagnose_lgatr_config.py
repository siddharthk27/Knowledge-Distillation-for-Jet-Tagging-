"""
Reverse-engineer LGATr configuration from checkpoint weight shapes.
Run this to find the exact hyperparameters needed.
"""
import torch
import sys
import sys
import lgatr

# Monkey-patch old module name used in checkpoint
sys.modules["gatr"] = lgatr


# Update this path to your checkpoint
CHECKPOINT_PATH = "/home/jay_agarwal_2022/lorentz-gatr/runs/topt/GATr_7327/models/model_run0_it169999.pt"

def infer_config_from_checkpoint(ckpt_path):
    """Infer LGATr configuration from checkpoint tensor shapes."""
    
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Extract model weights
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove 'net.' prefix
    clean_dict = {}
    for k, v in state_dict.items():
        new_k = k[4:] if k.startswith("net.") else k
        clean_dict[new_k] = v
    
    print("=" * 80)
    print("LGATR CONFIGURATION INFERENCE")
    print("=" * 80)
    
    # 1. Input channels from linear_in
    if "linear_in.weight" in clean_dict:
        in_mv_ch = clean_dict["linear_in.weight"].shape[0]
        print(f"\n✓ in_mv_channels: {in_mv_ch}")
    
    if "linear_in.s2s.weight" in clean_dict:
        in_s_ch = clean_dict["linear_in.s2s.weight"].shape[1]
        print(f"✓ in_s_channels: {in_s_ch}")
    else:
        in_s_ch = 0
        print(f"✓ in_s_channels: {in_s_ch} (no s2s layer found)")
    
    # 2. Hidden channels from first block
    if "blocks.0.attention.qkv_module.in_linear.weight" in clean_dict:
        qkv_weight = clean_dict["blocks.0.attention.qkv_module.in_linear.weight"]
        # Shape: [qkv_total, hidden_mv, 10]
        # qkv_total = 3 * num_heads * head_dim_mv
        qkv_total = qkv_weight.shape[0]
        hidden_mv = qkv_weight.shape[1]
        print(f"✓ hidden_mv_channels: {hidden_mv}")
        
        # Infer num_heads (common values: 4, 8, 16)
        # Assume head_dim_mv matches hidden_mv (common practice)
        if qkv_total == 3 * hidden_mv:
            num_heads = 1
        else:
            # Try common head counts
            for heads in [4, 8, 16, 32]:
                if qkv_total % (3 * heads) == 0:
                    head_dim = qkv_total // (3 * heads)
                    if head_dim <= hidden_mv:  # Sanity check
                        num_heads = heads
                        break
        print(f"✓ num_heads (inferred): {num_heads}")
    
    if "blocks.0.attention.qkv_module.in_linear.s2s.weight" in clean_dict:
        hidden_s = clean_dict["blocks.0.attention.qkv_module.in_linear.s2s.weight"].shape[0] // 3
        print(f"✓ hidden_s_channels: {hidden_s}")
    
    # 3. MLP hidden dimension
    if "blocks.0.mlp.layers.0.linear_left.weight" in clean_dict:
        mlp_hidden_mv = clean_dict["blocks.0.mlp.layers.0.linear_left.weight"].shape[0]
        print(f"✓ mlp_hidden_mv (first layer): {mlp_hidden_mv}")
    
    if "blocks.0.mlp.layers.0.linear_left.s2mvs.weight" in clean_dict:
        mlp_hidden_s = clean_dict["blocks.0.mlp.layers.0.linear_left.s2mvs.weight"].shape[0] // 2
        print(f"✓ mlp_hidden_s (first layer): {mlp_hidden_s}")
    
    # 4. Number of blocks
    num_blocks = 0
    for key in clean_dict.keys():
        if key.startswith("blocks."):
            block_num = int(key.split(".")[1])
            num_blocks = max(num_blocks, block_num + 1)
    print(f"✓ num_blocks: {num_blocks}")
    
    # 5. Output channels from linear_out (if exists)
    if "linear_out.weight" in clean_dict:
        out_mv = clean_dict["linear_out.weight"].shape[0]
        print(f"✓ out_mv_channels: {out_mv}")
    else:
        print(f"✓ out_mv_channels: (no linear_out, probably equals hidden_mv_channels)")
    
    if "linear_out.s2s.weight" in clean_dict:
        out_s = clean_dict["linear_out.s2s.weight"].shape[0]
        print(f"✓ out_s_channels: {out_s}")
    else:
        print(f"✓ out_s_channels: None (no separate output)")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 80)
    
    config_code = f"""
# Option 1: Using LGATRConfig (if available)
from lgatr.gatr_config import LGATRConfig

cfg = LGATRConfig(
    in_mv_channels={in_mv_ch},
    out_mv_channels={in_mv_ch},  # Usually same as input
    hidden_mv_channels={hidden_mv},
    in_s_channels={in_s_ch},
    out_s_channels=None,  # Will default to hidden_s_channels
    hidden_s_channels={hidden_s},
    num_blocks={num_blocks},
    num_heads={num_heads},
    dropout_prob=0.0
)
encoder = LGATr(cfg)

# Option 2: Using direct kwargs
from lgatr import LGATr, SelfAttentionConfig, MLPConfig

encoder = LGATr(
    in_mv_channels={in_mv_ch},
    out_mv_channels={in_mv_ch},
    hidden_mv_channels={hidden_mv},
    in_s_channels={in_s_ch},
    out_s_channels=None,
    hidden_s_channels={hidden_s},
    num_blocks={num_blocks},
    attention=SelfAttentionConfig(
        num_heads={num_heads},
        dropout_prob=0.0
    ),
    mlp=MLPConfig(
        dropout_prob=0.0
    )
)
"""
    
    print(config_code)
    print("=" * 80)
    
    return {
        "in_mv_channels": in_mv_ch,
        "in_s_channels": in_s_ch,
        "hidden_mv_channels": hidden_mv,
        "hidden_s_channels": hidden_s,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
    }

if __name__ == "__main__":
    config = infer_config_from_checkpoint(CHECKPOINT_PATH)