import torch
import sys
import lgatr

# 1. Monkey-Patch to allow loading
if 'gatr' not in sys.modules:
    sys.modules['gatr'] = lgatr

# 2. Path to your checkpoint (COPY-PASTE YOUR EXACT PATH HERE)
CKPT_PATH = "/home/jay_agarwal_2022/lorentz-gatr/runs/topt/GATr_7327/models/model_run0_it169999.pt"

def check_keys():
    print(f"--- INSPECTING CHECKPOINT: {CKPT_PATH} ---")
    try:
        state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    except:
        state = torch.load(CKPT_PATH, map_location="cpu")

    # Unwrap if needed
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]

    ckpt_keys = list(state.keys())
    print(f"\n[Checkpoint] Found {len(ckpt_keys)} keys.")
    print("First 5 keys in CHECKPOINT:")
    for k in ckpt_keys[:5]:
        print(f"   '{k}'")

    print("\n" + "="*30 + "\n")

    # 3. Initialize Wrapper to see what it WANTS
    from kd4jets.models.lgatr_wrapper import LGATrWrapper
    model = LGATrWrapper()
    model_keys = list(model.state_dict().keys())
    
    print(f"[Model Wrapper] Expects {len(model_keys)} keys.")
    print("First 5 keys expected by MODEL:")
    for k in model_keys[:5]:
        print(f"   '{k}'")

    print("\n" + "="*30)
    print("COMPARE THE PREFIXES ABOVE!")

if __name__ == "__main__":
    check_keys()