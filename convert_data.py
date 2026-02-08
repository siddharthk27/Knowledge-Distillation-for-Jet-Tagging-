import numpy as np
import h5py
import os

# --- Config ---
INPUT_FILE = "data/toptagging_full.npz" 
OUTPUT_DIR = "data"

def save_to_h5(filename, p4, labels):
    print(f"  Saving {filename} | Events: {len(labels)}")
    with h5py.File(filename, 'w') as f:
        # --- CORE FEATURES ---
        # 1. Pmu: (N, 200, 4) -> (E, px, py, pz)
        f.create_dataset("Pmu", data=p4, compression="gzip")
        
        # 2. is_signal: (N,)
        f.create_dataset("is_signal", data=labels, compression="gzip")
        
        # 3. truth / label (Aliases for safety)
        # Some legacy code looks for 'truth' or 'label' instead of 'is_signal'
        f.create_dataset("truth", data=labels, compression="gzip")
        f.create_dataset("label", data=labels, compression="gzip")
        
        # --- MASKS & COUNTS ---
        # 4. atom_mask: (N, 200) - True if particle exists
        # Calculate by checking if particle energy > 0
        mask = np.abs(p4[:, :, 0]) > 1e-5
        f.create_dataset("atom_mask", data=mask.astype(bool), compression="gzip")
        f.create_dataset("mask", data=mask.astype(bool), compression="gzip") # <--- Added Alias
        # 5. Nobj (CRITICAL MISSING KEY): Number of particles per jet
        # The dataloader uses this to know how many particles to read
        n_obj = np.sum(mask, axis=1).astype(np.int32)
        f.create_dataset("Nobj", data=n_obj, compression="gzip")
        
        # --- DERIVED JET KINEMATICS ---
        # 6. jet_p4: Total momentum of the jet (Sum of particles)
        jet_p4 = np.sum(p4, axis=1)
        f.create_dataset("jet_p4", data=jet_p4, compression="gzip")
        
        # 7. mass: Invariant mass of the jet
        # m^2 = E^2 - (px^2 + py^2 + pz^2)
        msq = jet_p4[:, 0]**2 - np.sum(jet_p4[:, 1:]**2, axis=1)
        # Clip to 0 to avoid NaNs from numerical noise
        mass = np.sqrt(np.maximum(msq, 0))
        f.create_dataset("mass", data=mass, compression="gzip")
        
        # 8. jet_pt, jet_eta, jet_phi (Optional convenience keys)
        pt = np.sqrt(jet_p4[:, 1]**2 + jet_p4[:, 2]**2)
        f.create_dataset("jet_pt", data=pt, compression="gzip")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"Loading {INPUT_FILE}...")
    raw = np.load(INPUT_FILE)
    keys = list(raw.keys())
    
    # --- Robust Key Detection ---
    if 'kinematics_train' in keys:
        print("-> Detected split format (kinematics_train/val/test)")
        X_train, y_train = raw['kinematics_train'], raw['labels_train']
        X_val,   y_val   = raw['kinematics_val'],   raw['labels_val']
        X_test,  y_test  = raw['kinematics_test'],  raw['labels_test']
    
    elif 'x' in keys:
        print("-> Detected monolithic format (x/y)")
        X = raw['x']
        y = raw['y']
        X_train, y_train = X[:1200000], y[:1200000]
        X_val,   y_val   = X[1200000:1600000], y[1200000:1600000]
        X_test,  y_test  = X[1600000:], y[1600000:]
    else:
        print("❌ Error: Could not understand .npz keys.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save all splits
    save_to_h5(os.path.join(OUTPUT_DIR, "train.h5"), X_train, y_train)
    save_to_h5(os.path.join(OUTPUT_DIR, "val.h5"), X_val, y_val)
    save_to_h5(os.path.join(OUTPUT_DIR, "test.h5"), X_test, y_test)
    
    print("✅ Conversion Complete. Data is now compatible with LorentzNet.")

if __name__ == "__main__":
    main()