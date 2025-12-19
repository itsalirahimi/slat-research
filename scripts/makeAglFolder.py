import json
import numpy as np
import os

pre_path      = "data/usegeo_3"
json_path     = f"{pre_path}/data.json"
out_dir       = f"{pre_path}/m_agl"

os.makedirs(out_dir, exist_ok=True)

# Read JSON file
with open(json_path, "r") as f:
    data = json.load(f)

# Iterate over entries
for entry in data:
    name = entry["name"]
    agl = entry["agl"]

    # Build output filename
    out_file = os.path.join(out_dir, f"{name}.npy")

    # Save scalar agl value
    np.save(out_file, np.array(agl))

    print(f"Saved {out_file} with agl={agl}")
