import json
import pickle
from pathlib import Path
from collections import defaultdict

# data_root = Path(
#     "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes/v1.0-trainval"
# )
output_file = Path("./data_distribution.json")
pth = '/home/rdr/Documents/master_thesis/data/nuscenes/nuscenes_infos_train.pkl'


# # Contains info of each scene
# scene_file = data_root / "scene.json"
# # Contains info of each data collection run
# log_file = data_root / "log.json"
scene_file = Path('./scene.json')
log_file = Path('./log.json')

if not scene_file.exists() or not log_file.exists():
    raise FileNotFoundError("Could not find nuScenes metadata files")

# Load metadata
with open(scene_file, "r") as f:
    scenes = json.load(f)

with open(log_file, "r") as f:
    logs = json.load(f)

with open(pth, "rb") as f:
    data = pickle.load(f)

data_tokens = [data['infos'][i]['scene_token'] for i in range(len(data['infos']))]

# Dict(log token: location)
log_token_to_location = {
    log["token"]: log["location"]
    for log in logs
}

# Group scenes by location
scenes_by_location = defaultdict(list)
for scene in scenes:
    scene_name = scene["name"]          # Scene name
    log_token = scene["log_token"]      # Log token
    location = log_token_to_location[log_token]

    scene_token = scene['token']
    if scene_token in data_tokens:
        scenes_by_location[location].append(scene_name)

# Build manifest: edge -> clients -> scenes
# Every location represents an edge, every scene represents a client
manifest = {"edges": {}}
client_id = 0

for location, scene_names in sorted(scenes_by_location.items()):
    manifest["edges"][location] = {"clients": {}}

    for scene_name in sorted(scene_names):
        cid = f"client_{client_id:04d}"
        manifest["edges"][location]["clients"][cid] = {
            "scenes": [scene_name]
        }
        client_id += 1

# Write output
with open(output_file, "w") as f:
    json.dump(manifest, f, indent=2)

# Print summary (important sanity check)
print("Manifest written to:", output_file)
print("Number of edges:", len(manifest["edges"]))
print("Clients per edge:")
for edge, data in manifest["edges"].items():
    print(f"  {edge}: {len(data['clients'])} clients")


