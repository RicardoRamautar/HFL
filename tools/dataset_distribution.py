# import json
# import pickle
# from pathlib import Path
# from collections import defaultdict

# output_file = Path("./data_distribution.json")
# pth = '/home/rdr/Documents/master_thesis/data/nuscenes/nuscenes_infos_train.pkl'


# # # Contains info of each scene
# scene_file = Path('./scene.json')
# log_file = Path('./log.json')

# if not scene_file.exists() or not log_file.exists():
#     raise FileNotFoundError("Could not find nuScenes metadata files")

# # Load metadata
# with open(scene_file, "r") as f:
#     scenes = json.load(f)

# with open(log_file, "r") as f:
#     logs = json.load(f)

# with open(pth, "rb") as f:
#     data = pickle.load(f)

# data_tokens = [data['infos'][i]['scene_token'] for i in range(len(data['infos']))]

# # Dict(log token: location)
# log_token_to_location = {
#     log["token"]: log["location"]
#     for log in logs
# }

# # Group scenes by location
# scenes_by_location = defaultdict(list)
# for scene in scenes:
#     scene_name = scene["name"]          # Scene name
#     log_token = scene["log_token"]      # Log token
#     location = log_token_to_location[log_token]

#     scene_token = scene['token']
#     if scene_token in data_tokens:
#         scenes_by_location[location].append(scene_name)

# # Build manifest: edge -> clients -> scenes
# # Every location represents an edge, every scene represents a client
# manifest = {"edges": {}}
# client_id = 0

# for location, scene_names in sorted(scenes_by_location.items()):
#     manifest["edges"][location] = {"clients": {}}

#     for scene_name in sorted(scene_names):
#         cid = f"client_{client_id:04d}"
#         manifest["edges"][location]["clients"][cid] = {
#             "scenes": [scene_name]
#         }
#         client_id += 1

# # Write output
# with open(output_file, "w") as f:
#     json.dump(manifest, f, indent=2)

# # Print summary (important sanity check)
# print("Manifest written to:", output_file)
# print("Number of edges:", len(manifest["edges"]))
# print("Clients per edge:")
# for edge, data in manifest["edges"].items():
#     print(f"  {edge}: {len(data['clients'])} clients")


import json
import pickle
from pathlib import Path
from collections import defaultdict
import random

# Paths to nuScenes metadata files
pkl_file = "/tudelft.net/staff-umbrella/rdramautar/datasets/nuscenes_infos/nuscenes_infos_train.pkl"
scene_file = Path('/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes/v1.0-trainval/scene.json')
log_file = Path('/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes/v1.0-trainval/log.json')

output_file = Path("/tudelft.net/staff-umbrella/rdramautar/HFL/data/data_distribution_2.json")

num_clients_per_location = 2
random.seed(42)


# Load metadata
with open(scene_file, "r") as f:
    scenes = json.load(f)
with open(log_file, "r") as f:
    logs = json.load(f)
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

# Extract all scene tokens in dataset
data_tokens = [info['scene_token'] for info in data['infos']]

# Dict(log token: location)
log_token_to_location = {
    log["token"]: log["location"]
    for log in logs
}


# Group scenes by location
scenes_by_location = defaultdict(list)
for scene in scenes:
    scene_name = scene["name"]
    log_token = scene["log_token"]
    location = log_token_to_location[log_token]

    scene_token = scene['token']
    if scene_token in data_tokens:
        scenes_by_location[location].append(scene_name)

def assign_data(scenes_by_location, cpl):
    def split_list(lst, num):
        k, m = divmod(len(lst), num)
        return [
            lst[i*k + min(i, m):(i+1)*k + min(i+1, m)]
            for i in range(num)
        ]

    manifest = {"edges": {}}
    client_id = 0
    for location, scene_names in sorted(scenes_by_location.items()):
        manifest["edges"][location] = {"clients": {}}

        scene_names = scene_names.copy()
        random.shuffle(scene_names)

        scenes_per_client = split_list(scene_names, cpl)
        for scenes in scenes_per_client:
            cid = f"client_{client_id:04d}"
            manifest["edges"][location]["clients"][cid] = {
                "scenes": scenes
            }
            client_id += 1

    return manifest

manifest = assign_data(scenes_by_location, num_clients_per_location)

    # Write output
with open(output_file, "w") as f:
    json.dump(manifest, f, indent=2)

all_scenes = []
for edge in manifest["edges"].values():
    for client in edge["clients"].values():
        all_scenes.extend(client["scenes"])

assert len(all_scenes) == len(set(all_scenes)), "Duplicate scenes detected"

# Print summary
total_scenes = 0
print("Manifest written to:", output_file)
print("Number of edges:", len(manifest["edges"]))
print("Clients per edge:")
for edge, data in manifest["edges"].items():
    print(f"  {edge}: {len(data['clients'])} clients")

    for client, scenes in data['clients'].items():
        num_scenes = len(scenes['scenes'])
        total_scenes += num_scenes
        print(f"    Client {client} has {num_scenes}")

print(f"Total number of scenes: {total_scenes}")

