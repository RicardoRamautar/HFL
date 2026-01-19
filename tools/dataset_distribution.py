import json
import pickle
from pathlib import Path
from collections import defaultdict
import random

def print_manifest(manifest):
    # Print summary
    total_scenes = 0
    print("Number of edges:", len(manifest["edges"]))
    print("Clients per edge:")
    for edge, data in manifest["edges"].items():
        print(f"  {edge}: {len(data['clients'])} clients")

        for client, scenes in data['clients'].items():
            num_scenes = len(scenes['scenes'])
            total_scenes += num_scenes
            print(f"    Client {client} has {num_scenes} scenes")

    print(f"Total number of scenes: {total_scenes}")


def store_manifest(manifest, file):
    with open(file, "w") as f:
        json.dump(manifest, f, indent=2)

def client_per_scene(scenes, logs, data_tokens):
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

    return manifest


def scenes_per_client(scenes, logs, data_tokens, cpl):
    def split_list(lst, num):
        k, m = divmod(len(lst), num)
        return [
            lst[i*k + min(i, m):(i+1)*k + min(i+1, m)]
            for i in range(num)
        ]

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


def client_per_log(scenes, logs, scene_tokens):
    manifest = {"edges": {}}
    client_id = 0
    for log in logs:
        location = log['location']
        log_token = log['token']

        if location not in manifest["edges"]:
            manifest["edges"][location] = {"clients": {}}

        client_scenes = [scene['name'] for scene in scenes if scene['log_token'] == log_token and scene['token'] in scene_tokens]

        if client_scenes:
            cid = f"client_{client_id:04d}"
            manifest["edges"][location]["clients"][cid] = {
                "scenes": client_scenes
            }
            client_id += 1

    return manifest


def main():
    # Paths to nuScenes metadata files
    pkl_file = "/home/rdr/Documents/master_thesis/data/nuscenes/nuscenes_infos_train.pkl"
    scene_file = Path('./scene.json')
    log_file = Path('./log.json')

    output_file = Path("./data_distribution_4.json")

    # Load metadata
    with open(scene_file, "r") as f:
        scenes = json.load(f)
    with open(log_file, "r") as f:
        logs = json.load(f)
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # Extract all scene tokens in dataset
    data_tokens = [info['scene_token'] for info in data['infos']]

    # manifest = client_per_scene(scenes, logs, data_tokens)
    # manifest = scenes_per_client(scenes, logs, data_tokens, 2)
    manifest = client_per_log(scenes, logs, data_tokens)

    store_manifest(manifest, output_file)

    print_manifest(manifest)

if __name__=='__main__':
    main()