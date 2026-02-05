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


def iid_day_night(num_edges, num_clients_per_edge, scenes, logs, train_scene_tokens):
    categorized_scenes = {}
    for log in logs:
        location = log['location']
        log_token = log['token']

        if location not in categorized_scenes:
            categorized_scenes[location] = {
                'day': [],
                'night': []
            }

        for scene in scenes:
            if (scene['log_token'] == log_token) and (scene['token'] in train_scene_tokens):
                description = scene['description'].lower()
                if 'night' in description:
                    categorized_scenes[location]['night'].append(scene)
                else:
                    categorized_scenes[location]['day'].append(scene)

    num_clients = num_edges*num_clients_per_edge
    client_scenes = {i: [] for i in range(num_clients)}

    for location, scenes_per_cat in categorized_scenes.items():
        for timeofday in ['day', 'night']:
            scenes_list = scenes_per_cat[timeofday]
            random.shuffle(scenes_list)

            for i, scene in enumerate(scenes_list):
                client_id = i % num_clients
                client_scenes[client_id].append(scene)

    clients = [i for i in range(num_clients)]
    manifest = {"edges": {}}
    for e in range(num_edges):
        edge_name = f'edge_{e}'
        manifest["edges"][edge_name] = {"clients": {}}

        for _ in range(num_clients_per_edge):
            i = random.choice(clients)
            client_name = f'client_{i}'

            scenes = [scene['name'] for scene in client_scenes[i]]

            manifest["edges"][edge_name]["clients"][client_name] = {
                "scenes": scenes
            }

            clients.remove(i)

    for i, scenes in client_scenes.items():
        rainy_scenes = 0
        night_scenes = 0
        day_scenes = 0
        total_scenes = 0

        for scene in scenes:
            description = scene['description'].lower()
            if 'rain' in description:
                rainy_scenes += 1
            if 'night' in description:
                night_scenes += 1
            if ('rain' not in description) and ('night' not in description):
                day_scenes += 1

            total_scenes += 1

        print(f"Client {i}: \n   Total scenes: {total_scenes} \n   Rainy scenes: {rainy_scenes} \n   Nights: {night_scenes} \n   Days: {day_scenes}")

    return manifest


def main():
    # Paths to nuScenes metadata files
    pkl_file = "/home/rdr/Documents/master_thesis/data/nuscenes/nuscenes_infos_train.pkl"
    scene_file = Path('./scene.json')
    log_file = Path('./log.json')

    output_file = Path("./data_distribution_5.json")

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
    # manifest = client_per_log(scenes, logs, data_tokens)
    manifest = iid_day_night(
        num_edges = 2,
        num_clients_per_edge = 2,
        scenes = scenes,
        logs = logs,
        train_scene_tokens = data_tokens
    )

    store_manifest(manifest, output_file)

    print(manifest)

if __name__ == '__main__':
    main()
