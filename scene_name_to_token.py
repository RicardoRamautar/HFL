import json
from pathlib import Path
from collections import defaultdict

data_root = Path(
    "/tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes/v1.0-trainval"
)
output_file = Path("./scene_name_to_token.json")

def main():
    meta_root = data_root

    # Contains info of each scene
    scene_file = meta_root / "scene.json"

    if not scene_file.exists():
        raise FileNotFoundError("Could not find nuScenes metadata files")

    # Load nuScenes scene metadata
    with open(scene_file, "r") as f:
        scenes = json.load(f)

    scene_name_to_token = {
        scene["token"]: scene["name"]
        for scene in scenes
    }

    # Write output
    with open(output_file, "w") as f:
        json.dump(scene_name_to_token, f, indent=2)

if __name__ == "__main__":
    main()