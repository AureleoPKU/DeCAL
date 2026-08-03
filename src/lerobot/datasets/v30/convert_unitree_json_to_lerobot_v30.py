r"""Convert supported raw JSON robot data to LeRobot v3.0.

Only ``Ur5_Sharpa`` is currently implemented. Supporting another robot requires
adding its state, action, tactile, camera, and image schema to this converter.

Example:
    python src/lerobot/datasets/v30/convert_unitree_json_to_lerobot_v30.py \
        --raw-dir /path/to/raw/data \
        --repo-id your_dataset_repo_id \
        --robot-type Ur5_Sharpa
"""

import os
import cv2
import tqdm
import tyro
import json
import glob
import dataclasses
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Literal, List, Dict, Optional

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class JsonDataset:
    def __init__(self, data_dirs: Path, robot_type: str, action_as_next_state: bool = True) -> None:
        """
        Initialize the dataset for loading and processing JSON robot data.

        Args:
            data_dirs: Path to directory containing training data
            action_as_next_state: If True, use state[t+1] as action[t]; if False, use separate "actions" key from JSON.
        """
        assert data_dirs is not None, "Data directory cannot be None"
        self.data_dirs = data_dirs
        self.json_file = "data.json"
        self.action_as_next_state = action_as_next_state

        if robot_type == "Ur5_Sharpa":
            self.json_state_data_name = ["left_arm", "right_arm", "left_hand", "right_hand"]
            self.json_action_data_name = ["left_arm", "right_arm", "left_hand", "right_hand"]
            self.camera_to_image_key = {
                "color_0": "cam_front",
                "color_1": "cam_left_wrist",
                "color_2": "cam_right_wrist",
            }
        else:
            raise NotImplementedError(
                f"Unsupported robot_type {robot_type!r}. Add its JSON field mappings to JsonDataset."
            )

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()

    def _init_paths(self) -> None:
        """Initialize episode and task paths - flexible version."""
        
        self.episode_paths = []
        self.task_paths = []

        def find_episodes_in_directory(directory):
            """Return all episode directories directly under a directory."""
            episodes = []
            episode_patterns = glob.glob(os.path.join(directory, "episode_*"))
            for path in episode_patterns:
                if os.path.isdir(path):
                    episodes.append(path)
            return episodes

        # Support both raw_dir/episode_* and raw_dir/task/episode_* layouts.
        direct_episodes = find_episodes_in_directory(self.data_dirs)
        
        if direct_episodes:
            self.task_paths.append(self.data_dirs)
            self.episode_paths.extend(direct_episodes)
        else:
            for potential_task in glob.glob(os.path.join(self.data_dirs, "*")):
                if os.path.isdir(potential_task):
                    task_episodes = find_episodes_in_directory(potential_task)
                    if task_episodes:
                        self.task_paths.append(potential_task)
                        self.episode_paths.extend(task_episodes)

        if not self.episode_paths:
            raise ValueError(f"No episode directories found in: {self.data_dirs}")

        self.episode_paths = sorted(self.episode_paths)
        
        print(f"Data source: {self.data_dirs}")
        print(f"Found {len(self.task_paths)} tasks: {[os.path.basename(p) for p in self.task_paths]}")
        print(f"Found {len(self.episode_paths)} episodes")

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)

    def _init_cache(self) -> List:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            with open(json_path, "r", encoding="utf-8") as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")

        return self.episodes_data_cached

    def _extract_data(self, episode_data: Dict, key: str, parts: List[str]) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.

        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')

        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        for sample_data in episode_data["data"]:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                if part in sample_data[key] and sample_data[key][part] is not None:
                    part_data = sample_data[key][part]

                    # Store only joint positions in float32.
                    if "qpos" in part_data:
                        qpos = np.array(part_data["qpos"], dtype=np.float32)
                        data_array = np.concatenate([data_array, qpos.astype(np.float32, copy=False)])

                    elif isinstance(part_data, list):
                        array_data = np.array(part_data, dtype=np.float32)
                        data_array = np.concatenate([data_array, array_data])

            result.append(data_array.astype(np.float32, copy=False))
        return np.array(result, dtype=np.float32)
    
    def _extract_tactile(self, episode_data: Dict) -> np.ndarray:
        """
        Extract tactile data from episode dictionary.

        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')

        Returns:
            Concatenated numpy array of the requested data (float32)
        """
        result = []
        parts = ["left_tactile", "right_tactile"]
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        for sample_data in episode_data["data"]:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                if part in sample_data["states"]["tactile"] and sample_data["states"]["tactile"][part] is not None:
                    for finger in fingers:
                        part_data = sample_data["states"]["tactile"][part][finger]
                        array_data = np.array(part_data, dtype=np.float32)
                        data_array = np.concatenate([data_array, array_data])

            result.append(data_array.astype(np.float32, copy=False))
        return np.array(result, dtype=np.float32)

    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)

        keys = episode_data["data"][0]["colors"].keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue

            for sample_data in episode_data["data"]:
                relative_path = sample_data["colors"].get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images
    
    def _parse_tactiles_deform(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack tactile deform images for a given key."""

        images = defaultdict(list)

        fingers = ['left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky',
                'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'] 
        
        for finger in fingers:
            tactile_key = f"{finger}_deform"
            for sample_data in episode_data["data"]:
                relative_path = sample_data["tactile_images_deform"].get(f"{finger}_deform_0000")
                if not relative_path:
                    raise RuntimeError(f"No tactile image path for finger: {finger}")
                
                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Tactile image path does not exist: {image_path}")


                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise RuntimeError(f"Failed to read tactile image: {image_path}")
                # image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
                # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                images[tactile_key].append(image_rgb)

        return images

    def _parse_tactiles_raw(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack tactile raw images for a given key."""

        images = defaultdict(list)

        fingers = ['left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky',
                'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'] 
        
        for finger in fingers:
            tactile_key = f"{finger}_raw"
            for sample_data in episode_data["data"]:
                relative_path = sample_data["tactile_images_raw"].get(f"{finger}_raw_0000")
                if not relative_path:
                    raise RuntimeError(f"No tactile image path for finger: {finger}")
                
                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Tactile image path does not exist: {image_path}")


                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise RuntimeError(f"Failed to read tactile image: {image_path}")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                images[tactile_key].append(image_rgb)

        return images

    def get_item(
        self,
        index: Optional[int] = None,
    ) -> Dict:
        """Get a training sample from the dataset."""

        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        # Load state and action data
        state = self._extract_data(episode_data, "states", self.json_state_data_name)
        tactile = self._extract_tactile(episode_data)
        if self.action_as_next_state:
            # action[t] = state[t+1]; last frame has no next state, so we have N-1 frames
            action = state[1:].astype(np.float32, copy=False)
            state = state[:-1]
            tactile = tactile[:-1]
            episode_length = len(state)
        else:
            action = self._extract_data(episode_data, "actions", self.json_action_data_name)
            episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else action.shape[0]
        tactile_dim = tactile.shape[1] if len(tactile.shape) == 2 else tactile.shape[0]

        # Load camera images
        cameras = self._parse_images(file_path, episode_data)
        tactile_deforms = self._parse_tactiles_deform(file_path, episode_data)
        tactile_raws = self._parse_tactiles_raw(file_path, episode_data)

        if self.action_as_next_state:
            # Trim to N-1 frames to match state/action (drop last frame)
            cameras = {k: v[:-1] for k, v in cameras.items()}
            tactile_deforms = {k: v[:-1] for k, v in tactile_deforms.items()}
            tactile_raws = {k: v[:-1] for k, v in tactile_raws.items()}
        
        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        tactile_deform_height, tactile_deform_width = next(img for imgs in tactile_deforms.values() if imgs for img in imgs).shape[:2]
        tactile_raw_height, tactile_raw_width = next(img for imgs in tactile_raws.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            "camera_names": list(cameras.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "tactile_deform_height": tactile_deform_height, 
            "tactile_deform_width": tactile_deform_width,
            "tactile_raw_height": tactile_raw_height,
            "tactile_raw_width": tactile_raw_width,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "tactile_dim": tactile_dim,
        }

        return {
            "episode_index": index,
            "episode_length": episode_length,
            "state": state,
            "action": action,
            "tactile": tactile,
            "cameras": cameras,
            "tactile_deforms": tactile_deforms,
            "tactile_raws": tactile_raws,
            "data_cfg": data_cfg,
        }

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    if robot_type == "Ur5_Sharpa":
        state_action_joint_dim = 6 + 6 + 22 + 22
        state_action_joint_names = ["left_arm", "right_arm", "left_hand", "right_hand"]
        state_tactile_dim = 30 + 30
        state_tactile_names = ["left_tactile", "right_tactile"]
    else:
        raise NotImplementedError(
            f"Unsupported robot_type {robot_type!r}. Add its dataset feature schema to "
            "create_empty_dataset."
        )
        
    cameras = [
        "cam_front",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    
    tactile_deform = [
        "left_thumb_deform",
        "left_index_deform",
        "left_middle_deform",
        "left_ring_deform",
        "left_pinky_deform",
        "right_thumb_deform",
        "right_index_deform",
        "right_middle_deform",
        "right_ring_deform",
        "right_pinky_deform",
    ]

    tactile_raw = [
        "left_thumb_raw",
        "left_index_raw",
        "left_middle_raw",
        "left_ring_raw",
        "left_pinky_raw",
        "right_thumb_raw",
        "right_index_raw",
        "right_middle_raw",
        "right_ring_raw",
        "right_pinky_raw",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_action_joint_dim,),
            "names": state_action_joint_names,
        },
        "observation.tactile": {
            "dtype": "float32",
            "shape": (state_tactile_dim,),
            "names": state_tactile_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (state_action_joint_dim,),
            "names": state_action_joint_names,
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (state_action_joint_dim,),
            "names": state_action_joint_names,
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (state_action_joint_dim,),
            "names": state_action_joint_names,
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }
        
    for tactile in tactile_deform:
        features[f"observation.tactile_deforms.{tactile}"] = {
            "dtype": mode,
            "shape": (240, 240, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }
    
    for tactile in tactile_raw:
        features[f"observation.tactile_raws.{tactile}"] = {
            "dtype": mode,
            "shape": (240, 320, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }
        
    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
    custom_task_description: str = "",
    action_as_next_state: bool = False,
) -> LeRobotDataset:
    json_dataset = JsonDataset(raw_dir, robot_type, action_as_next_state=action_as_next_state)
    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        episode_length = episode["episode_length"]
        tactile_deforms = episode["tactile_deforms"]
        tactile_raws = episode["tactile_raws"]
        tactile = episode["tactile"]
        task = custom_task_description

        num_frames = episode_length
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "observation.tactile": tactile[i],
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[i]
                
            for tactile_image, img_array in tactile_deforms.items():
                frame[f"observation.tactile_deforms.{tactile_image}"] = img_array[i]
            
            for tactile_image, img_array in tactile_raws.items():
                frame[f"observation.tactile_raws.{tactile_image}"] = img_array[i]
            frame["task"] = task
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    custom_task_description: str = "",
    action_as_next_state: bool = False,
):
    print(f"action_as_next_state: {action_as_next_state}")
    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
        custom_task_description=custom_task_description,
        action_as_next_state=action_as_next_state,
    )

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


def local_push_to_hub(
    repo_id: str,
    root_path: Path,
):
    dataset = LeRobotDataset(repo_id=repo_id, root=root_path)
    dataset.push_to_hub(upload_large_folder=True)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot)
