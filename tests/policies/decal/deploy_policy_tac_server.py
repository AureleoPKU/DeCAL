#!/usr/bin/env python
"""Serve a DeCAL policy over HTTP."""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from torchvision.utils import save_image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import cast_stats_to_numpy, load_json
from lerobot.policies.DeCAL import DeCALConfig, DeCALPolicy
from lerobot.policies.DeCAL.transform_decal import DeCALProcessorTransformFn
from lerobot.transforms.core import NormalizeTransformFn, UnNormalizeTransformFn
from lerobot.utils.constants import ACTION, OBS_STATE, PRETRAINED_MODEL_DIR

json_numpy.patch()

CAMERA_KEYS = (
    ("video.last_cam_right_high", "video.cam_right_high"),
    ("video.last_cam_left_wrist", "video.cam_left_wrist"),
    ("video.last_cam_right_wrist", "video.cam_right_wrist"),
)
FINGER_NAMES = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
)
STATE_KEYS = (
    "state.left_arm",
    "state.right_arm",
    "state.left_hand",
    "state.right_hand",
)
STAT_NAMES = ("min", "max", "mean", "std")


@dataclass
class DeployConfig:
    checkpoint: Path
    robot_type: str | None = None
    host: str = "0.0.0.0"
    port: int = 8787
    recon_dir: Path = Path("recon_images")


def _resolve_checkpoint(path: Path) -> Path:
    checkpoint = path.expanduser().resolve()
    model_dir = checkpoint / PRETRAINED_MODEL_DIR
    if model_dir.is_dir():
        checkpoint = model_dir
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    return checkpoint


def _load_action_mode(checkpoint: Path) -> str:
    config_path = checkpoint / "train_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    train_config = load_json(config_path)
    action_mode = train_config.get("dataset", {}).get("action_mode")
    if action_mode not in {"abs", "delta"}:
        raise ValueError(f"Invalid dataset action_mode in {config_path}: {action_mode!r}")
    return action_mode


def _load_stats(checkpoint: Path, robot_type: str | None) -> dict[str, Any]:
    stats_path = checkpoint / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {stats_path}")

    raw_stats = load_json(stats_path)
    if "action" in raw_stats:
        stats = raw_stats
    else:
        candidates = {
            name: values
            for name, values in raw_stats.items()
            if isinstance(values, dict) and "action" in values
        }
        if not candidates:
            raise ValueError(f"No robot statistics found in {stats_path}")
        if robot_type is not None:
            if robot_type not in candidates:
                raise KeyError(
                    f"robot_type={robot_type!r} not found in {stats_path}; "
                    f"available values: {sorted(candidates)}"
                )
            stats = candidates[robot_type]
        elif len(candidates) == 1:
            stats = next(iter(candidates.values()))
        else:
            raise ValueError(
                f"Multiple robot types found in {stats_path}: {sorted(candidates)}. "
                "Specify --robot_type."
            )

    required_keys = {ACTION, OBS_STATE, "observation.tactile"}
    missing_keys = required_keys.difference(stats)
    if missing_keys:
        raise KeyError(f"Missing statistics in {stats_path}: {sorted(missing_keys)}")
    return cast_stats_to_numpy(stats)


def _transform_stats(stats: dict[str, Any], key: str) -> dict[str, np.ndarray]:
    return {name: np.asarray([stats[key][name]]) for name in STAT_NAMES}


def _tactile_force_scale(stats: dict[str, Any]) -> float:
    tactile_stats = stats["observation.tactile"]
    extrema = np.concatenate(
        [
            np.asarray(tactile_stats["min"], dtype=np.float32).reshape(-1),
            np.asarray(tactile_stats["max"], dtype=np.float32).reshape(-1),
        ]
    )
    return max(float(np.max(np.abs(extrema))), 1e-6)


class DeCALServer:
    def __init__(self, cfg: DeployConfig) -> None:
        checkpoint = _resolve_checkpoint(cfg.checkpoint)
        stats = _load_stats(checkpoint, cfg.robot_type)
        action_mode = _load_action_mode(checkpoint)

        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, DeCALConfig):
            raise TypeError(f"Expected DeCALConfig, got {type(config).__name__}")

        config.compile_model = False
        config.compile_mode = "reduce-overhead"

        self.processor = DeCALProcessorTransformFn()
        self.policy = DeCALPolicy.from_pretrained(
            config=config,
            pretrained_name_or_path=checkpoint,
            local_files_only=True,
        )
        self.policy.cuda()
        self.policy.to(torch.float32)
        self.policy.eval()

        self.action_mode = action_mode
        self.chunk_size = config.chunk_size
        self.tactile_force_scale = _tactile_force_scale(stats)
        self.action_unnormalizer = UnNormalizeTransformFn(
            selected_keys=[ACTION],
            mode="mean_std",
            norm_stats={ACTION: _transform_stats(stats, ACTION)},
        )
        self.state_unnormalizer = UnNormalizeTransformFn(
            selected_keys=[OBS_STATE],
            mode="mean_std",
            norm_stats={OBS_STATE: _transform_stats(stats, OBS_STATE)},
        )
        self.state_normalizer = NormalizeTransformFn(
            selected_keys=[OBS_STATE],
            mode="mean_std",
            norm_stats={OBS_STATE: _transform_stats(stats, OBS_STATE)},
        )

        self.recon_dir = cfg.recon_dir.expanduser().resolve()
        self.recon_dir.mkdir(parents=True, exist_ok=True)
        self.request_index = 0

    @staticmethod
    def _frame_stack(previous: Any, current: Any, scale: float = 1.0) -> torch.Tensor:
        previous_tensor = torch.as_tensor(previous, dtype=torch.float32) / scale
        current_tensor = torch.as_tensor(current, dtype=torch.float32) / scale
        return torch.stack((previous_tensor, current_tensor, torch.rand_like(current_tensor)))

    def _build_sample(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = torch.cat(
            [torch.as_tensor(observation[key], dtype=torch.float32) for key in STATE_KEYS]
        )
        state = self.state_normalizer({OBS_STATE: state})[OBS_STATE]

        sample: dict[str, Any] = {
            "task": observation["annotation.human.action.task_description"],
            OBS_STATE: state,
        }

        for index, (previous_key, current_key) in enumerate(CAMERA_KEYS):
            image_key = f"observation.images.image{index}"
            sample[image_key] = self._frame_stack(
                observation[previous_key], observation[current_key], scale=255.0
            )
            sample[f"{image_key}_mask"] = torch.tensor(True)

        previous_deforms = observation["last_tactile_deforms"]
        current_deforms = observation["tactile_deforms"]
        for index, finger_name in enumerate(FINGER_NAMES):
            sample[f"observation.tactile_images.{finger_name}_deform"] = self._frame_stack(
                previous_deforms[index], current_deforms[index]
            )

        current_force = torch.as_tensor(observation["tactile_force"], dtype=torch.float32).flatten()
        previous_force_value = observation.get("last_tactile_force")
        previous_force = (
            current_force.clone()
            if previous_force_value is None
            else torch.as_tensor(previous_force_value, dtype=torch.float32).flatten()
        )
        if current_force.numel() != 60 or previous_force.numel() != 60:
            raise ValueError(
                "Expected current and previous tactile force vectors with 60 values each"
            )
        sample["observation.tactile"] = (
            torch.stack((previous_force, current_force, current_force.clone()))
            / self.tactile_force_scale
        )

        return self.processor(sample)

    @staticmethod
    def _to_model_inputs(sample: dict[str, Any]) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for key, value in sample.items():
            if key == "task":
                inputs[key] = [value]
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Expected tensor for {key}, got {type(value).__name__}")
            value = value.unsqueeze(0).cuda()
            inputs[key] = value if not value.is_floating_point() else value.float()
        return inputs

    def get_server_action(self, payload: dict[str, Any]) -> JSONResponse | str:
        try:
            double_encoded = "encoded" in payload
            if double_encoded:
                if set(payload) != {"encoded"}:
                    raise ValueError("An encoded request must contain only the 'encoded' field")
                payload = json.loads(payload["encoded"])

            start_time = time.time()
            sample = self._build_sample(payload["observation"])
            inputs = self._to_model_inputs(sample)

            save_image(
                inputs["observation.images.image0"][0],
                self.recon_dir / f"observation_{self.request_index}.png",
            )
            save_image(
                inputs["observation.tactile_images.left_thumb_deform"][0],
                self.recon_dir / f"tactile_left_thumb_{self.request_index}.png",
            )

            with torch.inference_mode():
                action_prediction, reconstructed_images, *_ = self.policy.predict_action_chunk(
                    inputs, decode_image=True
                )
            action_prediction = action_prediction[0]

            if reconstructed_images is not None:
                save_image(
                    (reconstructed_images + 1) / 2,
                    self.recon_dir / f"reconstruction_{self.request_index}.png",
                )
            self.request_index += 1

            action_prediction = self.action_unnormalizer({ACTION: action_prediction})[ACTION]
            if self.action_mode == "delta":
                state = inputs[OBS_STATE].repeat(self.chunk_size, 1)
                state = self.state_unnormalizer({OBS_STATE: state})[OBS_STATE]
                action_prediction += state[:, : action_prediction.shape[-1]]

            actions = action_prediction.cpu().numpy()
            logging.info("Inference completed in %.3f seconds", time.time() - start_time)
            response = json_numpy.dumps(actions) if double_encoded else actions
            return JSONResponse(response)
        except Exception:
            logging.exception("Invalid DeCAL inference request")
            return "error"

    def run(self, host: str, port: int) -> None:
        app = FastAPI()
        app.post("/act")(self.get_server_action)
        uvicorn.run(app, host=host, port=port)


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    DeCALServer(cfg).run(cfg.host, cfg.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    deploy()
