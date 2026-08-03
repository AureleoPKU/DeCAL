"""Online open-loop evaluation with tactile gate visualization.

Extends deploy_policy_tactile.py: records the per-forward tactile gate (Sigmoid output)
from DeCAL and plots frame vs gate for each of the first N episodes.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "0"

print("HF_HOME =", os.environ.get("HF_HOME"))

from pathlib import Path
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.DeCAL import DeCALConfig, DeCALPolicy
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

checkpoint_value = os.environ.get("DECAL_CHECKPOINT")
if not checkpoint_value:
    raise RuntimeError("Set DECAL_CHECKPOINT to a DeCAL checkpoint step or pretrained_model directory")
checkpoint_step_dir = Path(checkpoint_value).expanduser().resolve()
ckpt_path = (checkpoint_step_dir / PRETRAINED_MODEL_DIR).resolve()
if checkpoint_step_dir.name == PRETRAINED_MODEL_DIR:
    ckpt_path = checkpoint_step_dir
if not ckpt_path.is_dir():
    raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_path}")
config = PreTrainedConfig.from_pretrained(ckpt_path)
config.compile_model = False
config.compile_mode = "reduce-overhead"
dtype = torch.float32
assert isinstance(config, DeCALConfig)
policy = DeCALPolicy.from_pretrained(
    config=config,
    pretrained_name_or_path=ckpt_path,
)
policy.cuda()
policy.to(dtype)
policy.eval()

import numpy as np
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import write_json, load_json, cast_stats_to_numpy
from lerobot.transforms.core import UnNormalizeTransformFn
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION, OBS_TACTILE_IMAGES
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
cfg = TrainPipelineConfig.from_pretrained(ckpt_path)
cfg.dataset.repo_id = "unscrew_the_cap"
cfg.dataset.use_external_stats = True
action_mode = cfg.dataset.action_mode
dataset, _ = make_dataset(cfg)

if Path(ckpt_path).is_dir():
    stats_file = Path(ckpt_path) / "stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"stats.json not found in {ckpt_path}")
else:
    stats_file = hf_hub_download(
        repo_id=str(ckpt_path),
        filename="stats.json",
    )
stats = cast_stats_to_numpy(load_json(stats_file)[dataset.meta.robot_type])
dataset.meta.stats.update(stats)

stat_keys = ["min", "max", "mean", "std"]
action_stat = {stat_key: np.asarray([dataset.meta.stats["action"][stat_key]]) for stat_key in stat_keys}
state_stat = {
    stat_key: np.asarray([dataset.meta.stats["observation.state"][stat_key]]) for stat_key in stat_keys
}
act_unnorm_fn = UnNormalizeTransformFn(
    selected_keys=[ACTION],
    mode="mean_std",
    norm_stats={ACTION: action_stat},
)
state_unnorm_fn = UnNormalizeTransformFn(
    selected_keys=[OBS_STATE],
    mode="mean_std",
    norm_stats={OBS_STATE: state_stat},
)

import math

import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.ticker import FuncFormatter
from torchvision.utils import save_image
from pprint import pp

GATE_PLOT_FONT_SIZE = 12


def _infer_gate_scale_exponent(series_list: list[list[float]]) -> int:
    """Pick exponent e so gate value v is shown as (v / 10**e), e.g. 0.0005 -> 0.5 * 10^-3."""
    all_vals = [v for ep in series_list for v in ep if v > 0]
    if not all_vals:
        return 0
    return int(round(math.log10(max(all_vals))))


def _gate_ylabel(exp: int) -> str:
    if exp == 0:
        return "tactile gate"
    return rf"tactile gate ($\times 10^{{{exp}}}$)"


def _style_gate_axes(ax) -> None:
    ax.set_xlabel(ax.get_xlabel(), fontsize=GATE_PLOT_FONT_SIZE)
    ax.set_ylabel(ax.get_ylabel(), fontsize=GATE_PLOT_FONT_SIZE)
    ax.set_title(ax.get_title(), fontsize=GATE_PLOT_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=GATE_PLOT_FONT_SIZE, width=1.2)


def _style_gate_suptitle(suptitle) -> None:
    suptitle.set_fontsize(GATE_PLOT_FONT_SIZE)


def _tactile_deform_gt_10f_future(inputs: dict, t_idx: int = -1) -> torch.Tensor:
    keys = [
        f"{OBS_TACTILE_IMAGES}.left_thumb_deform",
        f"{OBS_TACTILE_IMAGES}.left_index_deform",
        f"{OBS_TACTILE_IMAGES}.left_middle_deform",
        f"{OBS_TACTILE_IMAGES}.left_ring_deform",
        f"{OBS_TACTILE_IMAGES}.left_pinky_deform",
        f"{OBS_TACTILE_IMAGES}.right_thumb_deform",
        f"{OBS_TACTILE_IMAGES}.right_index_deform",
        f"{OBS_TACTILE_IMAGES}.right_middle_deform",
        f"{OBS_TACTILE_IMAGES}.right_ring_deform",
        f"{OBS_TACTILE_IMAGES}.right_pinky_deform",
    ]
    return torch.stack([inputs[k][0, t_idx] for k in keys], dim=0)


from_ids = np.asarray(dataset.meta.episodes["dataset_from_index"]).tolist()
to_ids = np.asarray(dataset.meta.episodes["dataset_to_index"]).tolist()
total_num_episodes = dataset.num_episodes

output_dir = Path(f"outputs/decal/open_loop_results/last/{Path(cfg.dataset.repo_id).name}")
(output_dir / "plots").mkdir(exist_ok=True, parents=True)

# --- tactile gate capture (forward hook on nn.Sequential tactile_gate) ---
tactile_gate_module = policy.model.qwen3_vl_with_expert.tactile_gate
_gate_buffer_per_forward: list[float] = []


def _tactile_gate_forward_hook(_module, _inp, out: torch.Tensor) -> None:
    # out: (B, 1) after Sigmoid in modeling_decal.py
    g = out.detach().float().mean().item()
    _gate_buffer_per_forward.append(g)


_gate_hook_handle = tactile_gate_module.register_forward_hook(_tactile_gate_forward_hook)

num_episodes_infer = 3
n_ep_run = min(total_num_episodes, num_episodes_infer)
metric_mse = []
mse_arm_joint = []
mse_hand_joint = []
# Per-episode lists: one gate scalar per inference step (frame index 0..T-1)
episode_gate_series: list[list[float]] = []
inference_step = 10

try:
    for ep_id in range(n_ep_run):
        print(f"episode: {ep_id}")
        print(f"from_idx: {from_ids[ep_id]}, to_idx: {to_ids[ep_id]}")
        action_gt_list = []
        action_pred_list = []
        state_list = []
        gate_this_ep: list[float] = []
        frame_idx = 0
        for idx in range(from_ids[ep_id], to_ids[ep_id], inference_step):
            sample = dataset[idx]
            inputs = {}
            for key in sample.keys():
                if key == "task":
                    inputs[key] = [sample[key]]
                elif sample[key].dtype == torch.int64 or sample[key].dtype == torch.bool:
                    inputs[key] = sample[key][None].cuda()
                else:
                    inputs[key] = sample[key][None].cuda().to(dtype=dtype)
            os.makedirs("recon_images_tac_gate", exist_ok=True)
            save_image(inputs["observation.images.image0"][0], f"recon_images_tac_gate/observation_{ep_id}_{idx}.png")
            print(
                "inputs[observation.tactile_images.left_thumb_deform]:",
                inputs["observation.tactile_images.left_thumb_deform"].shape,
            )
            save_image(
                inputs["observation.tactile_images.left_thumb_deform"][0],
                f"recon_images_tac_gate/observation_tactile_deform_{ep_id}_{idx}.png",
            )
            with torch.no_grad():
                _gate_buffer_per_forward.clear()
                action_pred, recon_images, recon_tactile, recon_force, _ = policy.predict_action_chunk(
                    inputs, decode_image=True, decode_tactile=True
                )
                # tactile_gate runs once per transformer layer; values are identical → mean is stable
                if not _gate_buffer_per_forward:
                    raise RuntimeError(
                        "Tactile gate hook captured no values; check model path / tactile fusion enabled."
                    )
                gate_scalar = float(np.mean(_gate_buffer_per_forward))
                gate_this_ep.append(gate_scalar)

                action_pred = action_pred[0, :, :56]
                action_gt = inputs["action"][0, :, :56]
                action_gt_list.append(action_gt)
                action_pred_list.append(action_pred.clone())
                state_list.append(inputs[OBS_STATE].clone().repeat(config.chunk_size, 1)[:, :56])
                print("state:", state_unnorm_fn({OBS_STATE: inputs[OBS_STATE]})[OBS_STATE][:, :6].cpu().numpy())
                print("action_gt:", act_unnorm_fn({ACTION: action_gt})[ACTION][:, :6].cpu().numpy())
                print("action_pred:", act_unnorm_fn({ACTION: action_pred})[ACTION][:, :6].cpu().numpy())
                print(f"frame {frame_idx} tactile gate (mean over layers): {gate_scalar:.6f}")
                recon_images = (recon_images + 1) / 2
                save_image(recon_images, f"recon_images_tac_gate/recon_ep{ep_id}_{idx}.png")
                recon_tactile = (recon_tactile + 1) / 2
                recon_tactile = recon_tactile[0]
                save_image(recon_tactile, f"recon_images_tac_gate/recon_tactile_ep{ep_id}_{idx}.png")
                tactile_gt_vis = _tactile_deform_gt_10f_future(inputs)
                save_image(tactile_gt_vis, f"recon_images_tac_gate/tactile_gt_ep{ep_id}_{idx}.png")
            frame_idx += 1

        episode_gate_series.append(gate_this_ep)

        action_gt_tensor = torch.cat(action_gt_list, dim=0)
        action_gt_tensor = act_unnorm_fn({ACTION: action_gt_tensor})[ACTION]
        action_pred_tensor = torch.cat(action_pred_list, dim=0)
        action_pred_tensor = act_unnorm_fn({ACTION: action_pred_tensor})[ACTION]
        if action_mode == "delta":
            state_tensor = torch.cat(state_list, dim=0)
            state_tensor = state_unnorm_fn({OBS_STATE: state_tensor})[OBS_STATE]
            action_pred_tensor[:, :56] += state_tensor[:, :56]
            action_gt_tensor[:, :56] += state_tensor[:, :56]
        action_gt_tensor = action_gt_tensor.to(torch.float32)
        action_pred_tensor = action_pred_tensor.to(torch.float32)
        metric_mse.append(
            float(F.mse_loss(action_gt_tensor, action_pred_tensor, reduction="mean").detach().cpu().numpy())
        )
        mse_arm_joint.append(
            float(
                F.mse_loss(action_gt_tensor[:, :12], action_pred_tensor[:, :12], reduction="mean")
                .detach()
                .cpu()
                .numpy()
            )
        )
        mse_hand_joint.append(
            float(
                F.mse_loss(action_gt_tensor[:, 12:], action_pred_tensor[:, 12:], reduction="mean")
                .detach()
                .cpu()
                .numpy()
            )
        )
        action_gt_numpy = action_gt_tensor.detach().cpu().numpy()
        action_pred_numpy = action_pred_tensor.detach().cpu().numpy()
        fig, axs = plt.subplots(8, 7, figsize=(16, 12))
        axs = axs.ravel()
        num_dimensions = action_gt_numpy.shape[1]
        x_values = np.arange(action_gt_numpy.shape[0])
        for dim in range(num_dimensions):
            axs[dim].plot(x_values, action_gt_numpy[:, dim], label="Ground Truth", color="blue", linewidth=1.5)
            axs[dim].plot(
                x_values,
                action_pred_numpy[:, dim],
                label="Predicted",
                color="red",
                linestyle="--",
                linewidth=1.5,
            )
            axs[dim].set_title(f"Dimension {dim + 1}")
            axs[dim].set_xlabel("Time Step / Sample Index")
            axs[dim].set_ylabel(f"Value Dim {dim + 1}")
            axs[dim].legend(loc="upper right")
            axs[dim].grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.suptitle("Ground Truth vs Prediction", fontsize=16, y=1.02)
        plt.savefig(output_dir / "plots" / f"qwena1_open_loop_ep{ep_id}.jpg")
        plt.close(fig)

finally:
    _gate_hook_handle.remove()

# --- one subplot per episode: frame index vs tactile gate ---
gate_scale_exp = _infer_gate_scale_exponent(episode_gate_series)
gate_scale = 10.0**gate_scale_exp

# Shorter per-row height so the multi-episode gate figure is less tall.
_gate_row_h = 2.45
fig_gate, axs_gate = plt.subplots(
    n_ep_run,
    1,
    figsize=(10, max(2.8, _gate_row_h * n_ep_run)),
    sharex=False,
)
if n_ep_run == 1:
    axs_gate = [axs_gate]
for i in range(n_ep_run):
    y = np.asarray(episode_gate_series[i], dtype=np.float64)
    # One point is produced every `inference_step` frames in the dataset.
    x = np.arange(len(y)) * inference_step
    y_scaled = y / gate_scale
    axs_gate[i].plot(x, y_scaled, color="#FF7F0E", linewidth=2.0, marker="o", markersize=4)
    axs_gate[i].set_ylabel(_gate_ylabel(gate_scale_exp))
    axs_gate[i].set_title(f"Episode {i}: tactile gate vs frame")
    axs_gate[i].grid(True, linestyle="--", alpha=0.7)
    y_top = max(1.0, float(y_scaled.max()) * 1.05) if len(y_scaled) else 1.0
    axs_gate[i].set_ylim(0.0, y_top)
    axs_gate[i].yaxis.set_major_formatter(
        FuncFormatter(lambda val, _pos: f"{val:g}" if val != 0 else "0")
    )
    if i == n_ep_run - 1:
        axs_gate[i].set_xlabel("frame (within episode)")
    _style_gate_axes(axs_gate[i])
_style_gate_suptitle(
    fig_gate.suptitle(
        "Tactile fusion gate σ(global tactile token) over open-loop steps",
        fontsize=GATE_PLOT_FONT_SIZE,
        y=1.01,
    )
)
fig_gate.tight_layout()
gate_plot_path = output_dir / "plots" / "tactile_gate_per_episode.png"
fig_gate.savefig(gate_plot_path, dpi=150, bbox_inches="tight")
plt.close(fig_gate)
print(f"Saved tactile gate plot: {gate_plot_path}")

log = {
    "MSE": metric_mse,
    "Average MSE": np.mean(metric_mse),
    "MSE on arm joints": mse_arm_joint,
    "Average MSE on arm joints": np.mean(mse_arm_joint),
    "MSE on hand joints": mse_hand_joint,
    "Average MSE on hand joints": np.mean(mse_hand_joint),
    "tactile_gate_per_episode": episode_gate_series,
}
write_json(log, output_dir / "log_tactile_gate.json")
pp(log)
