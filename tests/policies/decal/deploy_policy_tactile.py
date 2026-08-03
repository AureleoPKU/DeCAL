import os
# Keep your existing setting if you want online access
os.environ["HF_HUB_OFFLINE"] = "0"

print("HF_HOME =", os.environ.get("HF_HOME"))

from pathlib import Path
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.DeCAL import DeCALConfig, DeCALPolicy
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

# Path to checkpoint: use .../checkpoints/STEP/pretrained_model (where config.json and model.safetensors live)
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

import os
import numpy as np
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import write_json, load_json, cast_stats_to_numpy
from lerobot.transforms.core import UnNormalizeTransformFn
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION, OBS_TACTILE_IMAGES
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1" 
cfg = TrainPipelineConfig.from_pretrained(ckpt_path)
cfg.dataset.repo_id = "unscrew_the_cap_gray"
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

stat_keys = ['min', 'max', 'mean', 'std']
action_stat = {stat_key: np.asarray([dataset.meta.stats["action"][stat_key]]) for stat_key in stat_keys}
state_stat = {stat_key: np.asarray([dataset.meta.stats["observation.state"][stat_key]]) for stat_key in stat_keys}
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

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
from pprint import pp

import lpips

# ImageNet mean/std (for DINOv2 / LPIPS-style pretraining)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINOV2_SIZE = 224

_dinov2_backbone: torch.nn.Module | None = None
_lpips_net: torch.nn.Module | None = None


def _get_dinov2(device: torch.device) -> torch.nn.Module:
    global _dinov2_backbone
    if _dinov2_backbone is None:
        # DINOv2: CLS feature for cosine; frozen for eval
        try:
            m = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=True,
                trust_repo=True,
            )
        except TypeError:
            m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        m = m.to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _dinov2_backbone = m
    return _dinov2_backbone


def _get_lpips(device: torch.device) -> torch.nn.Module:
    global _lpips_net
    if _lpips_net is None:
        m = lpips.LPIPS(net="alex", verbose=False)
        m = m.to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _lpips_net = m
    return _lpips_net


def _ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    if c == 3:
        return x
    if c == 1:
        return x.expand(-1, 3, -1, -1)
    if c == 2:
        return torch.cat([x, x[:, :1]], dim=1)
    raise ValueError(f"expected 1–3 input channels, got {c}")


def _interpolate_bchw(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-2] == size and x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def _dino_input_from_0_1(x: torch.Tensor, size: int) -> torch.Tensor:
    x = _ensure_3ch(x)
    x = _interpolate_bchw(x, size)
    mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _dino_cls_features(model: torch.nn.Module, x_01: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x_01.dtype != torch.float32:
        x_01 = x_01.to(torch.float32)
    xn = _dino_input_from_0_1(x_01, _DINOV2_SIZE).to(device)
    out = model.forward_features(xn)
    if isinstance(out, dict) and "x_norm_clstoken" in out:
        return out["x_norm_clstoken"]
    raise RuntimeError("DINOv2 forward_features did not return x_norm_clstoken; check torch hub DINOv2 version.")


def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f"expected (C,H,W) or (B,C,H,W), got shape {tuple(x.shape)}")


def _vision_recon_to_bchw1(recon: torch.Tensor) -> torch.Tensor:
    """Decode output may be (B,T,C,H,W) or (T,C,H,W) (no batch dim) — take last future frame as (1,C,H,W)."""
    if recon.dim() == 5:
        return recon[:, -1]
    if recon.dim() == 4:
        t, c, h, w = int(recon.shape[0]), int(recon.shape[1]), int(recon.shape[2]), int(recon.shape[3])
        if t > 1 and c in (1, 2, 3, 4) and h > 1 and w > 1:
            return recon[-1].unsqueeze(0)
        return recon[:1]
    raise ValueError(f"vision recon unexpected dim/shape: {tuple(recon.shape)}")


def _align_spatial_bchw(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if pred.shape[2:] == ref.shape[2:]:
        return pred
    dtp = pred.dtype
    p = F.interpolate(pred.float(), size=ref.shape[2:], mode="bilinear", align_corners=False)
    return p.to(dtp)


def dino_feature_cosine(model: torch.nn.Module, pred_01: torch.Tensor, gt_01: torch.Tensor, device: torch.device) -> float:
    """Cosine between DINOv2 CLS features; per-batch row then mean over B (e.g. B=1 or B=10)."""
    pred_01 = _as_bchw(pred_01)
    gt_01 = _as_bchw(gt_01)
    fp = _dino_cls_features(model, pred_01, device)
    fg = _dino_cls_features(model, gt_01, device)
    if fp.shape != fg.shape:
        raise RuntimeError(f"DINO feature shape mismatch: pred {fp.shape} vs gt {fg.shape}")
    return float(F.cosine_similarity(fp, fg, dim=1, eps=1e-8).mean().detach().cpu().item())


def cosmos_feature_cosine(policy_obj: DeCALPolicy, pred_01: torch.Tensor, gt_01: torch.Tensor) -> float:
    """Cosine on visual features from `model.get_cosmos_features`."""
    pred_01 = _as_bchw(pred_01).to(dtype=torch.float32)
    gt_01 = _as_bchw(gt_01).to(dtype=torch.float32)
    with torch.inference_mode():
        pred_feat = policy_obj.model.get_cosmos_features(pred_01)
        gt_feat = policy_obj.model.get_cosmos_features(gt_01)
    pred_feat = pred_feat.reshape(pred_feat.shape[0], -1)
    gt_feat = gt_feat.reshape(gt_feat.shape[0], -1)
    return float(F.cosine_similarity(pred_feat, gt_feat, dim=1, eps=1e-8).mean().detach().cpu().item())


def _lpips_input_from_0_1(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_3ch(x)
    return 2.0 * x - 1.0


def lpips_mean(lp_net: torch.nn.Module, pred_01: torch.Tensor, gt_01: torch.Tensor, device: torch.device) -> float:
    """LPIPS (lower is better). Same spatial size; [0,1]. Batch supported."""
    pred_01 = _as_bchw(pred_01)
    gt_01 = _as_bchw(gt_01)
    if pred_01.shape != gt_01.shape:
        raise ValueError(f"LPIPS shape mismatch: pred {pred_01.shape} vs gt {gt_01.shape}")
    p = _lpips_input_from_0_1(pred_01).to(device)
    g = _lpips_input_from_0_1(gt_01).to(device)
    with torch.inference_mode():
        out = lp_net(p, g)
    return float(out.mean().detach().cpu().item())


def _save_tactile_grid(tensor: torch.Tensor, path: str) -> None:
    """Save (N, C, H, W) tactile tiles; N=10 -> 5 cols x 2 rows."""
    save_image(tensor, path, nrow=5)


def _tactile_deform_gt_10f_future(inputs: dict, t_idx: int = -1) -> torch.Tensor:
    """Return one deform frame in policy finger order with shape (10, C, H, W)."""
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
    # print("inputs[f{OBS_TACTILE_IMAGES}.left_thumb_deform]:", inputs[f"{OBS_TACTILE_IMAGES}.left_thumb_deform"].shape)
    return torch.stack([inputs[k][0, t_idx] for k in keys], dim=0)


def _tactile_orig_gt_10f_future(inputs: dict, t_idx: int = -1) -> torch.Tensor | None:
    """Return one raw frame in policy finger order, or None when raw inputs are absent."""
    keys = [
        f"{OBS_TACTILE_IMAGES}.left_thumb_raw",
        f"{OBS_TACTILE_IMAGES}.left_index_raw",
        f"{OBS_TACTILE_IMAGES}.left_middle_raw",
        f"{OBS_TACTILE_IMAGES}.left_ring_raw",
        f"{OBS_TACTILE_IMAGES}.left_pinky_raw",
        f"{OBS_TACTILE_IMAGES}.right_thumb_raw",
        f"{OBS_TACTILE_IMAGES}.right_index_raw",
        f"{OBS_TACTILE_IMAGES}.right_middle_raw",
        f"{OBS_TACTILE_IMAGES}.right_ring_raw",
        f"{OBS_TACTILE_IMAGES}.right_pinky_raw",
    ]
    if not all(k in inputs for k in keys):
        return None
    gt = torch.stack([inputs[k][0, t_idx] for k in keys], dim=0)
    if gt.min() < 0:
        gt = (gt + 1) / 2
    return gt


from_ids = np.asarray(dataset.meta.episodes['dataset_from_index']).tolist()
to_ids = np.asarray(dataset.meta.episodes['dataset_to_index']).tolist()
total_num_episodes = dataset.num_episodes

output_dir = Path(f"outputs/decal/open_loop_results/last/{Path(cfg.dataset.repo_id).name}")
(output_dir / "plots").mkdir(exist_ok=True, parents=True)

_eval_device = next(policy.parameters()).device
_dinov2 = _get_dinov2(_eval_device)
_lp = _get_lpips(_eval_device)

num_episodes_infer = 3
metric_mse = []
mse_arm_joint = []
mse_hand_joint = []
vision_l2 = []
vision_dino_cos = []
vision_cosmos_cos = []
vision_lpips = []
tactile_l2 = []
tactile_dino_cos = []
tactile_lpips = []
force6d_mse = []
for ep_id in range(min(total_num_episodes, num_episodes_infer)):
    print(f"episode: {ep_id}")
    print(f"from_idx: {from_ids[ep_id]}, to_idx: {to_ids[ep_id]}")
    action_gt_list = []
    action_pred_list = []
    state_list = []
    ep_vision_l2 = []
    ep_vision_dino = []
    ep_vision_cosmos = []
    ep_vision_lpips = []
    ep_tactile_l2 = []
    ep_tactile_dino = []
    ep_tactile_lpips = []
    ep_force6d_mse = []
    for idx in range(from_ids[ep_id], to_ids[ep_id], config.chunk_size):
        sample = dataset[idx]
        # print("sample[task]:", sample["task"])
        inputs = {}
        for key in sample.keys():
            if key == 'task':
                inputs[key] = [sample[key]]
            elif sample[key].dtype == torch.int64 or sample[key].dtype == torch.bool:
                inputs[key] = sample[key][None].cuda()
            else:
                inputs[key] = sample[key][None].cuda().to(dtype=dtype)
        # print("keys:", inputs.keys())
        # print("inputs[observation.images.image0]:", inputs["observation.images.image0"].shape, inputs["observation.images.image0"])
        os.makedirs("recon_images_tac_no_gate", exist_ok=True)
        save_image(inputs["observation.images.image0"][0], f"recon_images_tac_no_gate/observation_{ep_id}_{idx}.png")
        tactile_obs_vis = _tactile_deform_gt_10f_future(inputs, t_idx=0)  # Current frame, (10, C, H, W).
        _save_tactile_grid(tactile_obs_vis, f"recon_images_tac_no_gate/observation_tactile_deform_{ep_id}_{idx}.png")
        with torch.no_grad():
            action_pred, recon_images, recon_tactile, recon_force, recon_tactile_orig = policy.predict_action_chunk(
                inputs, decode_image=True, decode_tactile=True
            )
            action_pred = action_pred[0, :, :56]
            action_gt = inputs['action'][0, :, :56]
            action_gt_list.append(action_gt)
            action_pred_list.append(action_pred.clone())
            state_list.append(inputs[OBS_STATE].clone().repeat(config.chunk_size, 1)[:, :56])
            print("state:", state_unnorm_fn({OBS_STATE: inputs[OBS_STATE]})[OBS_STATE][:, :6].cpu().numpy())
            print("action_gt:", act_unnorm_fn({ACTION: action_gt})[ACTION][:, :6].cpu().numpy())
            print("action_pred:", act_unnorm_fn({ACTION: action_pred})[ACTION][:, :6].cpu().numpy())
            os.makedirs("recon_images_tac_no_gate", exist_ok=True)
            recon_images = (recon_images + 1) / 2 
            save_image(recon_images, f"recon_images_tac_no_gate/recon_ep{ep_id}_{idx}.png")
            # recon_tactile: (B, 10, C, H, W) in [-1, 1]; save_image expects (N, C, H, W)
            recon_tactile = (recon_tactile + 1) / 2
            recon_tactile = recon_tactile[0]  # (10, C, H, W)
            _save_tactile_grid(recon_tactile, f"recon_images_tac_no_gate/recon_tactile_deform_ep{ep_id}_{idx}.png")
            tactile_gt_vis = _tactile_deform_gt_10f_future(inputs)  # (10, C, H, W), [0, 1]
            _save_tactile_grid(tactile_gt_vis, f"recon_images_tac_no_gate/tactile_gt_deform_ep{ep_id}_{idx}.png")
            if recon_tactile_orig is not None:
                recon_tactile_orig_vis = (recon_tactile_orig + 1) / 2
                recon_tactile_orig_vis = recon_tactile_orig_vis[0]  # (10, C, H, W)
                _save_tactile_grid(
                    recon_tactile_orig_vis,
                    f"recon_images_tac_no_gate/recon_tactile_orig_ep{ep_id}_{idx}.png",
                )
                tactile_orig_gt_vis = _tactile_orig_gt_10f_future(inputs)
                if tactile_orig_gt_vis is not None:
                    _save_tactile_grid(
                        tactile_orig_gt_vis,
                        f"recon_images_tac_no_gate/tactile_gt_orig_ep{ep_id}_{idx}.png",
                    )

            # 1) Vision: pixel L2, DINO CLS cosine, LPIPS
            gt_vision = _as_bchw(inputs["observation.images.image0"][:, -1])  # (1, C, H, W) e.g. 224
            recon_images = recon_images[:1]
            pred_vision = _vision_recon_to_bchw1(recon_images)  # (1, C, H, W) e.g. 256; last future frame
            pred_vision = _align_spatial_bchw(pred_vision, gt_vision)  # match GT H,W to avoid MSE/LPIPS broadcast
            ep_vision_l2.append(float(F.mse_loss(pred_vision, gt_vision, reduction="mean").detach().cpu().item()))
            ep_vision_dino.append(dino_feature_cosine(_dinov2, pred_vision, gt_vision, _eval_device))
            ep_vision_cosmos.append(cosmos_feature_cosine(policy, pred_vision, gt_vision))
            ep_vision_lpips.append(lpips_mean(_lp, pred_vision, gt_vision, _eval_device))

            # 2) Tactile metrics in policy finger order, aligned to the decoder output.
            pred_tactile = recon_tactile[:, :, :, :].contiguous()  # (10, C, H, W)
            gt_tactile = _tactile_deform_gt_10f_future(inputs)  # (10, C, H, W), aligned with predictions.
            print("gt_tactile:", gt_tactile)
            print("pred_tactile:", pred_tactile)
            if gt_tactile.shape[2:] != pred_tactile.shape[2:]:
                gt_tactile = _align_spatial_bchw(gt_tactile, _as_bchw(pred_tactile[0:1]))
            # print("pred_tactile:", pred_tactile.shape)
            # print("gt_tactile:", gt_tactile.shape)
            ep_tactile_l2.append(float(F.mse_loss(pred_tactile, gt_tactile, reduction="mean").detach().cpu().item()))
            ep_tactile_dino.append(
                dino_feature_cosine(_dinov2, pred_tactile.detach().float(), gt_tactile.detach().float(), _eval_device)
            )
            ep_tactile_lpips.append(
                lpips_mean(_lp, pred_tactile.detach().float(), gt_tactile.detach().float(), _eval_device)
            )

            # 3) 6D tactile force generation metric (MSE)
            gt_force_future = inputs["observation.tactile"][:, 2].reshape(inputs["observation.tactile"].shape[0], 10, 6)
            ep_force6d_mse.append(float(F.mse_loss(recon_force, gt_force_future, reduction="mean").detach().cpu().item()))
    action_gt_tensor = torch.cat(action_gt_list, dim=0)
    # print("action_gt_tensor:", action_gt_tensor.shape)
    action_gt_tensor = act_unnorm_fn({ACTION: action_gt_tensor})[ACTION]
    action_pred_tensor = torch.cat(action_pred_list, dim=0)
    action_pred_tensor = act_unnorm_fn({ACTION: action_pred_tensor})[ACTION]
    if action_mode == 'delta':
        state_tensor = torch.cat(state_list, dim=0)
        state_tensor = state_unnorm_fn({OBS_STATE: state_tensor})[OBS_STATE]
        action_pred_tensor[:, :56] += state_tensor[:, :56]
        action_gt_tensor[:, :56] += state_tensor[:, :56]
    action_gt_tensor = action_gt_tensor.to(torch.float32)
    action_pred_tensor = action_pred_tensor.to(torch.float32)
    metric_mse.append(float(F.mse_loss(action_gt_tensor, action_pred_tensor, reduction='mean').detach().cpu().numpy()))
    mse_arm_joint.append(float(F.mse_loss(action_gt_tensor[:, :12], action_pred_tensor[:, :12], reduction='mean').detach().cpu().numpy()))
    mse_hand_joint.append(float(F.mse_loss(action_gt_tensor[:, 12:], action_pred_tensor[:, 12:], reduction='mean').detach().cpu().numpy()))
    if len(ep_vision_l2) > 0:
        vision_l2.append(float(np.mean(ep_vision_l2)))
    if len(ep_vision_dino) > 0:
        vision_dino_cos.append(float(np.mean(ep_vision_dino)))
    if len(ep_vision_cosmos) > 0:
        vision_cosmos_cos.append(float(np.mean(ep_vision_cosmos)))
    if len(ep_vision_lpips) > 0:
        vision_lpips.append(float(np.mean(ep_vision_lpips)))
    if len(ep_tactile_l2) > 0:
        tactile_l2.append(float(np.mean(ep_tactile_l2)))
    if len(ep_tactile_dino) > 0:
        tactile_dino_cos.append(float(np.mean(ep_tactile_dino)))
    if len(ep_tactile_lpips) > 0:
        tactile_lpips.append(float(np.mean(ep_tactile_lpips)))
    if len(ep_force6d_mse) > 0:
        force6d_mse.append(float(np.mean(ep_force6d_mse)))
    action_gt_numpy = action_gt_tensor.detach().cpu().numpy()
    action_pred_numpy = action_pred_tensor.detach().cpu().numpy()
    fig, axs = plt.subplots(8, 7, figsize=(16, 12))
    axs = axs.ravel()
    num_dimensions = action_gt_numpy.shape[1]
    x_values = np.arange(action_gt_numpy.shape[0])
    for dim in range(num_dimensions):
        axs[dim].plot(x_values, action_gt_numpy[:, dim], label='Ground Truth', color='blue', linewidth=1.5)
        axs[dim].plot(x_values, action_pred_numpy[:, dim], label='Predicted', color='red', linestyle='--', linewidth=1.5)
        axs[dim].set_title(f'Dimension {dim+1}')
        axs[dim].set_xlabel('Time Step / Sample Index')
        axs[dim].set_ylabel(f'Value Dim {dim+1}')
        axs[dim].legend(loc='upper right')
        axs[dim].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.suptitle('Ground Truth vs Prediction', fontsize=16, y=1.02)
    plt.savefig(output_dir / "plots" / f"qwena1_open_loop_ep{ep_id}.jpg")
log = {
    "MSE": metric_mse, 
    "Average MSE": np.mean(metric_mse), 
    "MSE on arm joints": mse_arm_joint, 
    "Average MSE on arm joints": np.mean(mse_arm_joint), 
    "MSE on hand joints": mse_hand_joint, 
    "Average MSE on hand joints": np.mean(mse_hand_joint), 
    "Vision L2 (pixel)": vision_l2,
    "Average Vision L2 (pixel)": np.mean(vision_l2) if len(vision_l2) > 0 else None,
    "Vision DINO CLS cosine": vision_dino_cos,
    "Average Vision DINO CLS cosine": np.mean(vision_dino_cos) if len(vision_dino_cos) > 0 else None,
    "Vision Cosmos Encoder cosine": vision_cosmos_cos,
    "Average Vision Cosmos Encoder cosine": np.mean(vision_cosmos_cos) if len(vision_cosmos_cos) > 0 else None,
    "Vision LPIPS (alex)": vision_lpips,
    "Average Vision LPIPS (alex)": np.mean(vision_lpips) if len(vision_lpips) > 0 else None,
    "Tactile L2 (pixel)": tactile_l2,
    "Average Tactile L2 (pixel)": np.mean(tactile_l2) if len(tactile_l2) > 0 else None,
    "Tactile DINO CLS cosine (10 tiles mean)": tactile_dino_cos,
    "Average Tactile DINO CLS cosine": np.mean(tactile_dino_cos) if len(tactile_dino_cos) > 0 else None,
    "Tactile LPIPS (alex, 10 tiles mean)": tactile_lpips,
    "Average Tactile LPIPS (alex)": np.mean(tactile_lpips) if len(tactile_lpips) > 0 else None,
    "Force 6D MSE": force6d_mse,
    "Average Force 6D MSE": np.mean(force6d_mse) if len(force6d_mse) > 0 else None,
}
write_json(log, output_dir/"log.json")
pp(log)
