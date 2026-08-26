"""Generate WBench navigation videos with SANA-WM stage-1 (no refiner).

Kernel-owned and runs in the `sana` conda env. Agents cannot edit this file, so
every node is evaluated through an identical generation protocol; only the model
weights (base + the node's LoRA delta) and the data they were trained on differ.

Protocol: poses come from WBench's own `case_to_poses` (canonical fps=24,
temporal_compression=4), are interpolated to one pose per RGB frame, and the mp4
is written with WBench's writer settings.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

TARGET_H, TARGET_W = 704, 1280
WBENCH_FPS = 24


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- pose helpers ----------

def _quat_from_R(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / (np.linalg.norm(q) + 1e-12)


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if d > 0.9995:
        return q0 + t * (q1 - q0)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def interpolate_poses(poses: np.ndarray, n_out: int) -> np.ndarray:
    """Resample (N,4,4) camera-to-world matrices to `n_out` frames (slerp + lerp)."""
    n_in = poses.shape[0]
    if n_in == n_out:
        return poses.astype(np.float32)
    quats = np.stack([_quat_from_R(p[:3, :3]) for p in poses])
    trans = poses[:, :3, 3]
    src = np.linspace(0.0, n_in - 1, n_out)
    out = np.tile(np.eye(4, dtype=np.float32), (n_out, 1, 1))
    for i, s in enumerate(src):
        lo = int(np.floor(s))
        hi = min(lo + 1, n_in - 1)
        t = float(s - lo)
        out[i, :3, :3] = _R_from_quat(_slerp(quats[lo], quats[hi], t))
        out[i, :3, 3] = trans[lo] * (1 - t) + trans[hi] * t
    return out


def poses_json_to_array(poses: dict) -> np.ndarray:
    keys = sorted(poses, key=int)
    return np.stack([np.array(poses[k]["extrinsic"], dtype=np.float32) for k in keys])


def snap_frames(n: int, stride: int = 8) -> int:
    """LTX-2 VAE needs num_frames = stride*k + 1."""
    return int(np.ceil((n - 1) / stride) * stride + 1)


def build_prompt(case: dict) -> str:
    parts = [case.get("environment_prompt", ""), case.get("character_prompt", ""),
             case.get("perspective_prompt", "")]
    return " ".join(p.strip() for p in parts if p and p.strip())


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sana_root", required=True)
    ap.add_argument("--wbench_root", required=True)
    ap.add_argument("--out_dir", required=True, help="work_dirs/<model>/videos")
    ap.add_argument("--cases", required=True, help="JSON file: list of case ids")
    ap.add_argument("--config", default="hf://Efficient-Large-Model/SANA-WM_bidirectional/config.yaml")
    ap.add_argument("--model_path",
                    default="hf://Efficient-Large-Model/SANA-WM_bidirectional/dit/sana_wm_1600m_720p.safetensors")
    ap.add_argument("--lora", default="", help="peft adapter dir to merge into the DiT")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--duration", type=float, default=4.0, help="seconds per interaction turn")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames per case (0 = full)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    sana_root, wbench_root = Path(args.sana_root).resolve(), Path(args.wbench_root).resolve()
    sys.path.insert(0, str(sana_root))
    sys.path.insert(0, str(wbench_root))
    os.environ.setdefault("DISABLE_XFORMERS", "1")

    import cv2
    import torch
    from PIL import Image

    from src.models.camera.poses import DEFAULT_INTRINSIC, case_to_poses

    wm = load_module(sana_root / "inference_video_scripts" / "wm" / "inference_sana_wm.py", "sana_wm_inf")
    import pyrallis

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir.parent / f"generate_shard{args.shard}.json"

    ids = json.loads(Path(args.cases).read_text())
    mine = [c for i, c in enumerate(ids) if i % args.num_shards == args.shard]
    cases_dir = wbench_root / "data" / "cases"

    config = pyrallis.parse(config_class=wm.InferenceConfig,
                            config_path=wm.resolve_hf_path(args.config), args=[])
    pipeline = wm.SanaWMPipeline(
        config=config, model_path=wm.resolve_hf_path(args.model_path),
        device=torch.device("cuda"), refiner=None,   # stage-1 only
    )
    if args.lora:
        n = merge_lora(pipeline.model, Path(args.lora))
        print(f"[gen] merged LoRA from {args.lora} into {n} modules", flush=True)

    sampling_algo = (config.scheduler.vis_sampler
                     if config.scheduler.vis_sampler in {"chunk_flow_euler", "self_forcing"}
                     else "flow_euler_ltx")

    results = []
    for cid in mine:
        dst = out_dir / f"case_{cid}_combined.mp4"
        if args.resume and dst.exists() and dst.stat().st_size > 0:
            results.append({"id": cid, "status": "cached"})
            continue
        t0 = time.time()
        try:
            case = json.loads((cases_dir / f"case_{cid}.json").read_text())
            conv = case_to_poses(case, duration=args.duration)
            video_length = int(conv["video_length"])
            if args.max_frames:
                video_length = min(video_length, args.max_frames)

            img_rel = case["settings"]["initial_image"]
            image = Image.open(wbench_root / "data" / img_rel).convert("RGB")
            cropped, src_size, resized_size, crop_offset = wm.resize_and_center_crop(image)

            num_frames = snap_frames(video_length)
            c2w = interpolate_poses(poses_json_to_array(conv["poses"]), num_frames)

            K = np.array(DEFAULT_INTRINSIC, dtype=np.float32)
            intr_src = np.broadcast_to(
                np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32),
                (num_frames, 4)).copy()
            # WBench's K is defined for 1920x1080; rescale to this image, then to the crop.
            intr_src *= np.array([src_size[0] / 1920, src_size[1] / 1080,
                                  src_size[0] / 1920, src_size[1] / 1080], dtype=np.float32)
            intrinsics = wm.transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)

            params = wm.GenerationParams(
                num_frames=num_frames, fps=16, step=args.step, cfg_scale=args.cfg_scale,
                seed=args.seed, sampling_algo=sampling_algo,
            )
            out = pipeline.generate(cropped, build_prompt(case), c2w, intrinsics, params)
            video = np.asarray(out["video"])[:video_length]

            writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                                     WBENCH_FPS, (video.shape[2], video.shape[1]))
            for f in video:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            results.append({"id": cid, "status": "ok", "frames": int(video.shape[0]),
                            "seconds": round(time.time() - t0, 1)})
            print(f"[gen] case {cid} ok {video.shape[0]}f in {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            results.append({"id": cid, "status": "error", "error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-2000:]})
            print(f"[gen] case {cid} FAILED: {e}", flush=True)
        status_path.write_text(json.dumps(results, indent=1))

    status_path.write_text(json.dumps(results, indent=1))


def merge_lora(model, adapter_dir: Path) -> int:
    """Fold a peft LoRA adapter into the DiT weights in place."""
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    weights_file = adapter_dir / "adapter_model.safetensors"
    sd = load_file(str(weights_file)) if weights_file.exists() else \
        torch.load(adapter_dir / "adapter_model.bin", map_location="cpu")

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        if ".lora_A" in k:
            pairs.setdefault(k.split(".lora_A")[0], {})["A"] = v
        elif ".lora_B" in k:
            pairs.setdefault(k.split(".lora_B")[0], {})["B"] = v

    modules = dict(model.named_modules())
    merged = 0
    for name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        target = name.removeprefix("base_model.model.").removesuffix(".base_layer")
        mod = modules.get(target)
        if mod is None or not hasattr(mod, "weight"):
            print(f"[gen] WARNING: LoRA target not found: {target}", flush=True)
            continue
        delta = (ab["B"].to(torch.float32) @ ab["A"].to(torch.float32)) * scale
        with torch.no_grad():
            mod.weight.add_(delta.to(mod.weight.device, mod.weight.dtype))
        merged += 1
    if merged == 0:
        raise RuntimeError(f"LoRA merge matched no modules in {adapter_dir}")
    return merged


if __name__ == "__main__":
    main()
