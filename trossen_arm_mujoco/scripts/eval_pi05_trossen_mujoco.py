"""
Evaluation Script for Finetuned PI0.5 in Trossen MuJoCo Environments.
Records a 2x2 grid video from all 4 cameras.

Usage:
  python eval_pi05_trossen_mujoco.py --checkpoint /path/to/model
  python eval_pi05_trossen_mujoco.py --checkpoint /path/to/model --rtc --steps 200
  python eval_pi05_trossen_mujoco.py --checkpoint /path/to/model --xml trossen_ai_scene_2.xml --output video.mp4
"""

import argparse
import os

# Force Headless Rendering (EGL) BEFORE importing dm_control or mujoco
os.environ["MUJOCO_GL"] = "egl"

import json
import time as time_module
import cv2
import numpy as np
import torch
from pathlib import Path
import traceback

# Trossen Imports
from trossen_arm_mujoco.utils import make_sim_env
from trossen_arm_mujoco.sim_env import TransferCubeTask
import trossen_arm_mujoco.constants as constants
from trossen_arm_mujoco.wrapper import TrossenMujocoGymWrapper, CAM_MAPPING, SIM_CAMERAS

# LeRobot Imports
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
lerobot_path = os.path.join(current_dir, "../../../lerobot_main_private/src")
if os.path.exists(lerobot_path):
    sys.path.append(lerobot_path)

try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
except ImportError:
    print("Could not import PI05Policy. Ensure requirements are installed.")
    sys.exit(1)

from transformers import AutoTokenizer

# --- Constants ---
FIXED_BOX_POSE = np.array([0.50, 0.30, 0.0125, 1, 0, 0, 0])
MAX_STATE_DIM = 32
TOKENIZER_MAX_LENGTH = 200
ACTIVE_STATE_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]

GRID_CAM_ORDER = [
    ("observation.images.static1", "High Cam"),
    ("observation.images.static2", "Low Cam"),
    ("observation.images.wrist1", "Left Wrist"),
    ("observation.images.wrist2", "Right Wrist"),
]


# --- Helpers ---

def load_norm_stats(policy_path: Path) -> dict:
    """Load normalization statistics from the checkpoint's preprocessor safetensors."""
    from safetensors.torch import load_file as load_safetensors

    stats_path = policy_path / "policy_preprocessor_step_2_normalizer_processor.safetensors"

    if not stats_path.exists():
        old_stats_path = policy_path / "assets" / "trossen" / "norm_stats.json"
        if old_stats_path.exists():
            with open(old_stats_path) as f:
                data = json.load(f)
            stats = data["norm_stats"]
            return {
                "state_mean": np.array(stats["state"]["mean"], dtype=np.float32),
                "state_std": np.array(stats["state"]["std"], dtype=np.float32),
                "action_mean": np.array(stats["actions"]["mean"], dtype=np.float32),
                "action_std": np.array(stats["actions"]["std"], dtype=np.float32),
            }
        raise FileNotFoundError(f"Norm stats not found at {stats_path}")

    stats_tensors = load_safetensors(stats_path)
    return {
        "state_mean": stats_tensors["observation.state.mean"].numpy(),
        "state_std": stats_tensors["observation.state.std"].numpy(),
        "action_mean": stats_tensors["action.mean"].numpy(),
        "action_std": stats_tensors["action.std"].numpy(),
    }


def pad_vector_np(vector: np.ndarray, new_dim: int) -> np.ndarray:
    """Pad vector to new_dim with zeros."""
    if vector.shape[-1] >= new_dim:
        return vector
    pad_width = new_dim - vector.shape[-1]
    return np.pad(vector, (0, pad_width), mode="constant", constant_values=0)


def discretize_state(state_normalized: np.ndarray) -> np.ndarray:
    """Discretize normalized state into 256 bins (matching openpi PaligemmaTokenizer.tokenize())."""
    return np.digitize(state_normalized, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1


def build_prompt(task: str, state_normalized: np.ndarray) -> str:
    """Build the PI0.5 prompt with task and discretized state."""
    state_padded = pad_vector_np(state_normalized, MAX_STATE_DIM)
    discretized = discretize_state(state_padded)
    state_str = " ".join(map(str, discretized))
    return f"Task: {task}, State: {state_str};\nAction: "


# --- Main ---

def run_evaluation(
    checkpoint: str,
    xml: str = "trossen_ai_scene_white_table.xml",
    output: str = "./eval_output.mp4",
    steps: int = 400,
    task: str = "pick up the cube",
    fps: int = 20,
    use_rtc: bool = False,
    rtc_horizon: int = 10,
    rtc_guidance: float = 10.0,
):
    checkpoint_path = Path(checkpoint)
    output_path = Path(output)

    print(f"[INFO] Inference mode: {'RTC' if use_rtc else 'standard'}")
    print(f"[INFO] Checkpoint:     {checkpoint_path}")
    print(f"[INFO] Scene XML:      {xml}")
    print(f"[INFO] Steps:          {steps}")
    print(f"[INFO] Output:         {output_path}")

    # Environment
    print("[INFO] Initializing Environment...")
    constants.BOX_POSE[0] = FIXED_BOX_POSE
    dm_env_instance = make_sim_env(
        task_class=TransferCubeTask,
        xml_file=xml,
        cam_list=SIM_CAMERAS,
        onscreen_render=False,
    )
    env = TrossenMujocoGymWrapper(dm_env_instance)
    print("[INFO] Environment Ready.")

    # Normalization stats
    print("[INFO] Loading normalization stats...")
    norm_stats = load_norm_stats(checkpoint_path)
    state_mean = torch.from_numpy(norm_stats["state_mean"])
    state_std = torch.from_numpy(norm_stats["state_std"])
    action_mean = norm_stats["action_mean"]
    action_std = norm_stats["action_std"]
    print(f"[INFO] State dim: {state_mean.shape[0]}, Action dim: {action_mean.shape[0]}")

    # Tokenizer
    print("[INFO] Loading PaliGemma tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    print("[INFO] Tokenizer loaded.")

    # Policy
    print(f"[INFO] Loading Policy...")
    try:
        policy = PI05Policy.from_pretrained(checkpoint_path)
        if use_rtc:
            policy.config.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=rtc_horizon,
                max_guidance_weight=rtc_guidance,
            )
            policy.init_rtc_processor()
            print(f"[INFO] RTC enabled — horizon={rtc_horizon}, guidance={rtc_guidance}")
        policy.eval()
        device = (
            policy.config.device
            if hasattr(policy.config, "device")
            else next(policy.parameters()).device
        )
        print(f"[INFO] Policy Loaded on {device}")
    except Exception as e:
        print(f"[ERROR] Failed to load policy: {e}")
        traceback.print_exc()
        return

    state_mean = state_mean.to(device)
    state_std = state_std.to(device)

    # Reset
    constants.BOX_POSE[0] = FIXED_BOX_POSE
    obs, _ = env.reset()

    # Video writer — 2x2 grid
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_img = obs["observation.images.static1"]
    h, w = sample_img.shape[:2]
    grid_h, grid_w = h * 2, w * 2
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid_w, grid_h)
    )
    print(f"[INFO] Recording 2x2 grid video ({grid_w}x{grid_h} @ {fps}fps)")

    done = False
    step = 0

    # RTC state
    prev_chunk_left_over = None
    action_chunk = None
    chunk_exec_idx = 0

    while not done and step < steps:
        # 1. Preprocess observation
        batch = {}
        raw_state = None

        for key, value in obs.items():
            if "images" in key:
                tensor = torch.from_numpy(value).permute(2, 0, 1).float() / 255.0
            elif "state" in key:
                raw_state = value.copy()
                active_state = value[ACTIVE_STATE_INDICES]
                tensor = torch.from_numpy(active_state).float()
                tensor = (tensor - state_mean.cpu()) / state_std.cpu()
            else:
                tensor = torch.from_numpy(value)
            batch[key] = tensor.unsqueeze(0).to(device)

        # 2. Tokenize language prompt
        active_raw_state = raw_state[ACTIVE_STATE_INDICES]
        state_for_prompt = (active_raw_state - norm_stats["state_mean"]) / norm_stats["state_std"]
        prompt = build_prompt(task, state_for_prompt)
        tokenized = tokenizer(
            prompt,
            max_length=TOKENIZER_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch["observation.language.tokens"] = tokenized["input_ids"].to(device)
        batch["observation.language.attention_mask"] = tokenized["attention_mask"].to(device).bool()

        # 3. Inference
        if not use_rtc:
            with torch.no_grad():
                action_tensor = policy.select_action(batch)
            action = action_tensor.squeeze(0).cpu().numpy()
            action = action * action_std + action_mean
        else:
            need_new_chunk = (action_chunk is None) or (chunk_exec_idx >= rtc_horizon)

            if need_new_chunk:
                t0 = time_module.perf_counter()
                with torch.no_grad():
                    new_chunk = policy.predict_action_chunk(
                        batch,
                        inference_delay=chunk_exec_idx if action_chunk is not None else 0,
                        prev_chunk_left_over=prev_chunk_left_over,
                        execution_horizon=rtc_horizon,
                    )
                inference_time = time_module.perf_counter() - t0

                if action_chunk is not None:
                    prev_chunk_left_over = action_chunk[:, chunk_exec_idx:, :]
                else:
                    prev_chunk_left_over = None

                action_chunk = new_chunk
                chunk_exec_idx = 0
                print(f"  [RTC] New chunk at step {step}, inference={inference_time*1000:.1f}ms")

            action_tensor = action_chunk[:, chunk_exec_idx, :]
            chunk_exec_idx += 1
            action = action_tensor.squeeze(0).cpu().numpy()
            action = action * action_std + action_mean

        # 4. Map Policy 14-dim -> Sim 16-dim
        if action.shape[0] == 14:
            sim_action = np.zeros(16, dtype=np.float32)
            sim_action[:6] = action[:6]
            sim_action[6] = action[6]
            sim_action[8:14] = action[7:13]
            sim_action[14] = action[13]
        else:
            sim_action = action

        # 5. Step
        obs, reward, terminated, truncated, _ = env.step(sim_action)
        done = terminated or truncated

        # 6. Record 2x2 grid frame
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for idx, (cam_key, label) in enumerate(GRID_CAM_ORDER):
            row, col = divmod(idx, 2)
            frame = obs[cam_key]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = frame_bgr
        writer.write(grid)

        step += 1
        if step % 20 == 0:
            print(f"Step {step}/{steps}: Reward={reward}")

    writer.release()
    print(f"[DONE] Video saved to {output_path} ({step} steps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate finetuned PI0.5 in Trossen MuJoCo sim")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--xml", default="trossen_ai_scene_white_table.xml", help="MuJoCo scene XML file")
    parser.add_argument("--output", default="./eval_output.mp4", help="Output video path")
    parser.add_argument("--steps", type=int, default=400, help="Number of episode steps")
    parser.add_argument("--task", default="pick up the cube", help="Language task instruction")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--rtc", action="store_true", help="Enable Real-Time Chunking inference")
    parser.add_argument("--rtc-horizon", type=int, default=10, help="RTC execution horizon")
    parser.add_argument("--rtc-guidance", type=float, default=10.0, help="RTC max guidance weight")
    args = parser.parse_args()

    run_evaluation(
        checkpoint=args.checkpoint,
        xml=args.xml,
        output=args.output,
        steps=args.steps,
        task=args.task,
        fps=args.fps,
        use_rtc=args.rtc,
        rtc_horizon=args.rtc_horizon,
        rtc_guidance=args.rtc_guidance,
    )
