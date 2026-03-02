"""
Evaluation Script for Finetuned PI0.5 on White Table Environment.
Records a 2x2 grid video from all 4 cameras for 300 steps.

Uses proper PaliGemma tokenization with state discretization and
action unnormalization matching the PI0.5 preprocessing pipeline.
"""

import os

# Force Headless Rendering (EGL) BEFORE importing dm_control or mujoco
os.environ["MUJOCO_GL"] = "egl"

import json
import time
import cv2
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import dm_env
from pathlib import Path
import traceback

# Trossen Imports
from trossen_arm_mujoco.utils import make_sim_env
from trossen_arm_mujoco.sim_env import TransferCubeTask
import trossen_arm_mujoco.constants as constants

FIXED_BOX_POSE = np.array([0.50, 0.30, 0.0125, 1, 0, 0, 0])
constants.BOX_POSE[0] = FIXED_BOX_POSE

# LeRobot Imports
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
lerobot_path = os.path.join(current_dir, "../lerobot_main_private/src")
if os.path.exists(lerobot_path):
    sys.path.append(lerobot_path)

try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
except ImportError:
    print("Could not import PI05Policy. Ensure requirements are installed.")
    sys.exit(1)

from transformers import AutoTokenizer

# --- Configuration ---
POLICY_PATH = Path(
    "/home/dzmitry/sandbox/trossen/pi05_models/transfer_cube1_40000/pretrained_model"
)
XML_FILE = "trossen_ai_scene_white_table.xml"
EPISODE_LENGTH = 400
VIDEO_OUTPUT = Path("/home/dzmitry/sandbox/trossen/media_output/pi05_finetuned/eval_transfer_cube1_40000.mp4")
VIDEO_FPS = 20
TASK_INSTRUCTION = "pick up the cube"
MAX_STATE_DIM = 32
TOKENIZER_MAX_LENGTH = 200

# Camera Mapping
CAM_MAPPING = {
    "cam_high": "observation.images.static1",
    "cam_low": "observation.images.static2",
    "cam_left_wrist": "observation.images.wrist1",
    "cam_right_wrist": "observation.images.wrist2",
}
SIM_CAMERAS = list(CAM_MAPPING.keys())


def load_norm_stats(policy_path: Path) -> dict:
    """Load normalization statistics from the checkpoint's preprocessor safetensors."""
    from safetensors.torch import load_file as load_safetensors
    
    # New checkpoint structure uses preprocessor safetensors
    stats_path = policy_path / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    
    if not stats_path.exists():
        # Fallback to old format if needed
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


class TrossenMujocoGymWrapper(gym.Env):
    """Wraps Trossen dm_control env to Gymnasium interface."""

    def __init__(self, env):
        self.env = env
        self.cam_list = SIM_CAMERAS

        time_step = self.env.reset()
        dm_obs = time_step.observation

        obs_space_dict = {}
        if "images" in dm_obs:
            for sim_cam, policy_cam_suffix in CAM_MAPPING.items():
                if sim_cam in dm_obs["images"]:
                    img = dm_obs["images"][sim_cam]
                    h, w, c = img.shape
                    obs_space_dict[policy_cam_suffix] = spaces.Box(
                        low=0, high=255, shape=(h, w, c), dtype=np.uint8
                    )
        if "qpos" in dm_obs:
            qpos = dm_obs["qpos"]
            state_dim = qpos.shape[0]
            obs_space_dict["observation.state"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
            )
        self.observation_space = spaces.Dict(obs_space_dict)

        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            shape=action_spec.shape,
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        time_step = self.env.reset()
        return self._format_observation(time_step.observation), {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._format_observation(time_step.observation)
        reward = time_step.reward if time_step.reward is not None else 0.0
        terminated = False
        truncated = time_step.last()
        return obs, reward, terminated, truncated, {}

    def _format_observation(self, dm_obs):
        sim_obs = {}
        if "images" in dm_obs:
            for sim_cam, policy_key in CAM_MAPPING.items():
                if sim_cam in dm_obs["images"]:
                    sim_obs[policy_key] = dm_obs["images"][sim_cam].copy()
        if "qpos" in dm_obs:
            sim_obs["observation.state"] = dm_obs["qpos"].astype(np.float32)
        return sim_obs


def run_evaluation():
    print("[INFO] Initializing Environment...")
    dm_env_instance = make_sim_env(
        task_class=TransferCubeTask,
        xml_file=XML_FILE,
        cam_list=SIM_CAMERAS,
        onscreen_render=False,
    )
    env = TrossenMujocoGymWrapper(dm_env_instance)
    sim_action_dim = env.action_space.shape[0]
    print("[INFO] Environment Ready.")

    # Load normalization stats
    print("[INFO] Loading normalization stats...")
    norm_stats = load_norm_stats(POLICY_PATH)
    state_mean = torch.from_numpy(norm_stats["state_mean"])
    state_std = torch.from_numpy(norm_stats["state_std"])
    action_mean = norm_stats["action_mean"]
    action_std = norm_stats["action_std"]
    print(
        f"[INFO] State dim: {state_mean.shape[0]}, Action dim: {action_mean.shape[0]}"
    )

    # Load PaliGemma tokenizer
    print("[INFO] Loading PaliGemma tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    print("[INFO] Tokenizer loaded.")

    # Load Policy
    print(f"[INFO] Loading Policy from {POLICY_PATH}...")
    try:
        policy = PI05Policy.from_pretrained(POLICY_PATH)
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

    # Move norm stats to device
    state_mean = state_mean.to(device)
    state_std = state_std.to(device)

    # Reset
    constants.BOX_POSE[0] = FIXED_BOX_POSE
    obs, _ = env.reset()

    # Prepare video writer — 2x2 grid of all cameras
    VIDEO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sample_img = obs["observation.images.static1"]
    h, w = sample_img.shape[:2]
    grid_h, grid_w = h * 2, w * 2
    writer = cv2.VideoWriter(
        str(VIDEO_OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (grid_w, grid_h)
    )
    GRID_CAM_ORDER = [
        ("observation.images.static1", "High Cam"),
        ("observation.images.static2", "Low Cam"),
        ("observation.images.wrist1", "Left Wrist"),
        ("observation.images.wrist2", "Right Wrist"),
    ]
    print(f"[INFO] Recording 2x2 grid video: {VIDEO_OUTPUT} ({grid_w}x{grid_h} @ {VIDEO_FPS}fps)")

    done = False
    step = 0

    while not done and step < EPISODE_LENGTH:
        # 1. Preprocess observation
        batch = {}
        raw_state = None

        # Active state indices: filter out padding dims 7 and 15 from 16-dim sim state
        ACTIVE_STATE_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]

        for key, value in obs.items():
            if "images" in key:
                # HWC uint8 -> CHW float [0, 1]
                tensor = torch.from_numpy(value).permute(2, 0, 1).float() / 255.0
            elif "state" in key:
                raw_state = value.copy()
                # Filter to 14 active dims before normalization
                active_state = value[ACTIVE_STATE_INDICES]
                tensor = torch.from_numpy(active_state).float()
                tensor = (tensor - state_mean.cpu()) / state_std.cpu()
            else:
                tensor = torch.from_numpy(value)
            batch[key] = tensor.unsqueeze(0).to(device)

        # 2. Tokenize language prompt with discretized state
        # Build prompt matching PI0.5 preprocessing pipeline
        active_raw_state = raw_state[ACTIVE_STATE_INDICES]
        state_for_prompt = (active_raw_state - norm_stats["state_mean"]) / norm_stats[
            "state_std"
        ]
        prompt = build_prompt(TASK_INSTRUCTION, state_for_prompt)

        tokenized = tokenizer(
            prompt,
            max_length=TOKENIZER_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch["observation.language.tokens"] = tokenized["input_ids"].to(device)
        batch["observation.language.attention_mask"] = tokenized["attention_mask"].to(
            device
        )

        # 3. Inference
        with torch.no_grad():
            action_tensor = policy.select_action(batch)

        # 4. Postprocess action — unnormalize
        action = action_tensor.squeeze(0).cpu().numpy()
        action = action * action_std + action_mean

        # 5. Map to sim action dims: Policy 14 -> Sim 16
        # sim_env.before_step always expects 16-dim actions
        # (6 left arm + 1 left gripper + 1 pad + 6 right arm + 1 right gripper + 1 pad)
        if action.shape[0] == 14:
            sim_action = np.zeros(16, dtype=np.float32)
            sim_action[:6] = action[:6]  # Left arm
            sim_action[6] = action[6]  # Left gripper
            # Index 7 padded
            sim_action[8:14] = action[7:13]  # Right arm
            sim_action[14] = action[13]  # Right gripper
            # Index 15 padded
        else:
            sim_action = action

        # 6. Step
        obs, reward, terminated, truncated, info = env.step(sim_action)
        done = terminated or truncated

        # 7. Record 2x2 grid frame from all cameras
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
            print(f"Step {step}/{EPISODE_LENGTH}: Reward={reward}")

    writer.release()
    print(f"[DONE] Video saved to {VIDEO_OUTPUT} ({step} steps)")


if __name__ == "__main__":
    run_evaluation()
