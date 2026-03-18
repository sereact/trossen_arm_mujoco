"""
Gymnasium wrapper for Trossen bimanual dm_control environments.

Converts a dm_control Trossen environment into a standard Gymnasium interface
with observation keys matching LeRobot conventions.

Usage:
    from trossen_gym_wrapper import TrossenMujocoGymWrapper, CAM_MAPPING, SIM_CAMERAS
    from trossen_arm_mujoco.utils import make_sim_env
    from trossen_arm_mujoco.sim_env import TransferCubeTask

    dm_env = make_sim_env(task_class=TransferCubeTask, xml_file="scene.xml",
                          cam_list=SIM_CAMERAS, onscreen_render=False)
    env = TrossenMujocoGymWrapper(dm_env)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Default camera mapping: sim camera name -> obs dict key under "pixels"
CAM_MAPPING = {
    "cam_high": "static1",
    "cam_low": "static2",
    "cam_left_wrist": "wrist1",
    "cam_right_wrist": "wrist2",
}
SIM_CAMERAS = list(CAM_MAPPING.keys())

# Active joint indices (skip padding at 7 and 15): 6 left + 1 gripper + 6 right + 1 gripper = 14
ACTIVE_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]


class TrossenMujocoGymWrapper(gym.Env):
    """Wraps a Trossen dm_control environment into a Gymnasium interface.

    Observation keys (compatible with LeRobot preprocess_observation):
      pixels/<cam>   — (H, W, 3) uint8 images  (nested dict under "pixels")
      agent_pos      — (N,) float32 joint positions

    Action space: Box derived from dm_control action_spec.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, dm_env, cam_mapping: dict[str, str] | None = None, render_cam: str = "cam_high", max_episode_steps: int = 400):
        super().__init__()
        self._dm_env = dm_env
        self._cam_mapping = cam_mapping or CAM_MAPPING
        self._render_cam = render_cam
        self._last_dm_obs = None
        self._max_episode_steps = max_episode_steps

        # Derive spaces from a reset
        time_step = self._dm_env.reset()
        dm_obs = time_step.observation

        obs_dict = {}
        pixels_spaces = {}
        if "images" in dm_obs:
            for sim_cam, key in self._cam_mapping.items():
                if sim_cam in dm_obs["images"]:
                    h, w, c = dm_obs["images"][sim_cam].shape
                    pixels_spaces[key] = spaces.Box(0, 255, (h, w, c), np.uint8)
        if pixels_spaces:
            obs_dict["pixels"] = spaces.Dict(pixels_spaces)

        if "qpos" in dm_obs:
            obs_dict["agent_pos"] = spaces.Box(
                -np.inf, np.inf, (len(ACTIVE_JOINT_INDICES),), np.float32
            )
        self.observation_space = spaces.Dict(obs_dict)

        action_spec = self._dm_env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time_step = self._dm_env.reset()
        self._last_dm_obs = time_step.observation
        return self._obs(time_step.observation), {}

    def step(self, action):
        # Expand 14-dim action back to 16-dim (insert padding at indices 7 and 15)
        action_16 = np.zeros(16, dtype=np.float32)
        action_16[:7] = action[:7]       # left arm (6) + left gripper (1)
        action_16[7] = action[6]         # pad = copy of left gripper
        action_16[8:15] = action[7:14]   # right arm (6) + right gripper (1)
        action_16[15] = action[13]       # pad = copy of right gripper
        time_step = self._dm_env.step(action_16)
        self._last_dm_obs = time_step.observation
        obs = self._obs(time_step.observation)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        terminated = False
        truncated = time_step.last()
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self._last_dm_obs is None or "images" not in self._last_dm_obs:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self._last_dm_obs["images"][self._render_cam].copy()

    def _obs(self, dm_obs):
        out = {}
        if "images" in dm_obs:
            pixels = {}
            for sim_cam, key in self._cam_mapping.items():
                if sim_cam in dm_obs["images"]:
                    pixels[key] = dm_obs["images"][sim_cam].copy()
            if pixels:
                out["pixels"] = pixels
        if "qpos" in dm_obs:
            out["agent_pos"] = dm_obs["qpos"][ACTIVE_JOINT_INDICES].astype(np.float32)
        return out

    def close(self):
        pass
