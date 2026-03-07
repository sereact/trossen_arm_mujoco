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

# Default camera mapping: sim camera name -> LeRobot observation key
CAM_MAPPING = {
    "cam_high": "observation.images.static1",
    "cam_low": "observation.images.static2",
    "cam_left_wrist": "observation.images.wrist1",
    "cam_right_wrist": "observation.images.wrist2",
}
SIM_CAMERAS = list(CAM_MAPPING.keys())


class TrossenMujocoGymWrapper(gym.Env):
    """Wraps a Trossen dm_control environment into a Gymnasium interface.

    Observation keys:
      observation.images.<cam>  — (H, W, 3) uint8 images
      observation.state         — (N,) float32 joint positions

    Action space: Box derived from dm_control action_spec.
    """

    metadata = {"render_modes": []}

    def __init__(self, dm_env, cam_mapping: dict[str, str] | None = None):
        super().__init__()
        self._dm_env = dm_env
        self._cam_mapping = cam_mapping or CAM_MAPPING

        # Derive spaces from a reset
        time_step = self._dm_env.reset()
        dm_obs = time_step.observation

        obs_dict = {}
        if "images" in dm_obs:
            for sim_cam, key in self._cam_mapping.items():
                if sim_cam in dm_obs["images"]:
                    h, w, c = dm_obs["images"][sim_cam].shape
                    obs_dict[key] = spaces.Box(0, 255, (h, w, c), np.uint8)
        if "qpos" in dm_obs:
            dim = dm_obs["qpos"].shape[0]
            obs_dict["observation.state"] = spaces.Box(
                -np.inf, np.inf, (dim,), np.float32
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
        return self._obs(time_step.observation), {}

    def step(self, action):
        time_step = self._dm_env.step(action)
        obs = self._obs(time_step.observation)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        terminated = False
        truncated = time_step.last()
        return obs, reward, terminated, truncated, {}

    def _obs(self, dm_obs):
        out = {}
        if "images" in dm_obs:
            for sim_cam, key in self._cam_mapping.items():
                if sim_cam in dm_obs["images"]:
                    out[key] = dm_obs["images"][sim_cam].copy()
        if "qpos" in dm_obs:
            out["observation.state"] = dm_obs["qpos"].astype(np.float32)
        return out

    def close(self):
        pass
