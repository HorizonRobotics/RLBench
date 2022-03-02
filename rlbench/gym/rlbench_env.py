from typing import Union, Dict, List
import logging

import gym
import numpy as np
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete, Continuous
from rlbench.environment import Environment
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig, CameraConfig


class RLBenchEnv(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, task_class, observation_mode='state', obs_config=None):
        self._observation_mode = observation_mode
        if observation_mode != 'custom':
            obs_config = ObservationConfig()
            if observation_mode == 'state':
                # 'state' mode: low-dim state
                obs_config.set_all_high_dim(False)
                obs_config.set_all_low_dim(True)
            elif observation_mode == 'vision':
                # 'vision' mode: rgbd + low-dim state
                camera_config = CameraConfig(
                    rgb=True, depth=False, point_cloud=False, mask=False)
                obs_config = ObservationConfig(
                    left_shoulder_camera=camera_config,
                    right_shoulder_camera=camera_config,
                    overhead_camera=camera_config,
                    wrist_camera=camera_config,
                    front_camera=camera_config)
                # For low-dim state, only provide the robot arm states
                obs_config.set_all_low_dim(True)
                obs_config.task_low_dim_state = False
            else:
                raise ValueError(
                    'Unrecognised observation_mode: %s.' % observation_mode)
        else:
            assert obs_config is not None, (
                "You must need to provide a customized obs_config.")
            logging.info(f"Using customized observation config:\n {obs_config}")
        self._obs_config = obs_config

        action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self.task = self.env.get_task(task_class)

        _, obs = self.task.reset()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.env.action_shape)
        self.observation_space = self._extract_obs_space(
            self._extract_obs(obs))

        self._gym_cam = None

    def _setup_gym_cam(self, render_mode):
        if self._gym_cam is None and render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def _cast_uint8(self, img: np.ndarray) -> np.ndarray:
        return np.clip((img * 255.).astype(np.uint8), 0, 255)

    def _extract_obs_space(self, obs: Observation) -> gym.spaces.Space:
        if isinstance(obs, dict):
            space = dict()
            for k, v in obs.items():
                if k.endswith('_rgb') or k.endswith('_depth'):
                    space[k] = spaces.Box(
                        low=0, high=255, shape=v.shape, dtype=np.uint8)
                else:
                    space[k] = spaces.Box(
                        low=-np.inf, high=np.inf, shape=v.shape)
            return spaces.Dict(space)
        else:
            return spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape)

    def _extract_obs(self, obs: Observation) -> Union[
        np.ndarray, Dict[str, np.ndarray]]:

        if self._observation_mode == 'state':
            return obs.get_low_dim_data()
        else:
            extracted_obs = {"proprioceptive_state": obs.get_proprioceptive_state()}
            if len(obs.get_task_state()) > 0:
                extracted_obs["task_state"] = obs.get_task_state()
            keys = []
            # custom observations
            for k, v in self._obs_config.__dict__.items():
                if k.endswith('_camera'):
                    assert isinstance(v, CameraConfig)
                    # TODO: extracting mask, etc
                    if v.rgb:
                        keys.append(k.replace('_camera', '_rgb'))
                    if v.depth:
                        keys.append(k.replace('_camera', '_depth'))
                    if v.point_cloud:
                        keys.append(k.replace('_camera', '_point_cloud'))

            camera_obs = dict()
            for k in keys:
                val = getattr(obs, k)
                if k.endswith('_rgb'):
                    camera_obs[k] = self._cast_uint8(val)
                elif k.endswith('_depth'):
                    camera_obs[k] = np.expand_dims(
                        self._cast_uint8(val), axis=-1)
                else:
                    camera_obs[k] = val
            extracted_obs.update(camera_obs)
            return extracted_obs

    def render(self, mode: str='human') -> Union[None, np.ndarray]:
        self._setup_gym_cam(mode)
        if mode == 'rgb_array':
            frame = self._gym_cam.capture_rgb()
            return self._cast_uint8(frame)

    def reset(self, variation: bool=False) -> (
        List[str], Dict[str, np.ndarray]):
        """
        Args:
            variation: if True, then before resetting the task, we first sample
                a variation index which is used by ``init_episode()`` of the task.
        """
        if variation:
            self.task.sample_variation()
        descriptions, obs = self.task.reset()
        #del descriptions  # Not used.
        return descriptions, self._extract_obs(obs)

    def step(self, action: np.ndarray) -> (
        Dict[str, np.ndarray], float, bool, dict):
        obs, reward, terminate, success = self.task.step(action)
        info = {'success': False, 'failure': False}
        if terminate:
            key = 'success' if success else 'failure'
            info[key] = True
        return self._extract_obs(obs), reward, terminate, info

    def seed(self, seed: int=None) -> None:
        if seed is not None:
            np.random.seed(seed)

    @property
    def pyrep(self):
        return self.env._pyrep

    def close(self) -> None:
        self.env.shutdown()
