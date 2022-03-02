from abc import abstractmethod

import numpy as np

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.scene import Scene


def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


class GripperActionMode(object):

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass


class Discrete(GripperActionMode):
    """Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open

    def _actuate(self, action, scene):
        done = False
        while not done:
            done = scene.robot.gripper.actuate(action, velocity=0.2)
            scene.pyrep.step()
            scene.task.step()

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')
        open_condition = all(
            x > 0.9 for x in scene.robot.gripper.get_open_amount())
        current_ee = 1.0 if open_condition else 0.0
        action = float(action[0] > 0.5)

        if current_ee != action:
            done = False
            if not self._detach_before_open:
                self._actuate(action, scene)
            if action == 0.0 and self._attach_grasped_objects:
                # If gripper close action, the check for grasp.
                for g_obj in scene.task.get_graspable_objects():
                    scene.robot.gripper.grasp(g_obj)
            else:
                # If gripper open action, the check for un-grasp.
                scene.robot.gripper.release()
            if self._detach_before_open:
                self._actuate(action, scene)
            if action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()

    def action_shape(self, scene: Scene) -> tuple:
        return 1,


class Continuous(GripperActionMode):
    """Control the gripper in a continuous manner.

    Action values > 0. will open the gripper, and values < 0.
    will close the gripper. The magnitude of value will be multiplied to a max
    velocity to decide the open/close velocity.

    Note that unlike ``Discrete`` which always open/close the gripper to the end
    by executing a loop, this action mode only actuates once each time.
    """

    def __init__(self, max_velocity: float=0.2):
        """
        Args:
            max_velocity: the max velocity of opening and closing
        """
        self._max_velocity = max_velocity

    def _actuate(self, action, scene):
        #velocity = self._max_velocity * np.clip(action[0], -1., 1.)
        action = float(action[0] > 0.)
        scene.robot.gripper.actuate(action, velocity=self._max_velocity)
        scene.pyrep.step()
        scene.task.step()

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene.robot))
        if not (1. >= action[0] >= -1.):
            raise InvalidActionError(
                'Gripper action expected to be within -1. and 1.')
        self._actuate(action, scene)

    def action_shape(self, scene: Scene) -> tuple:
        return 1,
