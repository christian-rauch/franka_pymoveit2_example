#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from pymoveit2 import MoveIt2

from franka_msgs.action import Grasp, Move
from franka_msgs.msg import GraspEpsilon


class FrankaMove:
    def __init__(self, node) -> None:
        self.node = node

        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=[ "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",],
            base_link_name="panda_link0",
            end_effector_name="panda_hand_tcp",
            group_name="panda_arm", # tip: panda_link8
            use_move_group_action = True,
        )

        # reduce speed and acceleration
        self.moveit2.max_velocity = 0.1
        self.moveit2.max_acceleration = 0.1

        self.cli_gripper_move = ActionClient(self.node, Move, '/panda_gripper/move')
        self.cli_gripper_grasp = ActionClient(self.node, Grasp, '/panda_gripper/grasp')

        self.gripper_speed = float(0.05) # [m/s]
        self.gripper_width = float(0.08) # [m]
        self.gripper_force = float(1)    # [N]

    def _spin_action_complete(self, goal_future):
        self.node.executor.spin_until_future_complete(goal_future)
        goal_handle = goal_future.result()
        result_future = goal_handle.get_result_async()
        self.node.executor.spin_until_future_complete(result_future)
        return result_future.result().result

    def _action_send(self, action, goal):
        goal_future = action.send_goal_async(goal)
        result = self._spin_action_complete(goal_future)
        return result

    def move_to_pose(self, position, quat_xyzw):
        self.moveit2.move_to_pose(
            position=position,      # position xyz
            quat_xyzw=quat_xyzw,    # quaternion xyzw
        )
        return self._spin_action_complete(self.moveit2._MoveIt2__send_goal_future_move_action)

    def move_to_configuration(self, joint_positions):
        self.moveit2.move_to_configuration(joint_positions)
        return self._spin_action_complete(self.moveit2._MoveIt2__send_goal_future_move_action)

    def open(self):
        if not self.cli_gripper_move.server_is_ready():
            self.node.get_logger().error("no 'move' action server available")
            return False
        g = Move.Goal(
            width=self.gripper_width,    # move width [m]
            speed=self.gripper_speed,    # move speed [m/s]
        )
        result = self._action_send(self.cli_gripper_move, g)
        self.node.get_logger().info(f"gripper opened: {result.success}")
        return result.success

    def close(self):
        if not self.cli_gripper_grasp.server_is_ready():
            self.node.get_logger().error("no 'grasp' action server available")
            return False
        g = Grasp.Goal(
            width=float(0),              # grasp width [m]
            speed=self.gripper_speed,    # grasp speed [m/s]
            force=self.gripper_force,    # grasp force [N]
            epsilon=GraspEpsilon(inner=np.inf, outer=np.inf),
        )
        result = self._action_send(self.cli_gripper_grasp, g)
        self.node.get_logger().info(f"gripper closed: {result.success}")
        return result.success


def main():
    rclpy.init()

    node = Node("franka_move")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    franka = FrankaMove(node)

    # move to end-effector pose (panda_hand_tcp) in root (panda_link0)
    franka.move_to_pose(
        position=[0.4, 0.2, 0.3], # position xyz
        quat_xyzw=[1, 0, 0, 0], # quaternion xyzw
    )
    franka.close()

    # move to joint configuration
    ready_state = [0., -1/4 * np.pi, 0., -3/4 * np.pi, 0., 1/2 * np.pi, 1/4 * np.pi]
    franka.move_to_configuration(ready_state)
    franka.open()

    node.get_logger().info("DONE")


if __name__ == "__main__":
    main()
