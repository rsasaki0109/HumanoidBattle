"""RViz で実 G1 メッシュを動かす launch（ROS2 Jazzy, v0）。

actuator-space IK（`robotdance retarget-ik`）で得た .rdmotion の実関節角を、motion server が
/joint_states へ配信し、robot_state_publisher が実 URDF と合わせて TF を出す → RViz で本物の
G1 が動く。

使い方:
  ros2 launch robotdance_ros2/launch/g1_rviz.launch.py \\
      urdf:=/path/to/g1_23dof.urdf rdmotion:=/path/to/g1_joints.rdmotion.json

⚠️ URDF / mesh は同梱しない（利用者が g1_description を取得）。本番では .rdmotion を
sim_certificate で検証してから再生すること（ここでは安全レビュー前提で --allow-uncertified）。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    urdf = LaunchConfiguration("urdf")
    rdmotion = LaunchConfiguration("rdmotion")
    speed = LaunchConfiguration("speed")

    return LaunchDescription([
        DeclareLaunchArgument("urdf", description="ロボット URDF（g1_23dof.urdf 等）"),
        DeclareLaunchArgument("rdmotion", description="retarget-ik で得た .rdmotion JSON"),
        DeclareLaunchArgument("speed", default_value="1.0"),
        # 実 URDF を robot_description に。
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": ParameterValue(
                Command(["cat ", urdf]), value_type=str)}],
        ),
        Node(package="rviz2", executable="rviz2", output="screen"),
        # motion server: /joint_states へ実関節角を配信。
        ExecuteProcess(
            cmd=["python3", "-m", "robotdance_ros2.motion_server_node",
                 rdmotion, "--speed", speed, "--allow-uncertified"],
            output="screen",
        ),
    ])
