from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Base simulation (turtlebot_hoi scenario: robot + aruco box)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('turtlebot_simulation_1'),
                    'launch',
                    'turtlebot_hoi.launch.py'
                ])
            ])
        ),

        # Autonomous pick-and-place VMS node
        Node(
            package='hoi_control',
            executable='lab2_pick_place_vms_node.py',
            name='pick_place_vms',
            output='screen',
            emulate_tty=True,
        ),
    ])
