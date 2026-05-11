import os, yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue
from launch_ros.actions import PushRosNamespace
from launch.actions import TimerAction

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():

    robot_name_arg = DeclareLaunchArgument(
        'turtlebot_name',
        default_value='turtlebot',
        description='Name of the robot'
    )

    robot_xacro_path = DeclareLaunchArgument(
        'robot_xacro',
        default_value=PathJoinSubstitution([
            FindPackageShare('turtlebot_description'),
            'urdf',
            'turtlebot.urdf.xacro'
        ]),
        description='Path to the robot Xacro file'
    )

    robot_description = ParameterValue(
        Command([
            'xacro ',
            LaunchConfiguration('robot_xacro')
        ]),
        value_type=str
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description':robot_description}],
    )

    # mobile base
    share_dir   = get_package_share_directory('kobuki_node')
    params_file = os.path.join(share_dir, 'config', 'kobuki_node_params.yaml')

    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)['kobuki_ros_node']['ros__parameters']

    kobuki_ros_node = Node(
        package='kobuki_node',
        executable='kobuki_ros_node',
        output='both',
        parameters=[params]
    )

    # robot sensors
    kobuki_sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot'),
                'launch',
                'kobuki_sensors-launch.py'
            ])
        )
    )

    swiftpro_uarm_node = Node(
        package='turtlebot',
        executable='swiftpro_uarm_control',
        name='swiftpro_control',
        output='both'
    )

    return LaunchDescription([
        robot_name_arg,
        PushRosNamespace(LaunchConfiguration('turtlebot_name')),
        robot_xacro_path,
        robot_state_publisher,
        kobuki_ros_node,
        kobuki_sensors_launch,
        swiftpro_uarm_node,
    ])
