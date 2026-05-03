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
    robot_xacro_path = DeclareLaunchArgument(
        'robot_xacro',
        default_value=PathJoinSubstitution([
            FindPackageShare('swiftpro_description'),
            'urdf',
            'swiftpro_standalone.urdf.xacro'
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

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen'
    )

    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[{'robot_description': robot_description,},
        PathJoinSubstitution([
            FindPackageShare('swiftpro'),
            'config',
            'swiftpro.yaml'
        ])],
        output='screen',
    )

    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/turtlebot/controller_manager'],
        output='screen'
    )

    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare("swiftpro"),
            "config",
            "swiftpro.yaml",
        ]
    )

    joint_velocity_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_velocity_controller',  '--param-file', robot_controllers],
        output='screen'
    )

    gpio_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gpio_controller', '--param-file', robot_controllers],
        output='screen'
    )

    return LaunchDescription([
        robot_xacro_path,
        robot_state_publisher,
        joint_state_publisher,
        controller_manager,
        TimerAction(
            period=10.0,
            actions=[joint_state_broadcaster]
        ),
        TimerAction(
            period=20.0,
            actions=[joint_velocity_controller, gpio_controller]
        ),
    ])
