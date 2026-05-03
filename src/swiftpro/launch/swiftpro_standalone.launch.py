from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Read configuration from files
    # 1. URDF
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("swiftpro"),
                    "urdf",
                    "swiftpro_standalone.urdf.xacro",
                ]
            ),
        ]
    )
    robot_description = {"robot_description": robot_description_content}
    
    # 2. ROS2 control setup 
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare("swiftpro"),
            "config",
            "swiftpro.yaml",
        ]
    )

    # Create nodes
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_controllers],
        output="both",
    )

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_velocity_controller", "--param-file", robot_controllers],
    )

    gpio_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gpio_controller", "--param-file", robot_controllers],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    swiftpro_rviz_node = Node(
        package="swiftpro_description",
        executable="swiftpro_rviz_node",
        output="both",
        parameters=[{"manipulator_namespace": "swiftpro"}],
        #remappings=[('joint_states', '......')]
    )

    # 3. Populate launch description
    nodes = [
        control_node,
        robot_state_pub_node,
        robot_controller_spawner,
        gpio_controller_spawner,
        joint_state_broadcaster_spawner,
        swiftpro_rviz_node
    ]

    return LaunchDescription(nodes)