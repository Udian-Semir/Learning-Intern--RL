import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
def generate_launch_description():
    package_name = 'robot_model'
    # AR5 arms with RH6 dexterous hands mounted directly on both TCP frames.
    urdf_file_name = 'left_arm_with_wrist_camera.urdf'
    rviz_config_file = 'dual_hand.rviz'
    ros_domain_id = LaunchConfiguration('ros_domain_id')

    rviz_config_path = os.path.join(get_package_share_directory(package_name), 'rviz', rviz_config_file)
    urdf = os.path.join(
        get_package_share_directory(package_name),
        'urdf',
        urdf_file_name)
    
    # Read the URDF file content
    with open(urdf, 'r') as infp:
        robot_description_content = infp.read()

    # Wrap the URDF content in ParameterValue to specify it as a string parameter
    robot_description = ParameterValue(robot_description_content, value_type=str)

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    ))
    ld.add_action(DeclareLaunchArgument(
        'ros_domain_id',
        default_value='47',
        description='DDS domain used only by this robot visualization.'
    ))
    # Do not discover nodes on the LAN; the non-default DDS domain also
    # prevents other local robots on the default domain from replacing TF.
    ld.add_action(SetEnvironmentVariable(
        name='ROS_DOMAIN_ID', value=ros_domain_id))
    ld.add_action(SetEnvironmentVariable(
        name='ROS_LOCALHOST_ONLY', value='1'))

    ld.add_action(Node(
        package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}]
    ))

    ld.add_action(Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    ))

    ld.add_action(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d',rviz_config_path]

    ))
    
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
    ))

    return ld
