#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <std_srvs/srv/trigger.hpp>

#include <chrono>
#include <thread>

#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>

class SwiftproControl : public rclcpp::Node
{
protected:
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reboot_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr shutdown_srv_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr start_swiftpro_srv_;
    pid_t swiftpro_pid_;
    // pid_t fake_jsp_pid;

    void start_swiftpro(const std::shared_ptr<std_srvs::srv::SetBool::Request> request, std::shared_ptr<std_srvs::srv::SetBool::Response> response);
    void reboot_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/, std::shared_ptr<std_srvs::srv::Trigger::Response> res);
    void shutdown_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/, std::shared_ptr<std_srvs::srv::Trigger::Response> res);
    void publish_joint_states();

public:
    SwiftproControl();
};

SwiftproControl::SwiftproControl()
    : Node("swiftpro_control"), swiftpro_pid_(0) //, fake_jsp_pid(0)
{
    start_swiftpro_srv_ = this->create_service<std_srvs::srv::SetBool>(
        "start_swiftpro",
        std::bind(&SwiftproControl::start_swiftpro, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Service swiftpro_control prepared.");
    reboot_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "turtlebot_reboot",
        std::bind(&SwiftproControl::reboot_callback, this, std::placeholders::_1, std::placeholders::_2));

    shutdown_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "turtlebot_shutdown",
        std::bind(&SwiftproControl::shutdown_callback, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "System control services ready: system_reboot & system_shutdown");

    std::thread([this]()
                { this->publish_joint_states(); })
        .detach();
}

void SwiftproControl::publish_joint_states()
{
    auto joint_pub = this->create_publisher<sensor_msgs::msg::JointState>("/turtlebot/joint_states", 10);
    std::vector<std::string> joint_names = {
        "swiftpro/joint1",
        "swiftpro/joint2",
        "swiftpro/passive_joint1",
        "swiftpro/passive_joint2",
        "swiftpro/passive_joint3",
        "swiftpro/joint3",
        "swiftpro/passive_joint5",
        "swiftpro/passive_joint7",
        "swiftpro/joint4",
        "swiftpro/passive_joint8"};
    while (rclcpp::ok())
    {
        if (swiftpro_pid_ == 0)
        {
            auto msg = sensor_msgs::msg::JointState();
            msg.header.stamp = this->get_clock()->now();
            msg.name = joint_names;
            msg.position = std::vector<double>(joint_names.size(), 0.0);
            msg.velocity.clear();
            msg.effort.clear();

            joint_pub->publish(msg);
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

void SwiftproControl::start_swiftpro(const std::shared_ptr<std_srvs::srv::SetBool::Request> request, std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
    if (request->data)
    {
        // Kill the fake joint_state_publisher if it's running
        // if (fake_jsp_pid != 0) {
        //     kill(-fake_jsp_pid, SIGINT);
        //     fake_jsp_pid = 0;
        // }

        // Activate the arm
        if (swiftpro_pid_ == 0)
        {
            RCLCPP_INFO(this->get_logger(), "Running swiftpro uarm node...");

            pid_t pid = fork();
            if (pid == 0)
            {
                setpgid(0, 0);
                execlp("ros2", "ros2", "launch", "turtlebot", "swiftpro_uarm-launch.py", nullptr);
                std::exit(1);
            }
            else if (pid > 0)
            {
                swiftpro_pid_ = pid;
                response->success = true;
                response->message = "Swiftpro uarm activated.";
            }
            else
            {
                response->success = false;
                response->message = "Error while trying to create fork.";
            }
        }
        else
        {
            response->success = true;
            response->message = "Swiftpro uarm activated yet.";
        }
    }
    else
    {
        // Deactivate the arm
        if (swiftpro_pid_ != 0)
        {
            RCLCPP_INFO(this->get_logger(), "Closing swiftpro uarm node...");
            kill(-swiftpro_pid_, SIGINT);
            swiftpro_pid_ = 0;
            response->success = true;
            response->message = "Swiftpro uarm deactivated.";
        }
        else
        {
            response->success = true;
            response->message = "Swiftpro uarms deactivated yet.";
        }

        // Launch the fake joint_state_publisher
        // pid_t pid = fork();
        // if (pid == 0) {
        //     setpgid(0, 0);
        //     execlp("ros2", "ros2", "run", "joint_state_publisher", "joint_state_publisher", "--ros-args", "-r", "__ns:=/turtlebot", nullptr);
        //     std::exit(1);
        // } else if (pid > 0) {
        //     fake_jsp_pid = pid;
        // }
    }
}

void SwiftproControl::reboot_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/, std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
    RCLCPP_WARN(this->get_logger(), "Reboot command received. System will reboot NOW!");
    res->success = true;
    res->message = "Rebooting system...";
    std::system("sudo reboot now");
}

void SwiftproControl::shutdown_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/, std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
    RCLCPP_WARN(this->get_logger(), "Shutdown command received. System will shutdown NOW!");
    res->success = true;
    res->message = "Shutting down system...";
    std::system("sudo shutdown now");
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto swiftpro_control = std::make_shared<SwiftproControl>();
    rclcpp::spin(swiftpro_control);
    rclcpp::shutdown();
    return 0;
}