#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/set_bool.hpp"

#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>

class SwiftproControl : public rclcpp::Node
{
protected:
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr start_swiftpro_srv_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr load_swiftpro_arm_srv_;
    pid_t swiftpro_pid_;
    pid_t fake_jsp_pid;

    void start_swiftpro(const std::shared_ptr<std_srvs::srv::SetBool::Request> request, std::shared_ptr<std_srvs::srv::SetBool::Response> response);
    void load_swiftpro_arm(const std::shared_ptr<std_srvs::srv::SetBool::Request> request, std::shared_ptr<std_srvs::srv::SetBool::Response> response);

public:
    SwiftproControl();
};

SwiftproControl::SwiftproControl()
    : Node("swiftpro_control"), swiftpro_pid_(0), fake_jsp_pid(0)
{
    start_swiftpro_srv_ = this->create_service<std_srvs::srv::SetBool>(
        "start_swiftpro",
        std::bind(&SwiftproControl::start_swiftpro, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Service swiftpro_control prepared.");

    load_swiftpro_arm_srv_ = this->create_service<std_srvs::srv::SetBool>(
        "load_swiftpro_arm",
        std::bind(&SwiftproControl::load_swiftpro_arm, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Service load_swiftpro_arm prepared.");

    pid_t pid = fork();
    if (pid == 0)
    {
        setpgid(0, 0);
        execlp("ros2", "ros2", "run", "joint_state_publisher", "joint_state_publisher", "--ros-args", "-r", "__ns:=/turtlebot", nullptr);
        RCLCPP_INFO(this->get_logger(), "Running joint_state_publisher by default");
        std::exit(1);
    }
    // if (pid > 0)
    // {
    //     fake_jsp_pid = pid;
    // }
    if (pid > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10000));

        kill(-pid, SIGINT);
        pid = 0;

        RCLCPP_INFO(this->get_logger(), "joint_state_publisher stopped");
    }
}

void SwiftproControl::start_swiftpro(const std::shared_ptr<std_srvs::srv::SetBool::Request> request, std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
    if (request->data)
    {
        // Kill the fake joint_state_publisher if it's running
        if (fake_jsp_pid != 0)
        {
            kill(-fake_jsp_pid, SIGINT);
            fake_jsp_pid = 0;
        }

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
            response->message = "Swiftpro uarms deactivated.";
        }
    }
}

void SwiftproControl::load_swiftpro_arm(const std::shared_ptr<std_srvs::srv::SetBool::Request> request, std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
    if (request->data)
    {
        if (fake_jsp_pid != 0)
        {
            response->success = true;
            response->message = "Swiftpro uarm is already loaded.";
        }

        if (swiftpro_pid_ != 0)
        {
            RCLCPP_INFO(this->get_logger(), "Swiftpro uarm is active");
            response->success = false;
            response->message = "Swiftpro uarm is already loaded.";
        }
        else
        {
            pid_t pid = fork();
            if (pid == 0)
            {
                setpgid(0, 0);
                execlp("ros2", "ros2", "run", "joint_state_publisher", "joint_state_publisher", "--ros-args", "-r", "__ns:=/turtlebot", nullptr);
                std::exit(1);
            }
            if (pid > 0)
            {
                fake_jsp_pid = pid;
                response->success = true;
                response->message = "Swiftpro uarm loaded.";
                std::this_thread::sleep_for(std::chrono::milliseconds(10000));

                kill(-fake_jsp_pid, SIGINT);
                fake_jsp_pid = 0;

                RCLCPP_INFO(this->get_logger(), "Swiftpro joint_state_publisher stopped");
            }
        }
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto swiftpro_control = std::make_shared<SwiftproControl>();
    rclcpp::spin(swiftpro_control);
    rclcpp::shutdown();
    return 0;
}