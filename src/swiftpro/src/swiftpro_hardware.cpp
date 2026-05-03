/*
 *  swiftpro_hardware.cpp
 *  swiftpro
 *  
 *  Created by Patryk Cieslak on 05/02/2025.
 *  Copyright (c) 2025 Patryk Cieslak. All rights reserved.
 */

#include "swiftpro/swiftpro_hardware.hpp"

namespace swiftpro
{

hardware_interface::CallbackReturn SwiftProHardware::on_init(const hardware_interface::HardwareInfo & info)
{
	if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS)
	{
		return hardware_interface::CallbackReturn::ERROR;
	}

	swift_ = nullptr;
	cmd_.resize(4, 0.0);
	pos_.resize(4, 0.0);
	vel_.resize(4, 0.0);
	eff_.resize(4, 0.0);
	limit_switch_state_ = 0.0;
	pump_state_ = 0.0;
	pump_cmd_ = 0.0;
	
	for (const hardware_interface::ComponentInfo & joint : info_.joints)
	{
		if (joint.command_interfaces.size() != 1)
		{
			RCLCPP_FATAL(
				get_logger(), "Joint '%s' has %zu command interfaces found. 1 expected.",
				joint.name.c_str(), joint.command_interfaces.size());
			return hardware_interface::CallbackReturn::ERROR;
		}

		if (joint.command_interfaces[0].name != hardware_interface::HW_IF_VELOCITY)
		{
			RCLCPP_FATAL(
				get_logger(), "Joint '%s' have %s command interfaces found. '%s' expected.",
				joint.name.c_str(), joint.command_interfaces[0].name.c_str(), hardware_interface::HW_IF_VELOCITY);
			return hardware_interface::CallbackReturn::ERROR;
		}

		if (joint.state_interfaces.size() != 1)
		{
			RCLCPP_FATAL(
				get_logger(), "Joint '%s' has %zu state interface. 1 expected.", joint.name.c_str(),
				joint.state_interfaces.size());
			return hardware_interface::CallbackReturn::ERROR;
		}

		if (joint.state_interfaces[0].name != hardware_interface::HW_IF_POSITION)
		{
			RCLCPP_FATAL(
				get_logger(), "Joint '%s' have %s state interface. '%s' expected.", joint.name.c_str(), 
				joint.state_interfaces[0].name.c_str(), hardware_interface::HW_IF_POSITION);
			return hardware_interface::CallbackReturn::ERROR;
		}
	}

	if (info_.gpios.size() != 2)
  	{
    	RCLCPP_FATAL(
    		get_logger(), "SwiftPro has '%ld' GPIO components, '%d' expected.", info_.gpios.size(), 2);
    	return hardware_interface::CallbackReturn::ERROR;
  	}
  	
	if (info_.gpios[0].command_interfaces.size() != 0)
    {
		RCLCPP_FATAL(
			get_logger(), "GPIO component %s has '%ld' command interfaces, '%d' expected.",
        	info_.gpios[0].name.c_str(), info_.gpios[0].command_interfaces.size(), 0);
        return hardware_interface::CallbackReturn::ERROR;
    }
   	
  	if (info_.gpios[0].state_interfaces.size() != 1)
  	{
   		RCLCPP_FATAL(
      		get_logger(), "GPIO component %s has '%ld' state interfaces, '%d' expected.",
      		info_.gpios[0].name.c_str(), info_.gpios[0].state_interfaces.size(), 1);
    	return hardware_interface::CallbackReturn::ERROR;
  	}

	if (info_.gpios[1].command_interfaces.size() != 1)
    {
		RCLCPP_FATAL(
			get_logger(), "GPIO component %s has '%ld' command interfaces, '%d' expected.",
        	info_.gpios[1].name.c_str(), info_.gpios[1].command_interfaces.size(), 1);
        return hardware_interface::CallbackReturn::ERROR;
    }
   	
  	if (info_.gpios[1].state_interfaces.size() != 1)
  	{
   		RCLCPP_FATAL(
      		get_logger(), "GPIO component %s has '%ld' state interfaces, '%d' expected.",
      		info_.gpios[1].name.c_str(), info_.gpios[1].state_interfaces.size(), 1);
    	return hardware_interface::CallbackReturn::ERROR;
  	}
 
	return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> SwiftProHardware::export_state_interfaces()
{
    std::vector<hardware_interface::StateInterface> state_interfaces;
    for (size_t i = 0; i < info_.joints.size(); ++i) 
	{
    	state_interfaces.emplace_back(hardware_interface::StateInterface(
        	info_.joints[i].name, hardware_interface::HW_IF_POSITION, &pos_[i]));

    	state_interfaces.emplace_back(hardware_interface::StateInterface(
        	info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &vel_[i]));

    	state_interfaces.emplace_back(hardware_interface::StateInterface(
        	info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &eff_[i]));
  	}

	state_interfaces.emplace_back(hardware_interface::StateInterface(info_.gpios[0].name, "limit_switch", &limit_switch_state_));
	state_interfaces.emplace_back(hardware_interface::StateInterface(info_.gpios[1].name, "pump", &pump_state_));
	
 	return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> SwiftProHardware::export_command_interfaces()
{
	std::vector<hardware_interface::CommandInterface> command_interfaces;
  	for (size_t i = 0; i < info_.joints.size(); ++i) 
	{
    	command_interfaces.emplace_back(hardware_interface::CommandInterface(
        	info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &cmd_[i]));
  	}
	command_interfaces.emplace_back(hardware_interface::CommandInterface(info_.gpios[1].name, "pump", &pump_cmd_));

	return command_interfaces;
}

hardware_interface::CallbackReturn SwiftProHardware::on_configure(const rclcpp_lifecycle::State & previous_state)
{
	test_ = info_.hardware_parameters["test"] == "true";
	const std::string serial_port = info_.hardware_parameters["serial_port"]; 
	
	if(!test_)
	{
		swift_ = new SwiftPro(nullptr);
		if (!swift_->connect(serial_port))
		{	
			delete swift_;
			swift_ = nullptr;
			RCLCPP_FATAL_STREAM(get_logger(), "Unable to open port '" + serial_port + "'");
			return hardware_interface::CallbackReturn::ERROR;
		}	
	}
	RCLCPP_INFO_STREAM(get_logger(), "Manipulator connected on port '" + serial_port + "'");	
	return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn SwiftProHardware::on_cleanup(const rclcpp_lifecycle::State & previous_state)
{
	if(!test_ && swift_ != nullptr)
	{
		delete swift_;
		swift_ = nullptr;
	}
	RCLCPP_INFO_STREAM(get_logger(), "Cleanup done.");	
	return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn SwiftProHardware::on_activate(const rclcpp_lifecycle::State & previous_state)
{
	if(!test_)
	{
		swift_->setAutoupdate(false);
		if(!swift_->arm())
		{
			RCLCPP_FATAL(get_logger(), "Arming drives failed!");
			return hardware_interface::CallbackReturn::ERROR;
		}
		swift_->setAutoupdate(true);
	}
	RCLCPP_INFO_STREAM(get_logger(), "Drives armed.");	
	return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn SwiftProHardware::on_deactivate(const rclcpp_lifecycle::State & previous_state)
{
	if(!test_)
	{
		swift_->setAutoupdate(false);
		if(!swift_->disarm())
		{
			RCLCPP_FATAL(get_logger(), "Disarming drives failed!");
			return hardware_interface::CallbackReturn::ERROR;
		}
		swift_->setAutoupdate(true);
	}
	RCLCPP_INFO(get_logger(), "Drives disarmed.");
	return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type SwiftProHardware::read(const rclcpp::Time & time, const rclcpp::Duration & period)
{
	if(!test_)
	{
		std::vector<double> pos = swift_->getJointPositions();
		// Robot in simulation was built with Z axis down !!!
		pos_[0] = -pos[0];
		pos_[1] = -pos[1];
		pos_[2] = -pos[2];
		pos_[3] = -pos[3];
		pump_state_ = swift_->getPumpState() ? 1.0 : 0.0;
		limit_switch_state_ = swift_->getLimitSwitchState() ? 1.0 : 0.0;
	}
	else
	{
		for(size_t i=0; i<4; ++i)
			pos_[i] += cmd_[i]*0.01;
	}
	return hardware_interface::return_type::OK;
}

hardware_interface::return_type SwiftProHardware::write(const rclcpp::Time & time, const rclcpp::Duration & period)
{
	if(cmd_[0] == 0.0 && cmd_[1] == 0.0
 		&& cmd_[2] == 0.0 && cmd_[3] == 0.0)
	{
		if(!test_)
			swift_->stop();
		else
			RCLCPP_INFO_STREAM(get_logger(), "Manipulator stopped.");	
	}
	else
	{
		if(!test_)
		{
			std::vector<double> cmd(4);
			// Robot in simulation was built with Z axis down !!!
			cmd[0] = -cmd_[0];
			cmd[1] = -cmd_[1];
			cmd[2] = -cmd_[2];
			cmd[3] = -cmd_[3];
			swift_->requestJointVelocities(cmd);
		}
		else
			RCLCPP_INFO_STREAM(get_logger(), "Requesting joint velocities: [" << cmd_[0] << ", " << cmd_[1] << ", " << cmd_[2] << ", " << cmd_[3] << "]");	
 	}

	if(pump_cmd_ != pump_state_)
	{
		if(pump_cmd_ == 1.0)
			swift_->pumpOn();
		else
			swift_->pumpOff();
	}

	return hardware_interface::return_type::OK;
}

}

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(swiftpro::SwiftProHardware, hardware_interface::SystemInterface)