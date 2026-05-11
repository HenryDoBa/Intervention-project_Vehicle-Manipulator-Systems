/*
 *  swiftpro_hardware.hpp
 *  swiftpro
 *  
 *  Created by Patryk Cieslak on 05/02/2025.
 *  Copyright (c) 2025 Patryk Cieslak. All rights reserved.
 */

#ifndef __swiftpro_swiftpro_hardware__
#define __swiftpro_swiftpro_hardware__

#include "rclcpp/rclcpp.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "swiftpro/swiftpro.h"

namespace swiftpro
{

class SwiftProHardware : public hardware_interface::SystemInterface
{
public:
	RCLCPP_SHARED_PTR_DEFINITIONS(SwiftProHardware)
	
	hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
	hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
	hardware_interface::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous_state) override;
  	hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
	hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
	hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
	hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;
	std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
	std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

private:
	SwiftPro* swift_;
	std::vector<double> cmd_;
	std::vector<double> eff_;
	std::vector<double> vel_;
	std::vector<double> pos_;
	double pump_state_;
	double pump_cmd_;
	double limit_switch_state_;
	bool test_;
};

}

#endif // __swiftpro_swiftpro_hardware__