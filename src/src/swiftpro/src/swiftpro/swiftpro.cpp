/*
 *  swiftpro.cpp
 *  swiftpro
 *  
 *  Created by Patryk Cieslak on 19/01/2023.
 *  Copyright (c) 2023 Patryk Cieslak. All rights reserved.
 */

#include "swiftpro/swiftpro.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <math.h>

SwiftPro::SwiftPro(const std::function<void()>& updateCallback) : serial_('$', '#')
{
    serial_.registerPacketCallback(CMD_POSITION, std::bind(&SwiftPro::positionUpdate, this, std::placeholders::_1));
    jointPos_ = std::vector<double>(4, 0.0);
    pump_state_ = false;
    limit_switch_state_ = false;
    connected_ = false; 
    updateCallback_ = updateCallback;
}

SwiftPro::~SwiftPro()
{
    stop();
    disarm();
    serial_.closePort();
}

void SwiftPro::positionUpdate(SerialPacket* pck)
{
    if(pck->dataLen == 9)
    {
        jointPos_[0] = (double)( (int16_t)((uint16_t)pck->data[0] << 8 | pck->data[1]) )/(double)INT16_MAX * M_PI;
        jointPos_[1] = (double)( (int16_t)((uint16_t)pck->data[2] << 8 | pck->data[3]) )/(double)INT16_MAX * M_PI;
        jointPos_[2] = (double)( (int16_t)((uint16_t)pck->data[4] << 8 | pck->data[5]) )/(double)INT16_MAX * M_PI;
        jointPos_[3] = (double)( (int16_t)((uint16_t)pck->data[6] << 8 | pck->data[7]) )/(double)INT16_MAX * M_PI;
        pump_state_ = pck->data[8] & (1 << 1);
        limit_switch_state_ = pck->data[8] & (1 << 0);
        
        if(updateCallback_ != nullptr)
            updateCallback_();
    }
    delete pck;
}

std::vector<double> SwiftPro::getJointPositions() const
{
    return jointPos_;
}

bool SwiftPro::getPumpState()
{
    return pump_state_;
}

bool SwiftPro::getLimitSwitchState()
{
    return limit_switch_state_;
}

bool SwiftPro::connect(const std::string& port)
{
    if(connected_)
        return true;

    if(!serial_.openPort(port, B115200))
        return false;

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    
    //serial_.flush();

    SerialPacket pck;
    pck.cmd = CMD_CONNECT;
    pck.dataLen = 0;    
    if(!serial_.writePacket(pck))
    {
        return false;
    }

    SerialPacket* res = serial_.readPacket(CMD_CONNECT);
    if(res == nullptr)
    {
        return false;
    }
    delete res;
    connected_ = true;
    return true;
}

void SwiftPro::setAutoupdate(bool enabled)
{
    if(!connected_)
        return;

    SerialPacket pck;
    pck.cmd = enabled ? CMD_START_AUTOUPDATE : CMD_STOP_AUTOUPDATE;
    pck.dataLen = 0;
    serial_.writePacket(pck);
}

bool SwiftPro::arm()
{
    if(!connected_)
        false;

    SerialPacket pck;
    pck.cmd = CMD_ARM_DRIVES;
    pck.dataLen = 0;
    
    if(!serial_.writePacket(pck))
    {
        return false;
    }

    SerialPacket* res = serial_.readPacket(CMD_ARM_DRIVES);
    if(res == nullptr)
    {
        return false;
    }
    delete res;

    return true;
}

bool SwiftPro::disarm()
{
    if(!connected_)
        false;

    SerialPacket pck;
    pck.cmd = CMD_DISARM_DRIVES;
    pck.dataLen = 0;
    
    if(!serial_.writePacket(pck))
    {
        return false;
    }

    SerialPacket* res = serial_.readPacket(CMD_DISARM_DRIVES);
    if(res == nullptr)
    {
        return false;
    }
    delete res;

    return true;
}

void SwiftPro::stop()
{
    if(!connected_)
        return;

    SerialPacket pck;
    pck.cmd = CMD_STOP;
    pck.dataLen = 0;
    serial_.writePacket(pck);
}

void SwiftPro::requestJointVelocities(std::vector<double> vel)
{
    if(!connected_)
        return;
    
    int16_t drive_vel[4];
    drive_vel[0] = (int16_t)round(vel[0]/M_PI * INT16_MAX);
    drive_vel[1] = (int16_t)round(vel[1]/M_PI * INT16_MAX);
    drive_vel[2] = (int16_t)round(vel[2]/M_PI * INT16_MAX);
    drive_vel[3] = (int16_t)round(vel[3]/M_PI * INT16_MAX);
    
    SerialPacket pck;
    pck.cmd = CMD_SET_VELOCITY;
    pck.dataLen = 8;
    pck.data[0] = (uint8_t)(drive_vel[0] >> 8);
    pck.data[1] = (uint8_t)(drive_vel[0] & 0xFF);
    pck.data[2] = (uint8_t)(drive_vel[1] >> 8);
    pck.data[3] = (uint8_t)(drive_vel[1] & 0xFF);
    pck.data[4] = (uint8_t)(drive_vel[2] >> 8);
    pck.data[5] = (uint8_t)(drive_vel[2] & 0xFF);
    pck.data[6] = (uint8_t)(drive_vel[3] >> 8);
    pck.data[7] = (uint8_t)(drive_vel[3] & 0xFF);
    serial_.writePacket(pck);
}
    
void SwiftPro::pumpOn()
{
    if(!connected_)
        return;
    
    SerialPacket pck;
    pck.cmd = CMD_PUMP_ON;
    pck.dataLen = 0;
    serial_.writePacket(pck);
}

void SwiftPro::pumpOff()
{
    if(!connected_)
        return;

    SerialPacket pck;
    pck.cmd = CMD_PUMP_OFF;
    pck.dataLen = 0;
    serial_.writePacket(pck);
}

