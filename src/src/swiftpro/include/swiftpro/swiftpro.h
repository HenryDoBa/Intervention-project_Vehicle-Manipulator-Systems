/*
 *  swiftpro.h
 *  swiftpro
 *  
 *  Created by Patryk Cieslak on 19/01/2023.
 *  Copyright (c) 2023 Patryk Cieslak. All rights reserved.
 */

#ifndef __swiftpro_swiftpro__
#define __swiftpro_swiftpro__

#include "swiftpro/serial_cobs.h"
#include <vector>
#include <functional>

class SwiftPro
{
public: 
    SwiftPro(const std::function<void()>& updateCallback);
    ~SwiftPro();

    bool connect(const std::string& port);
    void setAutoupdate(bool enabled);
    bool arm();
    bool disarm();
    void stop();
    void requestJointVelocities(std::vector<double> vel);
    void pumpOn();
    void pumpOff();

    std::vector<double> getJointPositions() const;
    bool getPumpState();
    bool getLimitSwitchState();
    
private:
    void positionUpdate(SerialPacket* pck);
    
    SerialCOBS serial_;
    bool connected_;
    std::vector<double> jointPos_;
    std::function<void()> updateCallback_;
    bool pump_state_;
    bool limit_switch_state_;
};

#endif
