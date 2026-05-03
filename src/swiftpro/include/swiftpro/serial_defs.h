/*
 *  serial_defs.h
 *  swiftpro
 *  
 *  Created by Patryk Cieslak on 19/01/2023.
 *  Copyright (c) 2023 Patryk Cieslak. All rights reserved.
 */

#ifndef __swiftpro_serial_defs__
#define __swiftpro_serial_defs__

#include <cstdint>

enum CommandCode : uint8_t
{
    CMD_CONNECT = 0x00,			// NO Data
    CMD_START_AUTOUPDATE,       // NO Data
    CMD_STOP_AUTOUPDATE,        // NO Data
    CMD_ARM_DRIVES,             // NO Data
    CMD_DISARM_DRIVES,          // NO Data
    CMD_STOP,                   // NO Data
    CMD_POSITION,               // OUT Data(9): 2B->BASE 2B-> ARML 2B-> ARMR 2B-> EE 1B -> Pump and limit switch status
    CMD_SET_VELOCITY,		    // IN Data(16): 4B->BASE 4B-> ARML 4B-> ARMR 4B-> EE
    CMD_PUMP_ON,                // NO Data
    CMD_PUMP_OFF,               // NO Data
	CMD_RESET = 0xFD,  			// IN Data(1): 1B->0xCD
    CMD_ERROR = 0xFE,  			// OUT Data(2): 1B->command, 1B->error code
    CMD_INVALID = 0xFF 			// NO Data
};

enum ErrorCode : uint8_t
{
	ERR_INVALID_PACKET = 0x00,
	ERR_UNSUPPORTED_CMD,
	ERR_INVALID_DATA_LEN,
	ERR_INVALID_PAYLOAD,
};

static inline CommandCode byte2Command(const uint8_t b)
{
    if(b >= (uint8_t)CMD_CONNECT && b <= (uint8_t)CMD_PUMP_OFF)
        return (CommandCode)b;
    else if(b == (uint8_t)CMD_RESET)
        return CMD_RESET;
    else if(b == (uint8_t)CMD_ERROR)
        return CMD_ERROR;
    else
        return CMD_INVALID;
}

#endif