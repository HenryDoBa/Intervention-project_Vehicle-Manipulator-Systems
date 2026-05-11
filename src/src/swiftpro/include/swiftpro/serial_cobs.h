/*
 *  serial_cobs.h
 *  swiftpro
 *  
 *  Created by Patryk Cieslak on 19/01/2023.
 *  Copyright (c) 2023 Patryk Cieslak. All rights reserved.
 */

#ifndef __swiftpro_serial_cobs__
#define __swiftpro_serial_cobs__

#include <string>
#include <thread>
#include <deque>
#include <map>
#include <functional>
#include <cstdint>
#include <termios.h>
#include <mutex>

#include "swiftpro/serial_defs.h"

#define MAX_PACKET_DATA_LEN     16

//Packet structure: | PREAMBULE | CMD | DATA_LEN | DATA[0] .... DATA[DATA_LEN-1] | CRC8 |
struct SerialPacket
{
    CommandCode cmd;
    uint8_t dataLen;
    uint8_t data[MAX_PACKET_DATA_LEN];
};

//! Packet-based serial communication class with COBS encoding.
class SerialCOBS
{
public:
    SerialCOBS(uint8_t txPreambule, uint8_t rxPreambule);
    ~SerialCOBS();

    void registerPacketCallback(CommandCode cmd, const std::function<void(SerialPacket*)>& callback);
    bool openPort(const std::string& port, speed_t baudrate);
    void closePort();
    bool writePacket(const SerialPacket& pck);
    SerialPacket* readPacket(CommandCode expected);
    void flush();

private: 
    void listen();
    void encodePacket(const SerialPacket* pck, uint8_t* bytes, uint8_t* len);
    bool decodePacket(const uint8_t* bytes, size_t len, SerialPacket* pck);
    static uint8_t CRC8(const uint8_t* data, uint8_t len);
    static void cobsEncode(const uint8_t* in, uint8_t* out, const uint8_t len);
    static bool cobsDecode(const uint8_t* in, uint8_t* out, const uint8_t len);

    int serialPort_;
    unsigned int timeout_;
    std::thread listenThread_;
    std::mutex writeMutex_;
    uint8_t txPreambule_;
    uint8_t rxPreambule_;
    std::deque<SerialPacket*> rxPackets_;
    std::map<CommandCode, std::function<void(SerialPacket*)>> processPacket_;
};

#endif