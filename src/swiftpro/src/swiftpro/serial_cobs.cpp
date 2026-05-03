#include "swiftpro/serial_cobs.h"

#include <cstring> // Error integer and strerror() function
#include <chrono>
#include <vector>
#include <unistd.h> // write(), read(), close()
#include <fcntl.h> // Contains file controls like O_RDWR
#include <iostream>

//Static
uint8_t SerialCOBS::CRC8(const uint8_t* data, uint8_t len)
{
    uint8_t* p = (uint8_t*)data;
    uint8_t crc = 0x00;
    uint8_t poly = 0x07;
    uint8_t i;

    while(len--)
    {
        crc = crc ^ *p++;

        for(i=0; i<8; ++i)
        {
            if(crc & 0x80)
                crc = (uint8_t)((crc << 1) ^ poly);
            else
                crc = (uint8_t)(crc << 1);
        }
    }
    return crc;
}

void SerialCOBS::cobsEncode(const uint8_t* in, uint8_t* out, const uint8_t len)
{
    uint8_t blockPtr = 0;
    uint8_t blockLen = 1;
    out[len+1] = 0x00;

    for(uint8_t i=0; i<len; ++i)
    {
        if(in[i] == 0x00)
        {
            out[blockPtr] = blockLen;
            blockPtr += blockLen;
            blockLen = 1;
        }
        else
        {
            out[i+1] = in[i];
            ++blockLen;
        }
    }
    out[blockPtr] = blockLen;
}

bool SerialCOBS::cobsDecode(const uint8_t* in, uint8_t* out, const uint8_t len)
{
    uint8_t blockPtr = 0;
    uint8_t blockLen;
    uint8_t i;
    do
    {
        blockLen = in[blockPtr];
        if(blockPtr + blockLen >= len)
            return false;
        for(i=1; i<blockLen; ++i)
            *out++ = in[blockPtr+i];
        *out++ = 0x00;
        blockPtr += blockLen;
    } while (blockPtr < len-1);

    return true;
}

//Members
SerialCOBS::SerialCOBS(uint8_t txPreambule, uint8_t rxPreambule)
    : txPreambule_(txPreambule), rxPreambule_(rxPreambule)
{
    serialPort_ = -1;
    timeout_ = 2000;
}

SerialCOBS::~SerialCOBS()
{
    closePort();
}

void SerialCOBS::registerPacketCallback(CommandCode cmd, const std::function<void(SerialPacket*)>& callback)
{
    processPacket_[cmd] = callback;
}

bool SerialCOBS::openPort(const std::string& port, speed_t baudrate)
{
    if(serialPort_ > 0)
        closePort();

    serialPort_ = open(port.c_str(), O_RDWR);
    if(serialPort_ < 0)
    {
        printf("Error %i from open: %s\n", errno, strerror(errno));
        return false;
    }

    // Create new termios struct, we call it 'tty' for convention
    struct termios tty;

    // Read in existing settings, and handle any error
    if(tcgetattr(serialPort_, &tty) != 0) 
    {
        printf("Error %i from tcgetattr: %s\n", errno, strerror(errno));
        return false;
    }

    tty.c_cflag &= ~PARENB; // Clear parity bit, disabling parity (most common)
    tty.c_cflag &= ~CSTOPB; // Clear stop field, only one stop bit used in communication (most common)
    tty.c_cflag &= ~CSIZE; // Clear all bits that set the data size 
    tty.c_cflag |= CS8; // 8 bits per byte (most common)
    tty.c_cflag &= ~CRTSCTS; // Disable RTS/CTS hardware flow control (most common)
    tty.c_cflag |= CREAD | CLOCAL; // Turn on READ & ignore ctrl lines (CLOCAL = 1)

    tty.c_lflag &= ~ICANON; // Raw mode
    tty.c_lflag &= ~ECHO; // Disable echo
    tty.c_lflag &= ~ECHOE; // Disable erasure
    tty.c_lflag &= ~ECHONL; // Disable new-line echo
    tty.c_lflag &= ~ISIG; // Disable interpretation of INTR, QUIT and SUSP
    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // Turn off s/w flow ctrl
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL); // Disable any special handling of received bytes

    tty.c_oflag &= ~OPOST; // Prevent special interpretation of output bytes (e.g. newline chars)
    tty.c_oflag &= ~ONLCR; // Prevent conversion of newline to carriage return/line feed
 
    tty.c_cc[VTIME] = 0;
    tty.c_cc[VMIN] = 0;

    // Set in/out baud rate
    cfsetispeed(&tty, baudrate);
    cfsetospeed(&tty, baudrate);

    // Save tty settings, also checking for error
    if (tcsetattr(serialPort_, TCSANOW, &tty) != 0) 
    {
        printf("Error %i from tcsetattr: %s\n", errno, strerror(errno));
        return false;
    }

    tcflush(serialPort_, TCIOFLUSH);

    // Create and start listening thread
    listenThread_ = std::thread(&SerialCOBS::listen, this);

    return true;
}

void SerialCOBS::closePort()
{
    if(serialPort_ > 0)
    {
        int serialPortCopy_ = serialPort_;
        serialPort_ = -1;
        if(listenThread_.joinable())
            listenThread_.join();
        close(serialPortCopy_);
    }
}

void SerialCOBS::flush()
{
    if(serialPort_ > 0)
        tcflush(serialPort_, TCIOFLUSH);
}

void SerialCOBS::encodePacket(const SerialPacket* pck, uint8_t* bytes, uint8_t* len)
{
	//Convert packet to bytes
    uint8_t buffer[4+pck->dataLen];
	buffer[0] = txPreambule_;
	buffer[1] = (uint8_t)pck->cmd;
	buffer[2] = pck->dataLen;
	for(uint8_t i=0; i<pck->dataLen; ++i)
		buffer[3+i] = pck->data[i];

	//Append CRC
	buffer[3+pck->dataLen] = CRC8(buffer, 3+pck->dataLen);

	//COBS
	cobsEncode(buffer, bytes, 4+pck->dataLen);
	*len = 4+pck->dataLen+2;
}

bool SerialCOBS::decodePacket(const uint8_t* bytes, size_t len, SerialPacket* pck)
{
	//Check data length
	if(len == 0 || len > 4+MAX_PACKET_DATA_LEN+2)
		return false;

	//COBS
    uint8_t buffer[len];
	if(!cobsDecode(bytes, buffer, len))
        return false;

	//Check preambule
	if(buffer[0] != rxPreambule_)
		return false;

	//Check CRC
	if(CRC8(buffer, len-3) != buffer[len-3])
		return false;

	//Fill packet
	pck->cmd = byte2Command(buffer[1]);
	pck->dataLen = buffer[2];
	if(pck->dataLen != len-6)
		return false;
	for(uint8_t i=0; i<pck->dataLen; ++i)
		pck->data[i] = buffer[3+i];
	return true;
}

bool SerialCOBS::writePacket(const SerialPacket& pck)
{
    if(serialPort_ < 0)
        return false;

    uint8_t len;
    uint8_t buffer[4+pck.dataLen+2];
    encodePacket(&pck, buffer, &len);
    writeMutex_.lock();
    ssize_t tx = write(serialPort_, buffer, len);
    writeMutex_.unlock();
    return tx == len;
}

SerialPacket* SerialCOBS::readPacket(CommandCode expected)
{
    if(serialPort_ < 0)
        return nullptr;
    auto timeoutStart = std::chrono::steady_clock::now();
    do
    {
        if(rxPackets_.size() > 0)
        {
            auto it = rxPackets_.begin();
            do
            {
                SerialPacket* pck = *it++;
                rxPackets_.pop_front();
                if(pck->cmd == expected)
                    return pck;
            }
            while(it != rxPackets_.end());
        }
    }
    while(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - timeoutStart).count() < timeout_);
    return nullptr;
}

void SerialCOBS::listen()
{
    std::vector<uint8_t> buffer;
    uint8_t chunk[8];
    while(serialPort_ > 0)
    {
        ssize_t rx = read(serialPort_, chunk, sizeof(chunk));
        if(rx > 0)
        {   
            for(ssize_t i=0; i<rx; ++i)
            {
                buffer.push_back(chunk[i]);
                if(buffer.back() == 0x00)
                {
                    SerialPacket* pck = new SerialPacket;
                    if(decodePacket(buffer.data(), buffer.size(), pck))
                    {
                        auto it = processPacket_.find(pck->cmd);
                        if(it != processPacket_.end())
                            it->second(pck);
                        else
                            rxPackets_.push_back(pck);
                    }
                    else
                        delete pck;
                    buffer.clear();
                }
            }
        }
    }
}