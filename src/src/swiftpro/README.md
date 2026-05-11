### Firware update

Use AVRdude (check serial port):
avrdude -v -p atmega2560 -c wiring -P /dev/ttyACM0 -b 115200 -D -U flash:w:<filename.hex>:i
