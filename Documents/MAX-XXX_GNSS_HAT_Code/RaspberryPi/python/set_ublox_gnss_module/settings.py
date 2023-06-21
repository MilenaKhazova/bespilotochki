import time
import serial

# Please enable hardware uart on pi with 'sudo raspi-config' command firstly

# Open /dev/ttyS0,note that the default baudrate is 9600 8N1
ser = serial.Serial('/dev/ttyS0',9600,timeout=1)
#ser.open()

# cmd1 enable GPS+BD+Galileo+QZSS
cmd1 =  b'\x06\x3E\x3C\x00\x00\x20\x20\x07\x00\x08\x10\x00\x01\x00\x01\x01\x01\x01\x03\x00\x00\x00\x01\x01\x02\x04\x08\x00\x00\x00\x01\x01\x03\x08\x10\x00\x01\x00\x01\x01\x04\x00\x08\x00\x00\x00\x01\x03\x05\x00\x03\x00\x01\x00\x01\x05\x06\x08\x0E\x00\x00\x00\x01\x01'

# cmd2 enable NMEA version 4.10 to ouput BD sentences
cmd2 = b'\xB5\x62\x06\x17\x14\x00\x00\x41\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00'

# Please refer the file of u-blox 8 /u-blox M8 Receiver description Including protocol specification in www.u-blox.com for more info 

# two 8bit crc var
ck_a = 0
ck_b = 0

# func of calculating the cmd1 CRC value
def calcu_crc(cmdx):
    global ck_a
    global ck_b
    ck_a = 0
    ck_b = 0
    for i in cmdx:
        #print(i,' ',end='')
        ck_a = ck_a + i
        ck_b = ck_b + ck_a

# calculate the the cmd1 
calcu_crc(cmd1)
#print(ck_a & 0xff)
#print(ck_b & 0xff)

# combine the cmd1 bytes,like that header + conten + crc
cmd1 = (b'\xb5\x62' + cmd1 + bytes([ck_a&0xff,ck_b&0xff])) 
#print(cmd1.hex())
#print(cmd1)


# calculate the the cmd2 
calcu_crc(cmd2)
#print(ck_a & 0xff)
#print(ck_b & 0xff)

# combine the cmd2 bytes,like that header + conten + crc
cmd2 = (b'\xb5\x62' + cmd2 + bytes([ck_a&0xff,ck_b&0xff])) 
#print(cmd2.hex())
#print(cmd2)



if True == ser.is_open:
    # next two lines show the nmea sentence from module
    #nmea = ser.read_until()
    #print(nmea.decode())

    # next lines send cmd1
    ser.write(cmd1)
    # next lines send cmd2
    ser.write(cmd2)

    time.sleep(1)
    ser.close()
