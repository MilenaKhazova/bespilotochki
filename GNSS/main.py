import serial
import time 
import string
import pynmea2  

while True: 
    ser=serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1)
    dataout =pynmea2.NMEAStreamReader() 
    newdata=ser.readline()
    print(newdata)
    #print(newdata)
