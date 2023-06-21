#!/usr/bin/python
# -*- coding:utf-8 -*-
import time
import smbus
import math
import numpy as np
import matplotlib.pyplot as plt

Gyro  = [0,0,0]
Accel = [0,0,0]
Mag   = [0,0,0]
pitch = 0.0
roll  = 0.0
yaw   = 0.0
pu8data=[0,0,0,0,0,0,0,0]
U8tempX=[0,0,0,0,0,0,0,0,0]
U8tempY=[0,0,0,0,0,0,0,0,0]
U8tempZ=[0,0,0,0,0,0,0,0,0]
GyroOffset=[0,0,0]
Ki = 1.0
Kp = 4.50
q0 = 1.0
q1=q2=q3=0.0
angles=[0.0,0.0,0.0]
true                                 =0x01
false                                =0x00
# define QMI8658 and AK09918 Device I2C address
I2C_ADD_IMU_QMI8658                  = 0x6B
I2C_ADD_IMU_AK09918                  = 0x0C

# define QMI8658 Register
QMI8658_CTRL7_ACC_ENABLE    = 0x01
QMI8658_CTRL7_GYR_ENABLE    = 0x02

QMI8658Register_Ctrl1       = 0x02 # SPI Interface and Sensor Enable
QMI8658Register_Ctrl2       = 0x03 # Accelerometer control.
QMI8658Register_Ctrl3       = 0x04 # Gyroscope control.
QMI8658Register_Ctrl5       = 0x06 # Data processing settings.
QMI8658Register_Ctrl7       = 0x08 # Sensor enabled status.

QMI8658Register_Ax_L        = 0x35 # Accelerometer X axis least significant byte.
QMI8658Register_Ax_H        = 0x36 # Accelerometer X axis most significant byte.
QMI8658Register_Ay_L        = 0x37 # Accelerometer Y axis least significant byte.
QMI8658Register_Ay_H        = 0x38 # Accelerometer Y axis most significant byte.
QMI8658Register_Az_L        = 0x39 # Accelerometer Z axis least significant byte.
QMI8658Register_Az_H        = 0x3A # Accelerometer Z axis most significant byte.
QMI8658Register_Gx_L        = 0x3B # Gyroscope X axis least significant byte.
QMI8658Register_Gx_H        = 0x3C # Gyroscope X axis most significant byte.
QMI8658Register_Gy_L        = 0x3D # Gyroscope Y axis least significant byte.
QMI8658Register_Gy_H        = 0x3E # Gyroscope Y axis most significant byte.
QMI8658Register_Gz_L        = 0x3F # Gyroscope Z axis least significant byte.
QMI8658Register_Gz_H        = 0x40 # Gyroscope Z axis most significant byte.

QMI8658Register_Tempearture_L = 0x33 # tempearture low.
QMI8658Register_Tempearture_H = 0x34 # tempearture high.

QMI8658AccRange_2g          = 0x00 << 4 # !< \brief +/- 2g range
QMI8658AccRange_4g          = 0x01 << 4 # !< \brief +/- 4g range
QMI8658AccRange_8g          = 0x02 << 4 # !< \brief +/- 8g range
QMI8658AccRange_16g         = 0x03 << 4 # !< \brief +/- 16g range

QMI8658AccOdr_8000Hz        = 0x00 # !< \brief High resolution 8000Hz output rate.
QMI8658AccOdr_4000Hz        = 0x01 # !< \brief High resolution 4000Hz output rate. 
QMI8658AccOdr_2000Hz        = 0x02 # !< \brief High resolution 2000Hz output rate.
QMI8658AccOdr_1000Hz        = 0x03 # !< \brief High resolution 1000Hz output rate.
QMI8658AccOdr_500Hz         = 0x04 # !< \brief High resolution 500Hz output rate.
QMI8658AccOdr_250Hz         = 0x05 # !< \brief High resolution 250Hz output rate.
QMI8658AccOdr_125Hz         = 0x06 # !< \brief High resolution 125Hz output rate.
QMI8658AccOdr_62_5Hz        = 0x07 # !< \brief High resolution 62.5Hz output rate.
QMI8658AccOdr_31_25Hz       = 0x08 # !< \brief High resolution 31.25Hz output rate.
QMI8658AccOdr_LowPower_128Hz = 0x0C # !< \brief Low power 128Hz output rate.
QMI8658AccOdr_LowPower_21Hz  = 0x0D # !< \brief Low power 21Hz output rate.
QMI8658AccOdr_LowPower_11Hz  = 0x0E # !< \brief Low power 11Hz output rate.
QMI8658AccOdr_LowPower_3Hz   = 0x0F # !< \brief Low power 3Hz output rate.

QMI8658GyrRange_16dps       = 0 << 4 # !< \brief +-16 degrees per second. 
QMI8658GyrRange_32dps       = 1 << 4 # !< \brief +-32 degrees per second. 
QMI8658GyrRange_64dps       = 2 << 4 # !< \brief +-64 degrees per second. 
QMI8658GyrRange_128dps      = 3 << 4 # !< \brief +-128 degrees per second. 
QMI8658GyrRange_256dps      = 4 << 4 # !< \brief +-256 degrees per second. 
QMI8658GyrRange_512dps      = 5 << 4 # !< \brief +-512 degrees per second. 
QMI8658GyrRange_1024dps     = 6 << 4 # !< \brief +-1024 degrees per second. 
QMI8658GyrRange_248dps      = 7 << 4 # !< \brief +-2048 degrees per second. 

QMI8658GyrOdr_8000Hz        = 0x00 # !< \brief High resolution 8000Hz output rate.
QMI8658GyrOdr_4000Hz        = 0x01 # !< \brief High resolution 8000Hz output rate.
QMI8658GyrOdr_2000Hz        = 0x02 # !< \brief High resolution 8000Hz output rate.
QMI8658GyrOdr_1000Hz        = 0x03 # !< \brief High resolution 8000Hz output rate.
QMI8658GyrOdr_500Hz         = 0x04 # !< \brief High resolution 8000Hz output rate.
QMI8658GyrOdr_250Hz         = 0x05 # !< \brief High resolution 8000Hz output rate.
QMI8658GyrOdr_125Hz         = 0x06 # !< \brief High resolution 8000Hz output rate.
QMI8658GyrOdr_62_5Hz        = 0x07 # !< \brief High resolution 8000Hz output rate.
QMI8658GyrOdr_31_25Hz       = 0x08 # !< \brief High resolution 8000Hz output rate.

# define QMI8658 Register  end

# define AK09918 Register
AK09918_WIA1       = 0x00    # Company ID
AK09918_WIA2       = 0x01    # Device ID
AK09918_RSV1       = 0x02    # Reserved 1
AK09918_RSV2       = 0x03    # Reserved 2
AK09918_ST1        = 0x10    # DataStatus 1
AK09918_HXL        = 0x11    # X-axis data 
AK09918_HXH        = 0x12
AK09918_HYL        = 0x13    # Y-axis data
AK09918_HYH        = 0x14
AK09918_HZL        = 0x15    # Z-axis data
AK09918_HZH        = 0x16
AK09918_TMPS       = 0x17    # Dummy
AK09918_ST2        = 0x18    # Datastatus 2
AK09918_CNTL1      = 0x30    # Dummy
AK09918_CNTL2      = 0x31    # Control settings
AK09918_CNTL3      = 0x32    # Control settings

AK09918_SRST_BIT   = 0x01    # Soft Reset
AK09918_HOFL_BIT   = 0x08    # Sensor Over Flow
AK09918_DOR_BIT    = 0x02    # Data Over Run
AK09918_DRDY_BIT   = 0x01    # Data Ready


AK09918_POWER_DOWN = 0x00
AK09918_NORMAL     = 0x01
AK09918_CONTINUOUS_10HZ  = 0x02
AK09918_CONTINUOUS_20HZ  = 0x04
AK09918_CONTINUOUS_50HZ  = 0x06
AK09918_CONTINUOUS_100HZ = 0x08
AK09918_SELF_TEST         = 0x10 # ignored by switchMode() and initialize(), call selfTest() to use this mode
# define AK09918 Register  end

class IMU(object):
  def __init__(self):
    self._bus = smbus.SMBus(1)    
    self.Pprev = np.identity(6)
    if self._read_byte(I2C_ADD_IMU_QMI8658,0x00) != 0x05:
      print("QMI8658_init fail\n")
      return 0
    else:
      print("QMI8658_init slave=0x%x  QMI8658Register_WhoAmI=0x%x 0x%x\n" % (I2C_ADD_IMU_QMI8658,self._read_byte(I2C_ADD_IMU_QMI8658,0x00),self._read_byte(I2C_ADD_IMU_QMI8658,0x01)))
      self._write_byte(I2C_ADD_IMU_QMI8658,QMI8658Register_Ctrl1,0x60) # 设置为I2C通信

      self._write_byte(I2C_ADD_IMU_QMI8658,QMI8658Register_Ctrl2,QMI8658AccRange_2g | QMI8658AccOdr_1000Hz) # 设置加速度模式  2g 1000Hz
      self._write_byte(I2C_ADD_IMU_QMI8658,QMI8658Register_Ctrl3,QMI8658GyrRange_512dps | QMI8658GyrOdr_500Hz) # 设置陀螺仪模式  512dps 500Hz
      self._write_byte(I2C_ADD_IMU_QMI8658,QMI8658Register_Ctrl5,0x00) # 传感器数据处理设置
      self._write_byte(I2C_ADD_IMU_QMI8658,QMI8658Register_Ctrl7,QMI8658_CTRL7_ACC_ENABLE | QMI8658_CTRL7_GYR_ENABLE | 0x80) # 使能加速度跟陀螺仪传感器
      self.QMI8658_GyroOffset()
    if self._read_byte(I2C_ADD_IMU_AK09918,AK09918_WIA2) != 0x0C:
      print("AK09918 init fail\r\n")
      return 0
    else:
      print("AK09918 init Success\r\n")
      self._write_byte(I2C_ADD_IMU_AK09918,AK09918_CNTL3,AK09918_SRST_BIT) # 复位AK09918
      self._write_byte(I2C_ADD_IMU_AK09918,AK09918_CNTL2,AK09918_CONTINUOUS_20HZ) # 设置磁力计模式

  def _read_byte(self,address,cmd):
    return self._bus.read_byte_data(address,cmd)

  def _read_block(self,address, reg, length=1):
    return self._bus.read_i2c_block_data(address, reg, length)

  def _read_u16(self,address,cmd):
    LSB = self._bus.read_byte_data(address,cmd)
    MSB = self._bus.read_byte_data(address,cmd+1)
    return (MSB	<< 8) + LSB

  def _write_byte(self,address,cmd,val):
    self._bus.write_byte_data(address,cmd,val)
    time.sleep(0.0001)

  def QMI8658_Gyro_Accel_Read(self):
    data =self._read_block(I2C_ADD_IMU_QMI8658,QMI8658Register_Ax_L, 12)
    Accel[0] = (data[1]<<8)|data[0]
    Accel[1] = (data[3]<<8)|data[2]
    Accel[2] = (data[5]<<8)|data[4]
    Gyro[0]  = ((data[7]<<8)|data[6]) - GyroOffset[0]
    Gyro[1]  = ((data[9]<<8)|data[8]) - GyroOffset[1]
    Gyro[2]  = ((data[11]<<8)|data[10]) - GyroOffset[2]
    Accel[0] = Accel[0]
    Accel[1] = Accel[1]
    Accel[2] = Accel[2]
    #print(data)
    if Accel[0]>=32767:             #Solve the problem that Python shift will not overflow
      Accel[0]=Accel[0]-65535
    elif Accel[0]<=-32767:
      Accel[0]=Accel[0]+65535
    if Accel[1]>=32767:
      Accel[1]=Accel[1]-65535
    elif Accel[1]<=-32767:
      Accel[1]=Accel[1]+65535
    if Accel[2]>=32767:
      Accel[2]=Accel[2]-65535
    elif Accel[2]<=-32767:
      Accel[2]=Accel[2]+65535
    if Gyro[0]>=32767:
      Gyro[0]=Gyro[0]-65535
    elif Gyro[0]<=-32767:
      Gyro[0]=Gyro[0]+65535
    if Gyro[1]>=32767:
      Gyro[1]=Gyro[1]-65535
    elif Gyro[1]<=-32767:
      Gyro[1]=Gyro[1]+65535
    if Gyro[2]>=32767:
      Gyro[2]=Gyro[2]-65535
    elif Gyro[2]<=-32767:
      Gyro[2]=Gyro[2]+65535

  def AK09918_MagRead(self):
    counter = 20
    while(counter>0):
      time.sleep(0.01)
      u8Data = self._read_byte(I2C_ADD_IMU_AK09918,AK09918_ST1)
      if ((u8Data & 0x01)!= 0):
        break
      counter -= 1
    
    if counter!=0:
      for i in range(0,8):
        pu8data =self._read_block(I2C_ADD_IMU_AK09918,AK09918_HXL, 8)
        U8tempX[i] = (pu8data[1]<<8)|pu8data[0]
        U8tempY[i] = (pu8data[3]<<8)|pu8data[2]
        U8tempZ[i] = (pu8data[5]<<8)|pu8data[4]
      Mag[0]=(U8tempX[0]+U8tempX[1]+U8tempX[2]+U8tempX[3]+U8tempX[4]+U8tempX[5]+U8tempX[6]+U8tempX[7])/8
      Mag[1]=(U8tempY[0]+U8tempY[1]+U8tempY[2]+U8tempY[3]+U8tempY[4]+U8tempY[5]+U8tempY[6]+U8tempY[7])/8
      Mag[2]=(U8tempZ[0]+U8tempZ[1]+U8tempZ[2]+U8tempZ[3]+U8tempZ[4]+U8tempZ[5]+U8tempZ[6]+U8tempZ[7])/8
    if Mag[0]>=32767:            #Solve the problem that Python shift will not overflow
      Mag[0]=Mag[0]-65535
    elif Mag[0]<=-32767:
      Mag[0]=Mag[0]+65535
    if Mag[1]>=32767:
      Mag[1]=Mag[1]-65535
    elif Mag[1]<=-32767:
      Mag[1]=Mag[1]+65535
    if Mag[2]>=32767:
      Mag[2]=Mag[2]-65535
    elif Mag[2]<=-32767:
      Mag[2]=Mag[2]+65535

  def QMI8658_readTemp(self):
      temp = self._read_block(I2C_ADD_IMU_QMI8658,QMI8658Register_Tempearture_L, 2)
      return float((temp[1] << 8) | temp[0]) / 256.0

  def QMI8658_GyroOffset(self):
    s32TempGx = 0
    s32TempGy = 0
    s32TempGz = 0
    for i in range(0,32):
      self.QMI8658_Gyro_Accel_Read()
      s32TempGx += Gyro[0]
      s32TempGy += Gyro[1]
      s32TempGz += Gyro[2]
      time.sleep(0.01)
    GyroOffset[0] = s32TempGx >> 5
    GyroOffset[1] = s32TempGy >> 5
    GyroOffset[2] = s32TempGz >> 5

  def imuAHRSupdate(self,gx,gy,gz,ax,ay,az,mx,my,mz):    
    norm=0.0
    hx = hy = hz = bx = bz = 0.0
    vx = vy = vz = wx = wy = wz = 0.0
    exInt = eyInt = ezInt = 0.0
    ex=ey=ez=0.0 
    halfT = 0.024
    global q0
    global q1
    global q2
    global q3
    q0q0 = q0 * q0
    q0q1 = q0 * q1
    q0q2 = q0 * q2
    q0q3 = q0 * q3
    q1q1 = q1 * q1
    q1q2 = q1 * q2
    q1q3 = q1 * q3
    q2q2 = q2 * q2   
    q2q3 = q2 * q3
    q3q3 = q3 * q3          

    norm = float(1/math.sqrt(ax * ax + ay * ay + az * az))     
    ax = ax * norm
    ay = ay * norm
    az = az * norm

    norm = float(1/math.sqrt(mx * mx + my * my + mz * mz))      
    mx = mx * norm
    my = my * norm
    mz = mz * norm

    # compute reference direction of flux
    hx = 2 * mx * (0.5 - q2q2 - q3q3) + 2 * my * (q1q2 - q0q3) + 2 * mz * (q1q3 + q0q2)
    hy = 2 * mx * (q1q2 + q0q3) + 2 * my * (0.5 - q1q1 - q3q3) + 2 * mz * (q2q3 - q0q1)
    hz = 2 * mx * (q1q3 - q0q2) + 2 * my * (q2q3 + q0q1) + 2 * mz * (0.5 - q1q1 - q2q2)         
    bx = math.sqrt((hx * hx) + (hy * hy))
    bz = hz     

    # estimated direction of gravity and flux (v and w)
    vx = 2 * (q1q3 - q0q2)
    vy = 2 * (q0q1 + q2q3)
    vz = q0q0 - q1q1 - q2q2 + q3q3
    
    wx = 2 * bx * (0.5 - q2q2 - q3q3) + 2 * bz * (q1q3 - q0q2)
    wy = 2 * bx * (q1q2 - q0q3) + 2 * bz * (q0q1 + q2q3)
    wz = 2 * bx * (q0q2 + q1q3) + 2 * bz * (0.5 - q1q1 - q2q2)  

    # error is sum of cross product between reference direction of fields and direction measured by sensors
    ex = (ay * vz - az * vy) + (my * wz - mz * wy)
    ey = (az * vx - ax * vz) + (mz * wx - mx * wz)
    ez = (ax * vy - ay * vx) + (mx * wy - my * wx)

    if (ex != 0.0 and ey != 0.0 and ez != 0.0):
      exInt = exInt + ex * Ki * halfT
      eyInt = eyInt + ey * Ki * halfT  
      ezInt = ezInt + ez * Ki * halfT

      gx = gx + Kp * ex + exInt
      gy = gy + Kp * ey + eyInt
      gz = gz + Kp * ez + ezInt

    q0 = q0 + (-q1 * gx - q2 * gy - q3 * gz) * halfT
    q1 = q1 + (q0 * gx + q2 * gz - q3 * gy) * halfT
    q2 = q2 + (q0 * gy - q1 * gz + q3 * gx) * halfT
    q3 = q3 + (q0 * gz + q1 * gy - q2 * gx) * halfT  

    norm = float(1/math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3))
    q0 = q0 * norm
    q1 = q1 * norm
    q2 = q2 * norm
    q3 = q3 * norm

  def icm20948CalAvgValue(self):
    MotionVal[0]=Gyro[0]/32.8
    MotionVal[1]=Gyro[1]/32.8
    MotionVal[2]=Gyro[2]/32.8
    MotionVal[3]=Accel[0]
    MotionVal[4]=Accel[1]
    MotionVal[5]=Accel[2]
    MotionVal[6]=Mag[0]
    MotionVal[7]=Mag[1]
    MotionVal[8]=Mag[2]
    
  def predict(self):
    
    halfT = 0.024
    self.Xprev = self.XCurr
    self.Q = np.identity(6)*10e-8 
    self.Pcur = self.Pprev + self.Q
    self.C = np.concatenate((np.zeros((3,3)), np.identity(3)), axis = 1)
    self.z = self.Ycur - np.matmul(self.C,self.Xprev)
    sigmakv = .1 #Дисперсию подобрать
    R = np.identity(3)*sigmakv    
    self.Sk = np.matmul(np.matmul(self.C,self.Pcur),self.C.transpose()) + R
    self.Kk = np.matmul(np.matmul(self.Pcur,self.C.transpose()),np.linalg.inv(self.Sk))
    self.XCurr = self.Xprev + np.matmul(self.Kk,self.z)
    tmp1 = np.identity(6)-np.matmul(self.Kk,self.C)
    self.Pprev = np.matmul(np.matmul(tmp1, self.Pcur),tmp1.transpose()) + np.matmul(np.matmul(self.Kk, R), self.Kk.transpose())
    roll = self.XCurr[0]
    pitch = self.XCurr[1]
    yaw = self.XCurr[2]
    Gyro[0] = self.XCurr[3]
    Gyro[1] = self.XCurr[4]
    Gyro[2] = self.XCurr[5]
    return roll, pitch, yaw
    
class PCA9685:

  # Регистры
  __SUBADR1            = 0x02
  __SUBADR2            = 0x03
  __SUBADR3            = 0x04
  __MODE1              = 0x00
  __PRESCALE           = 0xFE
  __LED0_ON_L          = 0x06
  __LED0_ON_H          = 0x07
  __LED0_OFF_L         = 0x08
  __LED0_OFF_H         = 0x09
  __ALLLED_ON_L        = 0xFA
  __ALLLED_ON_H        = 0xFB
  __ALLLED_OFF_L       = 0xFC
  __ALLLED_OFF_H       = 0xFD

  def __init__(self, address=0x40, debug=False):
    self.bus = smbus.SMBus(1)
    self.address = address
    self.debug = debug
    if (self.debug):
      print("Reseting PCA9685")
    self.write(self.__MODE1, 0x00)
	
  def write(self, reg, value):
    "Writes an 8-bit value to the specified register/address"
    self.bus.write_byte_data(self.address, reg, value)
    if (self.debug):
      print("I2C: Write 0x%02X to register 0x%02X" % (value, reg))
	  
  def read(self, reg):
    "Read an unsigned byte from the I2C device"
    result = self.bus.read_byte_data(self.address, reg)
    if (self.debug):
      print("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" % (self.address, result & 0xFF, reg))
    return result
	
  def setPWMFreq(self, freq):
    "Sets the PWM frequency"
    prescaleval = 25000000.0    # 25MHz
    prescaleval /= 4096.0       # 12-bit
    prescaleval /= float(freq)
    prescaleval -= 1.0
    if (self.debug):
      print("Setting PWM frequency to %d Hz" % freq)
      print("Estimated pre-scale: %d" % prescaleval)
    prescale = math.floor(prescaleval + 0.5)
    if (self.debug):
      print("Final pre-scale: %d" % prescale)

    oldmode = self.read(self.__MODE1);
    newmode = (oldmode & 0x7F) | 0x10        # sleep
    self.write(self.__MODE1, newmode)        # go to sleep
    self.write(self.__PRESCALE, int(math.floor(prescale)))
    self.write(self.__MODE1, oldmode)
    time.sleep(0.005)
    self.write(self.__MODE1, oldmode | 0x80)

  def setPWM(self, channel, on, off):
    "Sets a single PWM channel"
    self.write(self.__LED0_ON_L+4*channel, on & 0xFF)
    self.write(self.__LED0_ON_H+4*channel, on >> 8)
    self.write(self.__LED0_OFF_L+4*channel, off & 0xFF)
    self.write(self.__LED0_OFF_H+4*channel, off >> 8)
    if (self.debug):
      print("channel: %d  LED_ON: %d LED_OFF: %d" % (channel,on,off))
	  
  def setServoPulse(self, channel, pulse):
    "Sets the Servo Pulse,The PWM frequency must be 50HZ"
    pulse = pulse*4096/20000        #PWM frequency is 50HZ,the period is 20000us
    self.setPWM(channel, 0, int(pulse))



def AngleToPulse(angle, ServoNum):
  if ServoNum == 90:
    if angle > 45:
      angle = 45
    if angle < -45:
      angle = -45
    hertz = 1350 + angle*round(2000/180)
  if ServoNum == 995:
    if angle > 60:
      angle = 60
    if angle < -60:
      angle = -60
    hertz = 900 + angle*15
  return hertz
  
"""
    Sq = np.array([[-q1, -q2, q3],
                  [q0, -q3, q2],
                  [q3, q0, -q1],
                  [-q2, q1, q0]])
                  
    tmp1 = np.concatenate((np.identify(4), -halfT / 2* Sq), axis = 1)
    tmp2 = np.concatenate((np.zeros((3,4)), np.identify(3)), axis = 1)
    self.A = np.concatenate((tmp1, tmp2), axis=0)
    self.B = np.concatenate((halfT / 2 * Sq, np.zeros((3, 3))), axis = 0)
    self.xHatBar = np.concatenate((self.A, self.xHat) + np.matmul(self.B, np.array(w).transpose())
    self.xHatBar[0:4] = self.normalizeQuat(self.xHatBar[0:4])
    self.xHatPrev = self.xHat
    
    self.yHatBar = self.predictAccelMag()
    self.pbar = np.matmul(np.matmul(self.A, self.p), self.A.transpose())+ self.Q
  
  def update(self, m)
    tmp1 = np.linalg.inv(np.matmul(np.matmul(self.C, self.pBar),self.C.transpose()) + self.R) #Sk-1
    self.K = np.matmul(np.matmul(self.pBar, self.C.transpose()), tmp1)#Kk
    
    magGuass_B = self.getMagVector(m)
    accel_B = self.getAccelVector(a)

    measurement = np.concatenate((accel_B, magGuass_B), axis=0)
    self.xHat = self.xHatBar + np.matmul(self.K, measurement - self.yHatBar)
    self.xHat[0:4] = self.normalizeQuat(self.xHat[0:4])
    self.p = np.matmul(np.identity(7) - np.matmul(self.K, self.C), self.pBar)
  """
if __name__ == '__main__':
  import time
  print("\nSense HAT Test Program ...\n")
  MotionVal=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
  imu=IMU()
  pwm = PCA9685(0x40, debug=False)
  pwm.setPWMFreq(50)
  
  for i in range(4):
    pwm.setServoPulse(i,AngleToPulse(-45, 995))
    time.sleep(1)
    pwm.setServoPulse(i,AngleToPulse(-0, 995))
    time.sleep(1)
    pwm.setServoPulse(i,AngleToPulse(45, 995))
    time.sleep(1)
  
  pwm.setServoPulse(0,AngleToPulse(0, 90))
  pwm.setServoPulse(1,AngleToPulse(0, 90))
  pwm.setServoPulse(2,AngleToPulse(0, 995))
  pwm.setServoPulse(3,AngleToPulse(0, 995))
  
  Omega = 0;
  K_int_w = [0, 0, 0];
  '''
  gyroscopchikx = np.array([])
  pitch_graph = np.array([])
  roll_graph = np.array([])
  yaw_graph = np.array([])
  #Imu_data = open("Imu_data","w++")
  for i in range(250):
  '''
  Kp_w_x = .5
  Ki_w_x = .01
  Kp_w_y = .5
  Ki_w_y = .01
  Kp_w_z = .5
  Ki_w_z = .01
  w = [0, 0, 0]
  int_w_x = 0
  int_w_y = 0
  int_w_z = 0
  
  while True:
    #try:
        imu.QMI8658_Gyro_Accel_Read()
        imu.AK09918_MagRead()
        imu.icm20948CalAvgValue()
        imu.imuAHRSupdate(MotionVal[0] * 0.0175, MotionVal[1] * 0.0175,MotionVal[2] * 0.0175,
                    MotionVal[3],MotionVal[4],MotionVal[5], 
                    MotionVal[6], MotionVal[7], MotionVal[8])
        
        
        pitch = math.asin(-2 * q1 * q3 + 2 * q0* q2)* 57.3
        roll  = math.atan2(2 * q2 * q3 + 2 * q0 * q1, -2 * q1 * q1 - 2 * q2* q2 + 1)* 57.3
        yaw   = math.atan2(-2 * q1 * q2 - 2 * q0 * q3, 2 * q2 * q2 + 2 * q3 * q3 - 1) * 57.3
        
        d_w_x = w[0] - Gyro[0]/180*3.14;
        int_w_x += d_w_x*0.048;
        d_e = Kp_w_x*d_w_x + Ki_w_x*int_w_x;
        pwm.setServoPulse(0,AngleToPulse(d_e, 90))   
        pwm.setServoPulse(1,AngleToPulse(d_e, 90))
        
        d_w_x = w[1] - Gyro[1]/180*3.14;
        int_w_x += d_w_x*0.048;
        d_n = Kp_w_y*d_w_x + Ki_w_y*int_w_y;        
        print(d_n)
        pwm.setServoPulse(2,AngleToPulse(d_n, 995)) 
        
        d_w_x = w[2] - Gyro[2]/180*3.14;
        int_w_x += d_w_x*0.048;
        d_v = Kp_w_z*d_w_x + Ki_w_z*int_w_z;
        pwm.setServoPulse(3,AngleToPulse(d_v, 995)) 
        
    
        '''
        imu.XCurr = [[roll], [pitch], [yaw], [Gyro[0]], [Gyro[1]], [Gyro[2]]]
        imu.Ycur = [[Gyro[0]],[Gyro[1]],[Gyro[2]]]
        roll, pitch, yaw = imu.predict()
        print("\r\n /-------------------------------------------------------------/ \r\n")
        print('\r\n Roll = %.2f , Pitch = %.2f , Yaw = %.2f\r\n'%(roll,pitch,yaw))
        print('\r\nAcceleration:  X = %d , Y = %d , Z = %d\r\n'%(Accel[0]/16384, Accel[1]/16384, Accel[2]/16384))  
        print('\r\nGyroscope:     X = %d , Y = %d , Z = %d\r\n'%(Gyro[0]/64,Gyro[1]/64,Gyro[2]/64))
        print('\r\nMagnetic:      X = %d , Y = %d , Z = %d\r\n'%((Mag[0]),Mag[1],Mag[2]))
        
        gyroscopchikx = np.append(gyroscopchikx, Gyro[0])
        angles = np.append(angles,[[roll],[pitch],[yaw]])
        pitch_graph = np.append(pitch_graph,pitch)
        roll_graph = np.append(roll_graph,roll)
        yaw_graph = np.append(yaw_graph,yaw)
        
        print("QMITemp=%.2f C\r\n"%imu.QMI8658_readTemp())
        time.sleep(0.1)
  #print(*gyroscopchikx)
  
  
  except(KeyboardInterrupt):
  print("\n")
  break 
  x = np.arange(0,250,1)
  sp = plt.subplot(311)
  plt.plot(pitch_graph)
  sp = plt.subplot(312)
  plt.plot(roll_graph)
  sp = plt.subplot(313)
  plt.plot(yaw_graph)
  plt.show()
'''
