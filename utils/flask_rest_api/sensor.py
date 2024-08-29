import time
import RPi.GPIO as GPIO
import Adafruit_DHT
import spidev
import mysql.connector
import requests

from flask import request
from flask import Flask

#sensor = Adafruit_DHT.DHT11

mysql_connection = mysql.connector.connect(
    host='127.0.0.1',
    #host='172.20.70.21',
    user='name',
    password='Biimeal@1234',
    database='ProgAr'
)

mysql_cursor = mysql_connection.cursor()

app = Flask(__name__)

#Moter Drive PIN
A1A = 5
A1B = 6

B1A = 17
B1B = 27

# 습도 임계치(%)
HUM_THRESHOLD=10

#센서를 물에 담갔을때의 토양습도센서 출력 값
HUM_MAX=0

#모터 드라이버 초기 설정
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(A1A, GPIO.OUT)
GPIO.output(A1A, GPIO.LOW)
GPIO.setup(A1B, GPIO.OUT)
GPIO.output(A1B, GPIO.LOW)

GPIO.setup(B1A, GPIO.OUT)
GPIO.output(B1A, GPIO.LOW)
GPIO.setup(B1B, GPIO.OUT)
GPIO.output(B1B, GPIO.LOW)

spi=spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz=500000

#ADC 값을 가져오는 함수
def read_spi_adc(adcChannel):

   adcValue=0

   buff =spi.xfer2([1,(8+adcChannel)<<4,0])

   adcValue = ((buff[1]&3)<<8)+buff[2]

   return adcValue
   
def read_spi_adc2(adcChannel2):

   adcValue2=1

   buff2 =spi.xfer2([1,(8+adcChannel2)<<4,0])

   adcValue2 = ((buff2[1]&3)<<8)+buff2[2]

   return adcValue2
   
# 센서 값을 백분율로 변환하기위한 map 함수

def map(value,min_adc, max_adc, min_hum,max_hum) :

   adc_range=max_adc-min_adc

   hum_range=max_hum-min_hum

   scale_factor=float(adc_range)/float(hum_range)

   return min_hum+((value-min_adc)/scale_factor)
   
def airhum():
	sensor = Adafruit_DHT.DHT11
	pin =19
	humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
	return humidity, temperature

try:

   adcChannel=0
   adcChannel2=7

   while True :
         adcValue=read_spi_adc(adcChannel)
         adcValue2=read_spi_adc2(adcChannel2)
         
         hum=100-int(map(adcValue,HUM_MAX,1023,0,100))
         print("토양수분: " , hum)
         
         hum2=100-int(map(adcValue2,HUM_MAX,1023,0,100))
         print("soil: ", hum2)
         
         #air
         air_hum, air_tem = airhum()
         if air_hum is not None and air_tem is not None:
         	print('Temp = ' , air_tem)
         	print('Hum = ' , air_hum)

         	sql = "INSERT INTO humidity (air, soil,tem, soil2) VALUES (%s, %s, %s, %s)"
         	val = (air_hum, hum, air_tem, hum2)
         	mysql_cursor.execute(sql, val)
         	mysql_connection.commit()
         	
         	requests.post('http://172.20.1.233:5000/update_humidity', json={'air_hum': air_hum, 'hum': hum, 'hum2': hum2, 'air_tem': air_tem})
         	
         else:
         	print('XXXXXXXXXX')
         #가져온 데이터를 %단위로 변환. 습도높을수록 낮은 값을 반환하므로

           #100에서 뺴주어 습도가 높을 수록 백분율이 높아지도록 계산

         #hum=100-int(map(adcValue,HUM_MAX,1023,0,100))
         #print("토양수분: " , hum)

         if hum < HUM_THRESHOLD : # 임계치보다 수분값이 작으면

              GPIO.output(A1A,GPIO.HIGH)  #워터펌프 가동

              GPIO.output(A1B,GPIO.LOW) 

         else :

              GPIO.output(A1A,GPIO.LOW)

              GPIO.output(A1B,GPIO.LOW) 
              
              
         if hum2 < HUM_THRESHOLD : # 임계치보다 수분값이 작으면

              GPIO.output(B1A,GPIO.HIGH)  #워터펌프 가동

              GPIO.output(B1B,GPIO.LOW) 

         else :

              GPIO.output(B1A,GPIO.LOW)

              GPIO.output(B1B,GPIO.LOW) 
              


except KeyboardInterrupt:
	print("x")

finally :

  GPIO.cleanup()

  spi.close()
