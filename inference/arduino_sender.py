# inference/arduino_sender.py
import serial
import time

SERIAL_PORT = '/dev/ttyACM0'  # on Windows use 'COM3' etc.
BAUD = 9600

try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    time.sleep(2)
except Exception as e:
    ser = None
    print('Arduino serial not available:', e)

def send_risk_to_arduino(risk_str):
    if ser is None:
        return
    code = '0'  # low
    if risk_str == 'MEDIUM':
        code = '1'
    elif risk_str == 'HIGH':
        code = '2'
    try:
        ser.write(code.encode())
    except Exception as e:
        print('Serial write failed', e)
