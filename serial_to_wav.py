import serial
import glob
import numpy as np
from scipy.io import wavfile


def read_data():
    devices = glob.glob("/dev/tty.usbmodem*")
    assert len(devices) == 1
    serial_path = devices[0]
    ser = serial.Serial(serial_path, baudrate=115200)
    output = []
    amp = 1
    try:
        while True:
            line = ser.readline()[:-3]
            print(line)
            data = [int(x, 16) * amp for x in line.decode("ascii").split(' ')]
            output.extend(data)
    except KeyboardInterrupt:
        wavfile.write("out.wav", 16000, np.array(output, dtype=np.int16))

if __name__ == "__main__":
    read_data()
