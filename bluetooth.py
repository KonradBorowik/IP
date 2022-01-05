import serial
import time

print("Start")
port = "COM6.HC-05"
bluetooth = serial.Serial(port, 9600)
print("Connected")
bluetooth.flushInput()

for i in range(5):
    print("Ping")
    bluetooth.write(b"BOOP " + str.encode(str(i)))
    input_data = bluetooth.readline()
    print(input_data.decode())
    time.sleep(0.1)

bluetooth.close()
print("Done")