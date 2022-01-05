import serial
import time
import keyboard

print("Start")
port = "COM6.HC-05"
bluetooth = serial.Serial(port, 9600)
print("Connected")
bluetooth.flushInput()


def go_forward():
    bluetooth.write(b"2")


def go_right():
    bluetooth.write(b"3")
    time.sleep(0.5)


def go_left():
    bluetooth.write(b"4")
    time.sleep(0.5)


while not keyboard.read_key() == "q":

    if keyboard.read_key() == "w":
        go_forward()

    if keyboard.read_key() == "d":
        go_right()

    if keyboard.read_key() == "a":
        go_left()

    input_data = bluetooth.readline()
    print(input_data.decode())


# for i in range(5):
#     print("Ping")
#     bluetooth.write(b"BOOP " + str.encode(str(i)))
#
#     # if i == 2:
#     #     go_forward()
#     # if i == 3:
#     #     go_left()
#     # if i == 4:
#     #     go_right()
#
#     input_data = bluetooth.readline()
#     print(input_data.decode())
#     time.sleep(1)

bluetooth.close()
print("Done")