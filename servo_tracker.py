from pyfirmata import Arduino, util, SERVO
import time
import keyboard

board = Arduino('COM4')
time.sleep(1)
print("On")

board.digital[6].mode = SERVO
angle = 90
board.digital[6].write(angle)
time.sleep(1)


def rotate(angle):
	board.digital[6].write(angle)
	# time.sleep(0.5)

# turn = [5, 10, 15, 20, 15 , 10, 5, 0]
# for i in turn:
# 	rotate(i)
# 	print(i)
# 	time.sleep(1)

# board.exit()

while True:
	if keyboard.is_pressed("q"):
		break
	if keyboard.is_pressed("a"):
		angle += 1
		if angle > 180:
			angle= 180
		rotate(angle)
		time.sleep(.05)
	if keyboard.is_pressed("s"):
		angle -= 1
		if angle <= 5:
			angle= 5
		rotate(angle)
		time.sleep(.05)


