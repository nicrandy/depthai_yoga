#Use the 'a' key to rotate left, 's' to rotate right, 'q' to exit program
#adjust speed variable to go faster or slower



from pyfirmata import Arduino, util, SERVO
import time
import keyboard

board = Arduino('COM4')
time.sleep(1)
print("On")

board.digital[6].mode = SERVO
angle = 90
speed = .05 # increase to go slower, decrease to go faster
board.digital[6].write(angle)
time.sleep(1)


def rotate(angle):
	board.digital[6].write(angle)


while True:
	if keyboard.is_pressed("q"):
		break
	if keyboard.is_pressed("a"):
		angle += 1
		if angle > 180:
			angle= 180
		rotate(angle)
		time.sleep(speed)
	if keyboard.is_pressed("s"):
		angle -= 1
		if angle <= 5:
			angle= 5
		rotate(angle)
		time.sleep(speed)


