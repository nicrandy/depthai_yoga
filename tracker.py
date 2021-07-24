from pyfirmata import Arduino, util
import time
import keyboard

board = Arduino('COM3')
time.sleep(1)
print("Tracker turned on")

def turn_on():
	return

counter = 1
# speed = 3
steps = 2038
#from BlazeposeRender.py to control the rotation direction and speed
def rotate(clockwise, speed): #direction is bool. True = clockwise
	global counter
	# global speed
	global steps
	while speed > 0:

		if clockwise == True:
			counter = counter + 1
			steps += 1
			if counter > 4:
				counter = 1
		if clockwise != True:
			counter = counter - 1
			steps -= 1
			if counter < 1:
				counter = 4

		if counter == 1:
			board.digital[8].write(0)
			board.digital[9].write(0)
			board.digital[10].write(0)
			board.digital[11].write(1)
		if counter == 2:
			board.digital[8].write(0)
			board.digital[9].write(0)
			board.digital[10].write(1)
			board.digital[11].write(0)
		if counter == 3:
			board.digital[8].write(0)
			board.digital[9].write(1)
			board.digital[10].write(0)
			board.digital[11].write(0)
		if counter == 4:
			board.digital[8].write(1)
			board.digital[9].write(0)
			board.digital[10].write(0)
			board.digital[11].write(0)
		
		if steps >= 2038:
			steps = 0
		if steps < 0:
			steps = 2038



		print("Steps: ", steps)
		speed -= 1

		time.sleep(0.000001)



