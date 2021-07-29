from pyfirmata import Arduino, util, SERVO
import time

port = 'COM3'
pin = 9
board = Arduino(port)

board.digital[pin].mode = SERVO


# board = Arduino('COM3')
# time.sleep(1)
# print("Tracker turned on")



# #from BlazeposeRender.py to control the rotation direction and speed
# def rotate(angle): 
# 	board.digital[9].write(angle)


# test = [0,10,20,30,20,10,0]

# for i in test:
# 	rotate(i)
# 	print("Rotate:", i)
# 	# time.sleep(1)
	

