import pygame, time
import math 
import random
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

PI = 3.1415
WINDOWWIDTH = 2000
WINDOWHEIGHT = 1000
RED =   (255, 0,  0)
GREEN = (0,  255, 0)
BLUE =  (0,   0, 255)
WHITE   = (255, 255, 255)
BLACK =   (0, 0, 0)
car_speed = 20
LEARNING_RATE = 0.001
DISCOUNT = 0.9
EPISODES = 9999999999
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01
state_size = 8
action_size = 7
batch_size = 32
score = 0
save = 0
sensor_range = 100

pygame.init()
screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))

def environment():   
    pygame.display.set_caption('CAR SIMULATION')
    BasicFont = pygame.font.Font('freesansbold.ttf', 20)  
    pygame.draw.line(screen, BLACK, (5, 5), (1500,5), 80)
    pygame.draw.line(screen, BLACK, (1500, 5), (1500,780), 80)
    pygame.draw.line(screen, BLACK, (1500,780), (250,780), 80)
    pygame.draw.line(screen, BLACK, (250,780), (250,5), 60)
    pygame.draw.line(screen, BLACK, (470, 230), (1230,230), 30)
    pygame.draw.line(screen, BLACK, (1200,230), (1200,570), 60)
    pygame.draw.line(screen, BLACK, (1200,550), (470,550), 30)
    pygame.draw.line(screen, BLACK, (500,550), (500, 230), 60)


clock = pygame.time.Clock()
screen_rect = screen.get_rect()
image_orig = pygame.image.load("car_image.jpeg").convert()
image_orig = pygame.transform.scale(image_orig, (110, 55))
image = image_orig.copy()
agent = image_orig.get_rect(center=screen_rect.center)
agent.left = 400
agent.top = 100
angle = 0

def degtorad(derece):
    return ((derece * 2 * PI) / 360)

def rear_sensor():
	a, b = agent.center
	angle_arka = angle + 180
	if(0 <= angle_arka <= 90):
		b -= int(math.sin(degtorad(abs(angle_arka))) * 55)
		a += int(math.cos(degtorad(abs(angle_arka))) * 55)
	elif(90 <= angle_arka <= 180):
		b -= int(math.sin(degtorad(abs(angle_arka))) * 55)
		a += int(math.cos(degtorad(abs(angle_arka))) * 55)
	elif(270 >= angle_arka >= 180):
		b -= int(math.sin(degtorad(180 - (angle_arka))) * 55)
		a -= int(math.cos(degtorad(180 - (angle_arka))) * 55)
	elif(360 >= angle_arka >= 270):
		b -= int(math.sin(degtorad(180 - (angle_arka))) * 55)
		a -= int(math.cos(degtorad(180 - (angle_arka))) * 55)
	
	x2,y2 = a,b
	for i in range(sensor_range):
		reference = (a,b)
		if (screen.get_at(reference) != (255, 255, 255, 255)):
			intersect = reference 
			break      
		else:
			if(0 <= angle_arka <= 90):
				b -= int(math.sin(degtorad(abs(angle_arka))) * 10)
				a += int(math.cos(degtorad(abs(angle_arka))) * 10)
			elif(90 <= angle_arka <= 180):
				b -= int(math.sin(degtorad(abs(angle_arka))) * 10)
				a += int(math.cos(degtorad(abs(angle_arka))) * 10)
			elif(270 >= angle_arka >= 180):
				b -= int(math.sin(degtorad(180 - (angle_arka))) * 10)
				a -= int(math.cos(degtorad(180 - (angle_arka))) * 10)
			elif(360 >= angle_arka >= 270):
				b -= int(math.sin(degtorad(180 - (angle_arka))) * 10)
				a -= int(math.cos(degtorad(180 - (angle_arka))) * 10)

	x1,y1 = reference
	pygame.draw.line(screen, BLACK, (x2,y2), (x1,y1), 2)  
	mesafe = math.sqrt(math.pow((x1-x2),2) + math.pow((y1 - y2),2))
	if mesafe < 0:
		mesafe = 0
	return round (mesafe, 5)

def rear_right_sensor():
	angle_sagarka = 45 + angle
	a, b = agent.center
	if(27 >= 27 + angle >= -63):
		b += int(math.sin(degtorad((27 + angle))) * 61.7)
		a -= int(math.cos(degtorad((27 + angle))) * 61.7)
	elif(-63 >= 27 + angle >= -153):
		b -= int(math.sin(degtorad(abs(27 + angle))) * 61.7)
		a -= int(math.cos(degtorad(abs(27 + angle))) * 61.7)
	elif(117 <= (27 + angle) <= 207):
		b += int(math.sin(degtorad(abs(27 + angle))) * 61.7)
		a -= int(math.cos(degtorad(abs(27 + angle))) * 61.7)
	elif(27 <= (27 + angle) <= 117):
		b += int(math.sin(degtorad(180 - abs(27 + angle))) * 61.7)
		a += int(math.cos(degtorad(180 - abs(27 + angle))) * 61.7)
	
	x2,y2 = a,b
	for i in range(sensor_range):
		reference = (a,b)
		if (screen.get_at(reference) != (255, 255, 255, 255)):
			intersect = reference 
			break      
		else:
			if(27 >= angle_sagarka >= -63):
				b += int(math.sin(degtorad((angle_sagarka))) * 10)
				a -= int(math.cos(degtorad((angle_sagarka))) * 10)
			elif(-63 >= angle_sagarka >= -153):
				b -= int(math.sin(degtorad(abs(angle_sagarka))) * 10)
				a -= int(math.cos(degtorad(abs(angle_sagarka))) * 10)
			elif(117 <= angle_sagarka <= 230):
				b += int(math.sin(degtorad(abs(angle_sagarka))) * 10)
				a -= int(math.cos(degtorad(abs(angle_sagarka))) * 10)
			elif(27 <= (angle_sagarka) <= 117):
				b += int(math.sin(degtorad(180 - abs(angle_sagarka))) * 10)
				a += int(math.cos(degtorad(180 - abs(angle_sagarka))) * 10)
	x1,y1 = reference
	pygame.draw.line(screen, BLACK, (x2,y2), (x1,y1), 2) 
	mesafe = math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2))
	if(mesafe < 0):
		mesafe = 0
	return round (mesafe, 5)

def rear_left_sensor():
	angle_solarka = 45 - angle
	a, b = agent.center
	if(0 <= (27 - angle) <= 117):
		b -= int(math.sin(degtorad(abs(27 - angle))) * 61.7)
		a -= int(math.cos(degtorad(abs(27 - angle))) * 61.7)
	elif(117 <= (27 - angle) <= 207):
		b -= int(math.sin(degtorad(abs(27 - angle))) * 61.7)
		a -= int(math.cos(degtorad(abs(27 - angle))) * 61.7)
	elif(-63 >= 27 - angle >= -153):
		b += int(math.sin(degtorad(abs(27 - angle))) * 61.7)
		a -= int(math.cos(degtorad(abs(27 - angle))) * 61.7)
	elif(27 >= (27 - angle) >= -63):
		b += int(math.sin(degtorad(180 - abs(27 - angle))) * 61.7)
		a += int(math.cos(degtorad(180 - abs(27 - angle))) * 61.7)
	
	x2,y2 = a,b
	for i in range(sensor_range):
		reference = (a,b)
		if (screen.get_at(reference) != (255, 255, 255, 255)):
			intersect = reference 
			break      
		else:
			if(0 <= angle_solarka <= 117):
				b -= int(math.sin(degtorad(abs(angle_solarka))) * 10)
				a -= int(math.cos(degtorad(abs(angle_solarka))) * 10)
			elif(117 <= angle_solarka <= 230):
				b -= int(math.sin(degtorad(abs(angle_solarka))) * 10)
				a -= int(math.cos(degtorad(abs(angle_solarka))) * 10)
			elif(-63 >= angle_solarka >= -153):
				b += int(math.sin(degtorad(abs(angle_solarka))) * 10)
				a -= int(math.cos(degtorad(abs(angle_solarka))) * 10)
			elif(27 >= angle_solarka >= -63):
				b += int(math.sin(degtorad(180 - abs(angle_solarka))) * 10)
				a += int(math.cos(degtorad(180 - abs(angle_solarka))) * 10)
	x1,y1 = reference
	pygame.draw.line(screen, BLACK, (x2,y2), (x1,y1), 2) 
	mesafe = math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2))
	if(mesafe < 0):
		mesafe = 0
	return round (mesafe, 5)

def left_sensor():
	angle_sol = angle + 90
	a, b =agent.center
	if(0 < (angle + 90) < 90):
		b -= int(math.sin(degtorad(abs(angle + 90))) * 27.5)
		a += int(math.cos(degtorad(abs(angle + 90))) * 27.5)
	elif(90 <= (angle + 90) <= 180):
		b -= int(math.sin(degtorad(abs(angle + 90))) * 27.5)
		a += int(math.cos(degtorad(abs(angle + 90))) * 27.5)
	elif(0 >= (angle + 90) > -90):
		b += int(math.sin(degtorad(abs(angle + 90))) * 27.5)
		a += int(math.cos(degtorad(abs(angle + 90))) * 27.5)
	elif(-90 >= (angle + 90) > -180):
		b += int(math.sin(degtorad(180 - abs(angle + 90))) * 27.5)
		a -= int(math.cos(degtorad(180 - abs(angle + 90))) * 27.5)

	x2,y2 = a,b
	for i in range(sensor_range):
		reference = (a,b)
		if (screen.get_at(reference) != (255, 255, 255, 255)):
			intersect = reference 
			break      
		else:
			if(0 <= angle_sol <= 90):
				b -= int(math.sin(degtorad(abs(angle_sol))) * 10)
				a += int(math.cos(degtorad(abs(angle_sol))) * 10)
			elif(180 <= angle_sol <= 270):
				b -= int(math.sin(degtorad(abs(angle_sol))) * 10)
				a += int(math.cos(degtorad(abs(angle_sol))) * 10)
			elif(0 >= angle_sol >= -90):
				b += int(math.sin(degtorad(abs(angle_sol))) * 10)
				a += int(math.cos(degtorad(abs(angle_sol))) * 10)
			elif(180 >= angle_sol >= 90):
				b += int(math.sin(degtorad(180 + abs(angle_sol))) * 10)
				a -= int(math.cos(degtorad(180 + abs(angle_sol))) * 10)
	x1,y1 = reference
	pygame.draw.line(screen, BLACK, (x2,y2), (x1,y1), 2) 
	mesafe = math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2))
	if(mesafe < 0):
		mesafe = 0
	return round (mesafe, 5)

def right_sensor():
	angle_sag = angle - 90
	a, b =agent.center
	if(0 < (angle - 90) < 90):
		b -= int(math.sin(degtorad(abs(angle - 90))) * 27.5)
		a += int(math.cos(degtorad(abs(angle - 90))) * 27.5)
	elif(90 <= (angle - 90) <= 180):
		b -= int(math.sin(degtorad(abs(angle - 90))) * 27.5)
		a += int(math.cos(degtorad(abs(angle - 90))) * 27.5)
	elif(0 >= (angle - 90) > -90):
		b += int(math.sin(degtorad(abs(angle - 90))) * 27.5)
		a += int(math.cos(degtorad(abs(angle - 90))) * 27.5)
	elif(-90 >= (angle - 90) > -180):
		b += int(math.sin(degtorad(180 - abs(angle - 90))) * 27.5)
		a -= int(math.cos(degtorad(180 - abs(angle - 90))) * 27.5)

	x2,y2 = a,b
	for i in range(sensor_range):
		reference = (a,b)
		if (screen.get_at(reference) != (255, 255, 255, 255)):
			intersect = reference 
			break      
		else:
			if(0 <= angle_sag <= 90):
				b -= int(math.sin(degtorad(abs(angle_sag))) * 10)
				a += int(math.cos(degtorad(abs(angle_sag))) * 10)
			elif(-270 <= angle_sag <= -180):
				b -= int(math.sin(degtorad(-abs(angle_sag))) * 10)
				a += int(math.cos(degtorad(-abs(angle_sag))) * 10)
			elif(0 >= angle_sag >= -90):
				b += int(math.sin(degtorad(abs(angle_sag))) * 10)
				a += int(math.cos(degtorad(abs(angle_sag))) * 10)
			elif(-90 >= angle_sag >= -180):
				b += int(math.sin(degtorad(180 - abs(angle_sag))) * 10)
				a -= int(math.cos(degtorad(180 - abs(angle_sag))) * 10)
	x1,y1 = reference
	pygame.draw.line(screen, BLACK, (x2,y2), (x1,y1), 2) 
	mesafe = math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2))
	if(mesafe < 0):
		mesafe = 0
	return round (mesafe, 5)

def front_right_sensor():
	angle_sagon = angle - 45
	a, b = agent.center
	if(0 < (angle - 27) < 90 ):
		b -= int(math.sin(degtorad(abs(angle - 27))) * 61.7)
		a += int(math.cos(degtorad(abs(angle - 27))) * 61.7)
	elif(90 <= (angle - 27) <= 180 ):
		b -= int(math.sin(degtorad(abs(angle - 27))) * 61.7)
		a += int(math.cos(degtorad(abs(angle - 27))) * 61.7)
	elif(0 >= (angle - 27) > -90):
		b += int(math.sin(degtorad(abs(angle - 27))) * 61.7)
		a += int(math.cos(degtorad(abs(angle - 27))) * 61.7)
	elif(-90 >= (angle - 27) >= -207):
		b += int(math.sin(degtorad(180 - abs(angle - 27))) * 61.7)
		a -= int(math.cos(degtorad(180 - abs(angle - 27))) * 61.7)
	
	x2,y2 = a,b
	for i in range(sensor_range):
		reference = (a,b)
		if (screen.get_at(reference) != (255, 255, 255, 255)):
			intersect = reference 
			break      
		else:
			if(0 <= angle_sagon <= 90):
				b -= int(math.sin(degtorad(abs(angle_sagon))) * 10)
				a += int(math.cos(degtorad(abs(angle_sagon))) * 10)
			elif(90 <= angle_sagon <= 180 ):
				b -= int(math.sin(degtorad(abs(angle_sagon))) * 10)
				a += int(math.cos(degtorad(abs(angle_sagon))) * 10)
			elif(0 >= angle_sagon >= -90 or -215 >= angle_sagon >= -225):
				b += int(math.sin(degtorad(abs(angle_sagon))) * 10)
				a += int(math.cos(degtorad(abs(angle_sagon))) * 10)
			elif(-90 >= angle_sagon >= -207):
				b += int(math.sin(degtorad(180 - abs(angle_sagon))) * 10)
				a -= int(math.cos(degtorad(180 - abs(angle_sagon))) * 10)
	x1,y1 = reference
	pygame.draw.line(screen, BLACK, (x2,y2), (x1,y1), 2) 
	mesafe = math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2))
	if(mesafe < 0):
		mesafe = 0
	return round (mesafe, 5)
	

def front_sensor():
	a, b = agent.center
	if(0 <= angle <= 90):
		b -= int(math.sin(degtorad(abs(angle))) * 55)
		a += int(math.cos(degtorad(abs(angle))) * 55)
	elif(90 <= angle <= 180):
		b -= int(math.sin(degtorad(abs(angle))) * 55)
		a += int(math.cos(degtorad(abs(angle))) * 55)
	elif(0 >= angle >= -90):
		b += int(math.sin(degtorad(abs(angle))) * 55)
		a += int(math.cos(degtorad(abs(angle))) * 55)
	elif(-90 >= angle >= -180):
		b += int(math.sin(degtorad(180 - abs(angle))) * 55)
		a -= int(math.cos(degtorad(180 - abs(angle))) * 55)
	
	x2,y2 = a,b
	for i in range(sensor_range):
		reference = (a,b)
		if (screen.get_at(reference) != (255, 255, 255, 255)):
			intersect = reference 
			break      
		else:
			if(0 <= angle <= 90):
				b -= int(math.sin(degtorad(abs(angle))) * 10)
				a += int(math.cos(degtorad(abs(angle))) * 10)
			elif(90 <= angle <= 180):
				b -= int(math.sin(degtorad(abs(angle))) * 10)
				a += int(math.cos(degtorad(abs(angle))) * 10)
			elif(0 >= angle >= -90):
				b += int(math.sin(degtorad(abs(angle))) * 10)
				a += int(math.cos(degtorad(abs(angle))) * 10)
			elif(-90 >= angle >= -180):
				b += int(math.sin(degtorad(180 - abs(angle))) * 10)
				a -= int(math.cos(degtorad(180 - abs(angle))) * 10)

	x1,y1 = reference
	pygame.draw.line(screen, BLACK, (x2,y2), (x1,y1), 2)  
	mesafe = math.sqrt(math.pow((x1-x2),2) + math.pow((y1 - y2),2))
	if mesafe < 0:
		mesafe = 0
	return round (mesafe, 5)

def front_left_sensor():
	angle_solon = angle + 45
	a, b = agent.center
	if(0 < (angle + 27) < 90):
		b -= int(math.sin(degtorad(abs(angle + 27))) * 61.7)
		a += int(math.cos(degtorad(abs(angle + 27))) * 61.7)
	elif(90 <= (angle + 27) <= 207):
		b -= int(math.sin(degtorad(abs(angle + 27))) * 61.7)
		a += int(math.cos(degtorad(abs(angle + 27))) * 61.7)
	elif(0 >= (angle + 27) > -90):
		b += int(math.sin(degtorad(abs(angle + 27))) * 61.7)
		a += int(math.cos(degtorad(abs(angle + 27))) * 61.7)
	elif(-90 >= (angle + 27) > -180):
		b += int(math.sin(degtorad(180 - abs(angle + 27))) * 61.7)
		a -= int(math.cos(degtorad(180 - abs(angle + 27))) * 61.7)
	
	x2,y2 = a,b
	for i in range(sensor_range):
		reference = (a,b)
		if (screen.get_at(reference) != (255, 255, 255, 255)):
			intersect = reference 
			break      
		else:
			if(0 <= angle_solon <= 90):
				b -= int(math.sin(degtorad(abs(angle_solon))) * 10)
				a += int(math.cos(degtorad(abs(angle_solon))) * 10)
			elif(90 <= angle_solon <= 225):
				b -= int(math.sin(degtorad(abs(angle_solon))) * 10)
				a += int(math.cos(degtorad(abs(angle_solon))) * 10)
			elif(0 >= angle_solon >= -90):
				b += int(math.sin(degtorad(abs(angle_solon))) * 10)
				a += int(math.cos(degtorad(abs(angle_solon))) * 10)
			elif(-90 >= angle_solon >= -180):
				b += int(math.sin(degtorad(180 - abs(angle_solon))) * 10)
				a -= int(math.cos(degtorad(180 - abs(angle_solon))) * 10)
	x1,y1 = reference
	pygame.draw.line(screen, BLACK, (x2,y2), (x1,y1), 2) 
	mesafe = math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2))
	if(mesafe < 0):
		mesafe = 0
	return round (mesafe, 5)


memory = deque(maxlen=500)
##BUILD MODEL
model = Sequential()
model.add(Dense(128, input_dim = state_size, activation = 'linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(128, activation='linear'))
model.add(Dense(action_size, activation='linear'))
model.compile(loss = 'mse', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

done=False

for time in range(0,EPISODES):
    image = pygame.transform.rotate(image_orig, angle)
    agent = image.get_rect(center=agent.center)
    screen.blit(image, agent)
    pygame.display.update()
    screen.fill(WHITE)
    environment()

    #Current state
    left_sensor_current = left_sensor()
    right_sensor_current  = right_sensor()
    front_right_sensor_current  = front_right_sensor()    
    front_sensor_current  = front_sensor()
    front_left_sensor_current  = front_left_sensor()
    rear_sensor_current  = rear_sensor()
    rear_left_sensor_current  = rear_left_sensor()
    rear_right_sensor_current  = rear_right_sensor()
    current_state = [left_sensor_current / 900 , front_left_sensor_current / 900 , front_sensor_current / 900, front_right_sensor_current / 900, right_sensor_current / 900, rear_sensor_current / 900, rear_left_sensor_current / 900, rear_right_sensor_current / 900]   
    current_state = np.reshape(current_state, [1, state_size])   
    
    ##SELECT ACTION
    action_values = model.predict(current_state)
    if(np.random.rand() <= epsilon):
        action = random.randrange(action_size)
    else:      
        action = np.argmax(action_values[0])
             
    if (action == 1):     #FORWARD LEFT - ILERI SOL
        if(angle >= 180):
            angle = -170
        else:
            angle += 10
        if(0 <= angle <= 90):
            agent.top -= math.sin(degtorad(abs(angle))) * car_speed
            agent.right += math.cos(degtorad(abs(angle))) * car_speed
        elif(90 <= angle <= 180):
            agent.top -= math.sin(degtorad(abs(angle))) * car_speed
            agent.right += math.cos(degtorad(abs(angle))) * car_speed
        elif(0 >= angle >= -90):
            agent.top += math.sin(degtorad(abs(angle))) * car_speed
            agent.right += math.cos(degtorad(abs(angle))) * car_speed
        elif(-90 >= angle >= -180):
            agent.top += math.sin(degtorad(180 - abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(180 - abs(angle))) * car_speed
    elif (action == 2):      #FORWARD RIGHT - ILERI SAG
        if(angle <= -180):
            angle = 170
        else:
            angle += -10
        if(0 <= angle <= 90):
            agent.top -= math.sin(degtorad(abs(angle))) * car_speed
            agent.right += math.cos(degtorad(abs(angle))) * car_speed
        elif(90 <= angle <= 180):
            agent.top -= math.sin(degtorad(abs(angle))) * car_speed
            agent.right += math.cos(degtorad(abs(angle))) * car_speed
        elif(0 >= angle >= -90):
            agent.top += math.sin(degtorad(abs(angle))) * car_speed
            agent.right += math.cos(degtorad(abs(angle))) * car_speed
        elif(-90 >= angle >= -180):
            agent.top += math.sin(degtorad(180 - abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(180 - abs(angle))) * car_speed
    elif (action == 0):     #FORWARD - ILERI
        if(0 <= angle <= 90):
            agent.top -= math.sin(degtorad(abs(angle))) * car_speed
            agent.right += math.cos(degtorad(abs(angle))) * car_speed
        elif(90 <= angle < 180):
            agent.top -= math.sin(degtorad(abs(angle))) * car_speed
            agent.right += math.cos(degtorad(abs(angle))) * car_speed
        elif(0 >= angle >= -90):
            agent.top += math.sin(degtorad(abs(angle))) * car_speed
            agent.right += math.cos(degtorad(abs(angle))) * car_speed
        elif(-90 >= angle >= -180 or angle == 180):
            agent.top += math.sin(degtorad(180 - abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(180 - abs(angle))) * car_speed
    elif (action == 6):       #BACKWARD LEFT - GERI SOL
        if(angle <= -180):
            angle = 170
        else:
            angle += -10
        if(0 <= angle <= 90):
            agent.top += math.sin(degtorad(abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(abs(angle))) * car_speed
        elif(90 <= angle <= 180):
            agent.top += math.sin(degtorad(abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(abs(angle))) * car_speed
        elif(0 >= angle >= -90):
            agent.top -= math.sin(degtorad(abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(abs(angle))) * car_speed
        elif(-90 >= angle >= -180):
            agent.top -= math.sin(degtorad(180 - abs(angle))) * car_speed
            agent.right += math.cos(degtorad(180 - abs(angle))) * car_speed
    elif (action == 4):       #BACKWARD RIGHT - GERI SAG
        if(angle >= 180):
            angle = -170
        else:
            angle += 10
        if(0 <= angle <= 90):
            agent.top += math.sin(degtorad(abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(abs(angle))) * car_speed
        elif(90 <= angle <= 180):
            agent.top += math.sin(degtorad(abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(abs(angle))) * car_speed
        elif(0 >= angle >= -90):
            agent.top -= math.sin(degtorad(abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(abs(angle))) * car_speed
        elif(-90 >= angle >= -180):
            agent.top -= math.sin(degtorad(180 - abs(angle))) * car_speed
            agent.right += math.cos(degtorad(180 - abs(angle))) * car_speed
    elif (action == 5):        #BACKWARD
        if(0 <= angle <= 90):
            agent.top += math.sin(degtorad(abs(angle))) * car_speed
            if ( angle != 0):
            	agent.right -= math.cos(degtorad(abs(angle))) * car_speed
        elif(90 <= angle <= 180):
            agent.top += math.sin(degtorad(abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(abs(angle))) * car_speed
        elif(0 >= angle > -90):
            agent.top -= math.sin(degtorad(abs(angle))) * car_speed
            agent.right -= math.cos(degtorad(abs(angle))) * car_speed
        elif(-90 >= angle >= -180):
            agent.top -= math.sin(degtorad(180 - abs(angle))) * car_speed
            agent.right += math.cos(degtorad(180 - abs(angle))) * car_speed
    elif (action == 3):         #DO NOT MOVE - SABIT KAL
        pass

    #next state
    left_sensor_new = left_sensor()
    right_sensor_new  = right_sensor()
    front_right_sensor_new  = front_right_sensor()    
    front_sensor_new  = front_sensor()
    front_left_sensor_new  = front_left_sensor()
    rear_sensor_new  = rear_sensor()
    rear_left_sensor_new  = rear_left_sensor()
    rear_right_sensor_new  = rear_right_sensor()
    next_state = [left_sensor_new/ 900 , front_left_sensor_new /900 , front_sensor_new /900 , front_right_sensor_new /900, right_sensor_new /900, rear_sensor_new /900, rear_left_sensor_new / 900 , rear_right_sensor_new / 900 ]   
    next_state = np.reshape(next_state, [1, state_size]) 

    
    #saving
    score+=1 
    if(score == 200):
        model.save("model_200")

    if(score == 400):
    	model.save("model_400")
    	egitim = 1

    if(score == 600):
    	model.save("model_600")
    	egitim = 1

    #reward mechanism
    if(front_left_sensor_new < (1/900) or front_right_sensor_new < (1/900) or front_sensor_new < (1/900) or rear_left_sensor_new < (1/900) or rear_right_sensor_new < (1/900)):
        done = True
        agent.left = 500
        agent.top = 100
        angle = 0
        reward = -5  #collision reward
        score = 0
        egitim = 1
    elif(action == 3 or action == 4 or action == 5 or action == 6):
    	reward = -1
    else:
    	reward = 1

    ##REMEMBER
    memory.append((current_state, action, reward, next_state, done))

    if done:
        done = False   
    
    if(epsilon > epsilon_min):
        epsilon *= epsilon_decay 
   

    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for current_state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + DISCOUNT * np.amax(model.predict(next_state)[0]))
            target_f = model.predict(current_state)
            target_f[0][action] = target
            model.fit(current_state, target_f, epochs=1, verbose = 0)

