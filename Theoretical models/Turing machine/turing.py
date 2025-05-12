import numpy as np
import pandas as pd
import pygame
import sys
import time

screenwidth = 600
screenheight = 30
speed = 0.0001

tapefile = sys.argv[1]
tapedata = pd.read_excel(tapefile)
tape = tapedata.to_numpy()
tapesize = len(tape)
tapemin = np.min(tape)
tapemax = np.max(tape)

programfile = sys.argv[2]
programdata = pd.read_excel(programfile)
program = programdata.to_numpy()
programsize = len(program)

position = 0
state = 0

def DrawMachine():
    display.fill(white)
    DrawTape()
    DrawReader()

def DrawTape():
    i = 0
    cellsizex = screenwidth / tapesize
    cellsizey = (2/3) * screenheight
    for x in range(0,tapesize,1):
        color = ((tape[x]-tapemin)/(tapemax-tapemin)) *200
        pygame.draw.rect(display, (color,color,color),[cellsizex*x,(1/3)*screenheight,cellsizex,cellsizey])

def DrawReader():
    cellsizex = screenwidth / tapesize
    cellsizey = (1/3)*screenheight
    pygame.draw.rect(display, green,[cellsizex*position,0,cellsizex,cellsizey])

def Compute():

    global tape, program, position, state

    value = tape[position]
    for i in range(programsize):
        if (state == program[i][0]):
            if (value == program[i][1]):
                tape[position] = int(program[i][2])
                newposition = position + int(program[i][3])
                if (newposition >= 0 and newposition < tapesize):
                    position = newposition
                state = int(program[i][4])
                break

pygame.init()
pygame.display.set_caption("Turing machine")
display = pygame.display.set_mode((screenwidth,screenheight))

white,black,red,green,blue = (255,255,255),(0,0,0),(255,0,0),(0,255,0),(0,0,255)

running = True
while running:

    DrawMachine()
    Compute()

    time.sleep(speed)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.display.update()

pygame.quit()
quit()