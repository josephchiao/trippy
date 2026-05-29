import math
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np


class Motor:

    '''Meant to imitate a physical motor.'''

    def __init__(self, mass = 0.05, quality = 0, timescale = 1000, max_t = 2, max_speed = 2000, max_torque = 100):

        self.torque = 0 # Nm float
        self.speed = 0 # rad/s float
        self.location = 0 # rad [0, 2pi)
                
        self.t = 0 # ticks
        self.timescale = timescale # ticks/s
        self.max_t = max_t * timescale
        
        self.max_torque = max_torque # Nm
        self.max_speed = max_speed # rad/s
        self.m = mass #kg
        self.quality = quality # 0 for perfect motor

        self.log = np.array([[0],[0],[0],[0]]) # Motor run log

    def set_location(self, location):
        self.location = location
        if self.location > 2*math.pi or self.location < 0:
            self.location = self.location % (2*math.pi)

    def set_speed(self, speed):
        if speed > self.max_speed:
            speed = self.max_speed
        elif speed < -self.max_speed:
            speed = -self.max_speed
        self.speed = speed
    
    def set_torque(self, torque):
        if torque > self.max_torque:
            torque = self.max_torque
        elif torque < -self.max_torque:
            torque = -self.max_torque
        self.torque = torque

    def get_status(self):
        return np.array([[self.t/self.timescale], [self.location], [self.speed], [self.torque]])
    
    def update(self):
        self.t += 1
        self.set_speed(self.speed + self.torque/self.m/self.timescale)
        self.set_location(self.location + self.speed)

    def log_data(self):
        self.log = np.hstack((self.log, self.get_status()))

    def run(self):
        while self.t < self.max_t:
            self.update()
            self.log_data()

class MotorDisplay(Motor):
    
    def __init__(self):
        super().__init__()

    def get_position(self, theta):
        return math.cos(theta), math.sin(theta)

    def animate(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        artists = []
        for i in range(self.max_t):
            x,y = self.get_position(self.log[1][i])
            label = ax.text(0.8, 0.9, f'Time: {i/self.timescale}s', transform=ax.transAxes, fontsize=12, fontweight='bold')
            speed = ax.text(0.8, 0.8, f'Speed: {self.log[2][i]}s', transform=ax.transAxes, fontsize=12, fontweight='bold')
            artists.append(ax.plot([0,x],[0,y], c="b") + [label] + [speed])
            if i % 100 == 0:
                print(i)
        ani = animation.ArtistAnimation(fig=fig, artists=artists, interval = 1000/self.timescale)
        plt.show()



animate = MotorDisplay()
animate.set_torque(-0.01)
animate.set_speed(0.1)
animate.run()
print('done')
animate.animate()
