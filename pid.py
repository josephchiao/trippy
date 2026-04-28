import numpy as np
from matplotlib import pyplot as plt
import math 

class pid_controller:
    def __init__(self, target, location, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.location = location
        self.prev_error = 0
        self.integral = 0

    def update(self):
        
        output = self.proportional() + self.integrate() - self.derivative()
        if self.prev_error * (self.target - self.location) < 0:
            self.integral = 0
            self.prev_error = self.target - self.location
        else:
            self.prev_error = self.target - self.location
            self.integral += self.prev_error
        return output

    def proportional(self):
        return self.kp * (self.target - self.location)
    
    def integrate(self):
        return self.ki * self.integral
    
    def derivative(self):
        return self.kd * (self.prev_error - (self.target - self.location))
    
