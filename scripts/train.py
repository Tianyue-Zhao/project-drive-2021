import numpy as np

class state:
    def __init__(self):
        #initialize state with empty variables
        self.cur_points = np.zeros((2,1))
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.velocity = 0.0
        self.angular_vel = 0.0