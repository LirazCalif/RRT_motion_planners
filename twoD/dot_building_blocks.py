import itertools
import numpy as np
from shapely.geometry import Point, LineString

class DotBuildingBlocks2D(object):

    def __init__(self, env):
        self.env = env
        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        # TODO: HW3 2.1
        dis = np.linalg.norm(np.array(next_config) - np.array(prev_config))
        return dis

    def sample_random_config(self, goal_prob, goal):
        # TODO: HW3 2.1
        # generating a random number [0,1), for probability of goal_prob - return the goal
        if np.random.rand() < goal_prob:
            return np.array(goal)

        while True:
            x=np.random.uniform(low=-self.env.xlimit[0], high=self.env.xlimit[1])
            y=np.random.uniform(low=-self.env.ylimit[0], high=self.env.ylimit[1])
            sample=np.array([x,y])
            if self.config_validity_checker(sample):
                return sample

    def config_validity_checker(self, state):
        return self.env.config_validity_checker(state)

    def edge_validity_checker(self, state1, state2):
        return self.env.edge_validity_checker(state1, state2)


