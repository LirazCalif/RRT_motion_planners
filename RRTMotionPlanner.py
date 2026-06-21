import numpy as np
from RRTTree import RRTTree
import time


class RRTMotionPlanner(object):

    def __init__(self, bb, ext_mode, goal_prob, start, goal):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob



    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        # TODO: HW3 2.2.3
        goal_id=None
        t1=time.time()
        self.tree.add_vertex(self.start)
        while True:
            rand_config=self.bb.sample_random_config(self.goal_prob,self.goal)
            near_id,near_config=self.tree.get_nearest_config(rand_config)
            new_config=self.extend(near_config, rand_config)
            if new_config is None:
                continue
            if self.bb.edge_validity_checker(near_config, new_config):
                new_id=self.tree.add_vertex(new_config)
                edge_cost=self.bb.compute_distance(near_config, new_config)
                self.tree.add_edge(near_id, new_id,edge_cost)
                if  np.array_equal(new_config, self.goal):
                    goal_id=new_id
                    break
        plan = []
        if goal_id is None:
            print("No plan was found")
            return None
        current = goal_id
        while current != self.tree.get_root_id():
            plan.append(self.tree.vertices[current].config)
            current = self.tree.edges[current]

        plan.append(self.start)
        plan.reverse()
        t2=time.time()
        self.t=t2-t1
        print(f"The time it takes to compute plan is {self.t:.3f} seconds")
        self.path_cost=self.compute_cost(plan)
        print(f"The path cost is {self.path_cost:.2f}")
        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: HW3 2.2.2
        cost=0
        for idx in range(len(plan)-1):
            cost+=self.bb.compute_distance(plan[idx], plan[idx+1])
        return cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # TODO: HW3 2.2.1
        if np.allclose(rand_config, near_config):
            return None

        if self.ext_mode == "E1":
            new_config=rand_config
        else:
            eta=0.4
            vec=rand_config-near_config
            dist=self.bb.compute_distance(near_config,rand_config)
            if dist<=eta:
                new_config = rand_config
            else:
                direction = vec / dist
                new_config=near_config+direction*eta

        if self.bb.config_validity_checker(new_config):
            return new_config

        return None




