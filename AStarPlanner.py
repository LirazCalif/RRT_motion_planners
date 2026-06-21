import numpy as np
import heapq



class AStarPlanner(object):
    def __init__(self, bb, start, goal):
        self.bb = bb
        self.start = start
        self.goal = goal

        self.nodes = dict()

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = []

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []

        # define all directions the agent can take - order doesn't matter here
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]

        self.epsilon = 20
        plan = self.a_star(self.start, self.goal)
        return np.array(plan)

    # compute heuristic based on the planning_env
    def compute_heuristic(self, state):
        '''
        Return the heuristic function for the A* algorithm.
        @param state The state (position) of the robot.
        '''
        # TODO: HW3 2.1
        h=self.bb.compute_distance(state, self.goal)
        return h

    def a_star(self, start_loc, goal_loc):
        # TODO: HW3 2.1
        open_heap=[]
        start_loc = tuple(start_loc)
        goal_loc = tuple(goal_loc)
        h=self.compute_heuristic(start_loc)
        heapq.heappush(open_heap, (self.epsilon*h, start_loc))
        parent={}
        g={start_loc:0}
        while open_heap:
            f,current=heapq.heappop(open_heap)
            if current in self.expanded_nodes:
                continue
            self.expanded_nodes.append(current)

            if current==goal_loc:
                path=[current]
                while current in parent:
                    current=parent[current]
                    path.append(current)
                path.reverse()
                print(f"For epsilon={self.epsilon}")
                print(f"The number of expanded nodes is {len(self.expanded_nodes)}")
                print(f"The cost of the path is {g[goal_loc]:.2f}")
                return np.array(path)

            for dx,dy in self.directions:
                nb=(current[0]+dx,current[1]+dy)

                if not self.bb.config_validity_checker(nb):
                    continue

                ten_g=g[current]+ self.bb.compute_distance(current, nb)

                if nb not in g or ten_g < g[nb]:
                    g[nb] = ten_g
                    parent[nb] = current
                    h_nb=self.compute_heuristic(nb)
                    f_nb = ten_g + self.epsilon * h_nb
                    heapq.heappush(open_heap, (f_nb, nb))
        return []




