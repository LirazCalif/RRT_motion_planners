import numpy as np
from RRTTree import RRTTree
import time


class RRTInspectionPlanner(object):

    def __init__(self, bb, start, ext_mode, goal_prob, coverage):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb, task="ip")
        self.start = start

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage

        self.validity_cache = {}
        self.dim = len(start)
        self.stuck=0
        self.perv_max_id=0

        self.unseen_nodes_cache = []
        self.cache_update_counter = 0
        self.cache_update_frequency = 50

        self.validity_cache = {}
        self.max_validity_cache_size = 3000
        # set step size - remove for students
        self.step_size = min(self.bb.env.xlimit[-1] / 50, self.bb.env.ylimit[-1] / 200)
        self.unique_config=None
        self.total_coverage_with_unique=0.0




    def cached_validity_check(self, config):
        # Convert config to immutable key
        key = np.round(config, decimals=3).tobytes()
        # Fast cache hit
        if key in self.validity_cache:
            return self.validity_cache[key]

        # Actual collision checking
        is_valid = self.bb.config_validity_checker(config)

        # Store result (bounded cache)
        if len(self.validity_cache) < self.max_validity_cache_size:
            self.validity_cache[key] = is_valid

        return is_valid






    def choose_next_goal(self):
        max_v = self.tree.vertices[self.tree.max_coverage_id]
        max_config = max_v.config
        max_ee = max_v.ee_pos
        step_size_vec=[0.25,0.5,1,5]


        # Identify unseen points
        all_points = self.bb.env.inspection_points
        seen_points = set(map(tuple, max_v.inspected_points))
        unseen = [p for p in all_points if tuple(p) not in seen_points]

        if not unseen:
            return self.bb.sample_random_config(0, None)

        unseen_np = np.array(unseen)
        dists = np.linalg.norm(unseen_np - max_ee, axis=1)

        if self.stuck>300 and self.tree.max_coverage>=50:
                target_point = unseen_np[np.argmax(dists)]
                step_size_vec = [0.5, 1,4]
        else:
            # Find the nearest unseen point to the current EE
            target_point = unseen_np[np.argmin(dists)]

        #  Step towards that point (don't go all the way, just move closer)
        direction = target_point - max_ee
        unit_dir = direction / np.linalg.norm(direction)

        # Try a few steps along that vector
        for step_size in step_size_vec:
            goal_ee = max_ee + unit_dir * step_size
            new_config = self.bb.inverse_kinematics(goal_ee, initial_guess=max_config)

            if new_config is not None and self.cached_validity_check(new_config):
                return new_config

        return self.bb.sample_random_config(0, None)  # Random fallback





    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        # TODO: HW3 2.3.3
        t1 = time.time()
        inspec_points=self.bb.get_inspected_points(self.start)
        self.tree.add_vertex(self.start, inspec_points)
        i=0
        while True:
            max_id = self.tree.max_coverage_id
            if self.perv_max_id==max_id:
                self.stuck+=1
            else:
                self.stuck=0
            self.perv_max_id=max_id
            if max_id == 0:
                # first iteration, sample completely random
                rand_config = self.bb.sample_random_config(0, None)
            elif self.stuck>150 and self.tree.max_coverage<50:
                rand_config = self.bb.sample_random_config(0, None)
            else:
                # extend from the current max coverage vertex
                goal = self.choose_next_goal()
                rand_config = self.bb.sample_random_config(self.goal_prob, goal)

            near_id, near_config = self.tree.get_nearest_config(rand_config)
            new_config = self.extend(near_config, rand_config)
            if new_config is None:
                continue

            if self.bb.edge_validity_checker(near_config, new_config):

                parent_points = self.tree.vertices[near_id].inspected_points
                new_points = self.bb.get_inspected_points(new_config)
                inspec_points = self.bb.compute_union_of_points(parent_points, new_points)

                new_id = self.tree.add_vertex(new_config,inspec_points)

                edge_cost = self.bb.compute_distance(near_config, new_config)
                self.tree.add_edge(near_id, new_id, edge_cost)
                if  i%1000==0:
                    print("the converage:",self.tree.max_coverage)
                    print("time:",time.time()-t1)
                if self.tree.max_coverage >= self.coverage:
                    break
                i+=1

        plan = []
        if self.tree.max_coverage<self.coverage :
            print("No plan was found")
            return None

        current = self.tree.max_coverage_id
        while current != self.tree.get_root_id():
            plan.append(self.tree.vertices[current].config)
            current = self.tree.edges[current]

        plan.append(self.start)
        plan.reverse()

        t2 = time.time()
        self.t = t2 - t1
        print(f"The time it takes to compute plan is {self.t:.3f} seconds")
        self.path_cost = self.compute_cost(plan)
        print(f"The path cost is {self.path_cost:.2f}")
        print(f"the coverage is {self.tree.max_coverage:.2f}")
        print(f"len of found points:{len(inspec_points)}")
        return np.array(plan)


    def sample_random_inspection_point(self):
        points = self.bb.env.inspection_points
        idx = np.random.randint(len(points))
        return points[idx]

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: HW3 2.3.1
        cost = 0
        for idx in range(len(plan) - 1):
            cost += self.bb.compute_distance(plan[idx], plan[idx + 1])
        return cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # TODO: HW3 2.3.1
        if np.allclose(rand_config, near_config):
            return None

        if self.ext_mode == "E1":
            new_config = rand_config
        else:
            eta = 0.4
            vec = rand_config - near_config
            dist = self.bb.compute_distance(near_config, rand_config)
            if dist <= eta:
                new_config = rand_config
            else:
                direction = vec / dist
                new_config = near_config + direction * eta

        if self.cached_validity_check(new_config):
            return new_config

        return None
