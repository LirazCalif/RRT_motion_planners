import numpy as np
from RRTTree import RRTTree
import time


class RRTStarPlanner(object):

    def __init__(
        self,
        bb,
        ext_mode,
        max_step_size,
        start,
        goal,
        max_itr=None,
        stop_on_goal=None,
        k=None,
        goal_prob=0.01,
    ):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        self.max_itr = max_itr
        self.stop_on_goal = stop_on_goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.k = k

        self.max_step_size = max_step_size

        self.goal_id=None
        #added timeout
        self.timeout=None

        if self.max_itr is not None:
            self.best_cost_per_iter = np.full(self.max_itr, np.inf)
            self.success_per_iter = np.zeros(self.max_itr)
        else:
            self.best_cost_per_iter = []
            self.success_per_iter = []



        self.best_cost_per_t = []
        self.success_per_t = []

        self.two_d=False
        self.three_d=False

    def plan(self):
        """
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        """
        # TODO: HW3 3
        i=0
        t1 = time.time()
        d=len(self.start)
        self.tree.add_vertex(self.start)
        while True:
            if self.timeout:
                if time.time() -t1 > self.timeout:
                    break
            if self.max_itr is not None:
                if i > self.max_itr:
                    break

            rand_config = self.bb.sample_random_config(self.goal_prob, self.goal)
            near_id, near_config = self.tree.get_nearest_config(rand_config)
            new_config = self.extend(near_config, rand_config)
            if new_config is None:
                if self.three_d:
                    self.update_statistics(i)
                if self.two_d:
                    self.update_statistics2d(time.time() - t1)
                i += 1
                continue
            if self.bb.edge_validity_checker(near_config, new_config):
                new_id = self.tree.add_vertex(new_config)
                edge_cost = self.bb.compute_distance(near_config, new_config)
                self.tree.add_edge(near_id, new_id, edge_cost)

                if  np.array_equal(new_config, self.goal):
                    if self.stop_on_goal:
                        break


                if self.k is not None:
                    k_i=min(self.k,len(self.tree.vertices)-1)
                else:
                    n_vertices=len(self.tree.vertices)
                    k_i = int(np.ceil(2 * (1 + 1/d) * np.log(n_vertices)))
                    k_i = min(k_i, n_vertices - 1)
                    k_i = max(k_i, 1)

                x_near_id,x_near=self.tree.get_k_nearest_neighbors(new_config, k_i)
                # rewire neighbors
                for n_id in x_near_id:
                    self.rewire(n_id, new_id, new_config)
                    if self.timeout:
                        if time.time() - t1 > self.timeout:
                            break

            if self.three_d:
                self.update_statistics(i)
            if self.two_d:
                self.update_statistics2d(time.time()-t1)
            i+=1

        plan = []
        self.goal_id=self.tree.get_idx_for_config(self.goal)
        if self.goal_id is None:
            print("No plan was found")
            return None
        print("Goal:",self.goal_id)


        current = self.goal_id
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
        return np.array(plan)

    def rewire(self, n_id, new_id, new_config):
        x_near = self.tree.vertices[n_id]
        new_vertex = self.tree.vertices[new_id]

        # potential new cost
        edge_cost = self.bb.compute_distance(new_config, x_near.config)
        new_cost = new_vertex.cost + edge_cost

        # check edge validity
        if self.bb.edge_validity_checker(new_config, x_near.config) and new_cost < x_near.cost:
            # change parent
            self.tree.edges[n_id] = new_id
            x_near.set_cost(new_cost)

            self.propagate_costs(n_id)

    def update_statistics(self, itr):
        """
        Update success and best cost statistics at iteration itr

        """
        if self.tree.is_goal_exists(self.goal):
            success = 1
            goal_ver=self.tree.get_vertex_for_config(self.goal)
            best_cost = goal_ver.cost
        else:
            best_cost = np.inf
            success = 0

        if self.max_itr is None:
            # dynamic logging
            self.best_cost_per_iter.append(best_cost)
            self.success_per_iter.append(success)
        else:
            # fixed-size logging
            if itr < self.max_itr:
                self.best_cost_per_iter[itr] = best_cost
                self.success_per_iter[itr] = success

    def update_statistics2d(self, t_elapsed):
        """
        Update success and best cost statistics based on elapsed time.
        Logs only when the best cost improves.
        """
        if self.tree.is_goal_exists(self.goal):
            best_cost = self.tree.get_vertex_for_config(self.goal).cost
            success = 1
        else:
            best_cost = np.inf
            success = 0

        # only log when best cost improves
        if not hasattr(self, "best_cost_so_far"):
            self.best_cost_so_far = np.inf

        if best_cost < self.best_cost_so_far:
            self.best_cost_so_far = best_cost
            self.best_cost_per_t.append((t_elapsed, best_cost))
            self.success_per_t.append((t_elapsed, success))

    def propagate_costs(self, vertex_id):
        """
        Recursively update the costs of all descendants of the given vertex.
        """
        vertex = self.tree.vertices[vertex_id]

        # iterate over all children of this vertex
        for child_id, parent_id in self.tree.edges.items():
            if parent_id == vertex_id:
                child = self.tree.vertices[child_id]
                # update cost of child
                child.set_cost(vertex.cost + self.bb.compute_distance(vertex.config, child.config))
                # recursively update its descendants
                self.propagate_costs(child_id)


    def compute_cost(self, plan):
        # TODO: HW3 3
        cost = 0
        for idx in range(len(plan) - 1):
            cost += self.bb.compute_distance(plan[idx], plan[idx + 1])
        return cost

    def extend(self, x_near, x_rand):
        # TODO: HW3 3
        if np.allclose(x_rand, x_near):
            return None

        if self.ext_mode == "E1":
            new_config = x_rand
        else:
            eta = self.max_step_size
            vec = x_rand - x_near
            dist = self.bb.compute_distance(x_near, x_rand)
            if dist <= eta:
                new_config = x_rand
            else:
                direction = vec / dist
                new_config = x_near + direction * eta

        if self.bb.config_validity_checker(new_config):
            return new_config

        return None
