import numpy as np


class BuildingBlocks3D(object):
    """
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    """

    def __init__(self, transform, ur_params, env, resolution=0.1):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechanical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [
            ["shoulder_link", "forearm_link"],
            ["shoulder_link", "wrist_1_link"],
            ["shoulder_link", "wrist_2_link"],
            ["shoulder_link", "wrist_3_link"],
            ["upper_arm_link", "wrist_1_link"],
            ["upper_arm_link", "wrist_2_link"],
            ["upper_arm_link", "wrist_3_link"],
            ["forearm_link", "wrist_2_link"],
            ["forearm_link", "wrist_3_link"],
        ]

    def sample_random_config(self, goal_prob, goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        # TODO: HW2 5.2.1
        # mechanical limits to each link
        limits = self.ur_params.mechanical_limits
        # generating a random number [0,1), for probability of goal_prob - return the goal
        if np.random.rand() < goal_prob:
            return np.array(goal_conf)

        # initialize for a new sample
        sample = []
        # generating a random sample within limits
        for link in limits:
            low, high = limits[link]  # link i limits
            sample.append(np.random.uniform(low=low, high=high))  # generate a random angle in limits
        return np.array(sample)

    def spheres_collide(self, c1, r1, c2, r2):
        """"
        c1: (N,3) array of spheres
        r1: (N,) array of radii
        c2: (M,3) array of spheres
        r2: (M,) array of radii
        Returns True if any spheres collide
        """
        diff = c1[:, None, :] - c2[None, :, :]  # shape (N,M,3)
        dist_sq = np.sum(diff**2, axis=2)       # shape (N,M)
        radius_sum = (r1[:, None] + r2[None, :])**2
        return np.any(dist_sq <= radius_sum)

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return False if in collision
        @param conf - some configuration
        """
        # TODO: HW2 5.2.2- Pay attention that function is a little different than in HW2
        sphere_coords = self.transform.conf2sphere_coords(conf)
        spheres_radius = self.transform.sphere_radius
        links = list(sphere_coords.keys())


        # wall collision
        all_centers = np.vstack([sphere_coords[link] for link in links])
        if np.any(all_centers[:, 0] > 0.4):
            return False

        # Self-collision
        for link1, link2 in self.possible_link_collisions:
            c1 = np.asarray(sphere_coords[link1])  # (N,3)
            c2 = np.asarray(sphere_coords[link2])  # (M,3)
            r1 = np.full(len(c1), spheres_radius[link1])  # (N,)
            r2 = np.full(len(c2), spheres_radius[link2])  # (M,)

            if self.spheres_collide(c1, r1, c2, r2):
                return False

        #  Obstacle collision
        obstacles = np.asarray(self.env.obstacles)  # (K,3)
        r_ob = np.full(len(obstacles), self.env.radius)  # (K,)

        for link in links:
            c = np.asarray(sphere_coords[link])  # (N,3)
            r = np.full(len(c), spheres_radius[link])  # (N,)

            if self.spheres_collide(c, r, obstacles, r_ob):
                return False

        # Floor collision (except first link)
        for i, link in enumerate(links):
            if i == 0:
                continue
            c = np.asarray(sphere_coords[link])  # (N,3)
            r = spheres_radius[link]
            if np.any(c[:, 2] - r <= 0):
                return False

        return True


    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        """check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        """
        # TODO: HW2 5.2.4
        # distance between configurations
        dist = self.compute_distance(prev_conf, current_conf)
        min_confs = 3  # minimum configurations
        # compute number of interpolation steps based on resolution
        steps = max(int(dist / self.resolution), min_confs)
        ts = np.linspace(0, 1, steps)[:, None]  # shape (steps,1)
        qs = (1 - ts) * prev_conf + ts * current_conf  # shape (steps, dim)
        for q in qs:
            if not self.config_validity_checker(q):
                return False
        return True

    def compute_distance(self, conf1, conf2):
        """
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        """
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5
