import numpy as np

class Environment():

    def __init__(self, n_particles):
        self.action_size = 4
        self.dt = 1.0/30.0
        self.mass = 0.05
        self.dv_agent, self.v_particles = 0.01, 0.05
        self.radius_agent, self.radius_particles = 0.05, 0.1
        self.box_half_length = 2.0
        self.add_wall_collisions = True
        self.add_agent_particle_interactions = True
        # self.state = np.array([x_pos, y_pos, x_vel, y_vel, type])
        self.state_init = self._init_state(n_particles)
        self.types_detect = [0, 1]
        # Frame
        self.box_dict = {"x_min": 0, "x_max": 1, "y_min": 2, "y_max": 3}
        self.walls = self._get_walls()
        # Make checks
        if self.state_init.shape[1] != 5:
            raise Exception("initial state has wrong dimensions!")
        self.reset()

    def _init_state(self, n_particles):
        state = -0.5 + np.random.random((1 + n_particles, 5))  # initialize
        state[:, [0, 1]] *= 4.0  # spread out particle positions
        state[0, 4] = -1  # agent type = -1
        state[1:, 4] = 1  # particles type = 1
        if n_particles > 0:
            v_norm = np.linalg.norm(state[1:, [2, 3]], axis=1)  # velocity magnitudes
            # Make v_speed = 0.05, v_direction = random
            state[1:, [2, 3]] = self.v_particles * np.divide(state[1:, [2, 3]], v_norm[:, None])
        state[0, [2, 3]] = 0.0  # zero vx, zero vy
        return state

    def reset(self):
        self.state = self.state_init.copy()
        self.reward_history = [] # rewards of episode
        self.state_history = [] # states of episode
        return self.state


    def step(self, action):
        # self.action = 0:"up", 1:"right", 2:"down", 3:"left"
        # self.state = 0:"x", 1:"y", 2:"vx", 3:"vy", 4:"type"
        # Build next state
        state, reward, done = self.state.copy(), 0.0, False
        if self.state[0, 4] >= 0: # check agent is of type < 0
            raise Exception("err: Trying to perform action on non-agent particle!")
        if action == 0:     # up: vy += dv
            self.state[0, 3] += self.dv_agent
        elif action == 1:   # right: vx += dv
            self.state[0, 2] += self.dv_agent
        elif action == 2:   # down: vy -= dv
            self.state[0, 3] += -self.dv_agent
        elif action == 3:   # left: vx -= dv
            self.state[0, 2] += -self.dv_agent
        else:
            raise Exception("invalid action!")
        # Update positions: x = x + v * dt
        self.state[:, 0] += self.state[:, 2] * self.dt
        self.state[:, 1] += self.state[:, 3] * self.dt
        # Add wall interactions
        if self.add_wall_collisions:
            # Box wall locations
            xmin_box, xmax_box = -self.box_half_length, self.box_half_length
            ymin_box, ymax_box = -self.box_half_length, self.box_half_length
            # Find the radius of every object
            radius_state = self.radius_particles * np.ones(len(self.state), dtype=np.float)
            radius_state[0] = self.radius_agent
            # Find which particles/agents crossed the wall (they need to be rebounded back in)
            crossed_x1 = (self.state[:, 0] < xmin_box + radius_state[:])
            crossed_x2 = (self.state[:, 0] > xmax_box - radius_state[:])
            crossed_y1 = (self.state[:, 1] < ymin_box + radius_state[:])
            crossed_y2 = (self.state[:, 1] > ymax_box - radius_state[:])
            # Update locations after wall collision
            self.state[crossed_x1, 0] = xmin_box + radius_state[crossed_x1]
            self.state[crossed_x2, 0] = xmax_box - radius_state[crossed_x2]
            self.state[crossed_y1, 1] = ymin_box + radius_state[crossed_y1]
            self.state[crossed_y2, 1] = ymax_box - radius_state[crossed_y2]
            # Update velocities after wall collision (flip velocities using momentum equations)
            self.state[crossed_x1|crossed_x2, 2] *= -1
            self.state[crossed_y1|crossed_y2, 3] *= -1
        # Add agent-particle interactions (gives rewards)
        if self.add_agent_particle_interactions:
            xy_particles_relative = self.state[1:, [0, 1]] - self.state[0, [0, 1]]
            dist_particles = np.linalg.norm(xy_particles_relative, axis=1)  # distance between agent-particles
            idxs_captured = np.where(dist_particles < (self.radius_agent + self.radius_particles))[0] + 1
            types_captured = np.array(self.state[idxs_captured, 4], dtype=np.int)
            n_type_captured = [0, int(np.sum(types_captured == 1))]
            self.state = np.delete(self.state, idxs_captured, axis=0)
            reward += 1  # reward for surviving
            done = (n_type_captured[1] > 0)
        self.reward_history.append(reward) # track reward
        self.state_history.append(self.state) # track states
        return self.state, reward, done

    def _get_walls(self):
        corner_1 = np.array([-self.box_half_length, -self.box_half_length])
        corner_2 = np.array([-self.box_half_length, self.box_half_length])
        corner_3 = np.array([self.box_half_length, self.box_half_length])
        corner_4 = np.array([self.box_half_length, -self.box_half_length])
        walls = [(corner_1, corner_2), (corner_2, corner_3), (corner_3, corner_4), (corner_4, corner_1)]
        return walls
