import numpy as np


class VelodyneEnv:
    """Simplified environment placeholder with shaped rewards."""

    def __init__(self, goal_threshold: float = 0.2):
        self.state_dim = 24
        self.action_dim = 2
        self.goal_threshold = goal_threshold
        self.prev_dist_to_goal = None
        self.reset()

    def reset(self):
        self.robot_pos = np.array([0.0, 0.0])
        self.goal = np.array([5.0, 0.0])
        self.prev_dist_to_goal = np.linalg.norm(self.goal - self.robot_pos)
        return self._get_state()

    def step(self, action):
        lin_vel, ang_vel = action
        self.robot_pos += np.array([lin_vel * 0.1, 0])
        next_state = self._get_state()

        curr_dist = np.linalg.norm(self.goal - self.robot_pos)
        distance_diff = self.prev_dist_to_goal - curr_dist
        progress_reward = distance_diff
        self.prev_dist_to_goal = curr_dist

        d_min = self._fake_lidar().min()
        reward = progress_reward
        safe_dist = 0.5
        if d_min < safe_dist:
            reward -= 0.5 * (safe_dist - d_min)
        reward -= 0.05 * abs(ang_vel)

        done = False
        if curr_dist < self.goal_threshold:
            reward += 100.0
            done = True
        if d_min < 0.05:
            reward -= 100.0
            done = True

        return next_state, reward, done, {}

    def _get_state(self):
        lidar = self._fake_lidar()
        robot_state = np.zeros(4, dtype=np.float32)
        return np.concatenate([lidar, robot_state])

    def _fake_lidar(self):
        # generate fake lidar distances between 0.1 and 3.5
        return np.random.uniform(0.1, 3.5, size=20).astype(np.float32)

