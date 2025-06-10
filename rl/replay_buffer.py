import numpy as np

class ReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, state_dim, action_dim, max_size=int(1e6), alpha=0.6, beta=0.4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.beta = beta

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.priorities = np.zeros(max_size, dtype=np.float32)

    def add(self, state, action, reward, done, next_state):
        index = self.ptr
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.next_states[index] = next_state

        max_prio = self.priorities[:self.size].max() if self.size > 0 else 1.0
        self.priorities[index] = max_prio

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        assert self.size > 0, "Replay buffer is empty"
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return (self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.dones[indices],
                self.next_states[indices],
                indices,
                weights)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-3
