import gym
import numpy as np

MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05  # Learning rate
GAMMA = 0.98  # Discount factor
NUM_DISCRETE_BINS = 30  # Number of bins to Discretize each observation dim


class SARSA_Learner(object):
    def __init__(self, env):

        # Set the observation parameters for discretization
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins

        # Set the action space and initialize Q-values table
        self.action_shape = env.action_space.n
        self.Q_Table = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.action_shape))

        # Set learning parameters
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

    def discretize(self, obs):
        """Discretize the continuous observation space into discrete bins."""
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        """Epsilon-greedy action selection."""
        discretized_obs = self.discretize(obs)
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q_Table[discretized_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs, next_action):
        """Update Q-values using SARSA learning rule."""
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * self.Q_Table[discretized_next_obs][next_action]
        td_error = td_target - self.Q_Table[discretized_obs][action]
        self.Q_Table[discretized_obs][action] += self.alpha * td_error


def train(agent, env):
    """Train the agent using the SARSA learning algorithm."""
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs, _ = env.reset()
        action = agent.get_action(obs)
        total_reward = 0.0

        # Loop through the environment until the episode ends
        while not done:
            next_obs, reward, done, info, _ = env.step(action)
            next_action = agent.get_action(next_obs)
            agent.learn(obs, action, reward, next_obs, next_action)
            obs = next_obs
            action = next_action
            total_reward += reward

        # Logging
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode, total_reward, best_reward, agent.epsilon))

    # Return the trained policy
    return np.argmax(agent.Q_Table, axis=2)


def test(agent, env, policy):
    """Test the agent with the learned policy."""
    done = False
    obs, _ = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info, _ = env.step(action)
        obs = next_obs
        total_reward += reward

    return total_reward


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    agent = SARSA_Learner(env)
    learned_policy = train(agent, env)

    # Record some episodes using the learned policy
    gym_monitor_path = "./gym_monitor_output"
    env = gym.wrappers.RecordVideo(env, gym_monitor_path, episode_trigger=lambda x: x % 100 == 0)
    env.reset()
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()
