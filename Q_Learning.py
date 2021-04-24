
import numpy as np

class Agent:
    def __init__(self, env, epsilon=0.2, learn_rate=0.8):
        self.env = env
        self.gamma = 0.99

        self.episode_reward = 0.0
        self.epsilon = epsilon
        self.learn_rate = learn_rate
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q[state, :])
        return action

    def learn(self):
        # one episode learning
        state = self.env.reset()
        t_reward = 0

        for t in range(100000000):
            act = self.choose_action(state)
            next_state, reward, done, info = self.env.step(act)
            t_reward += reward

            q_next_max = np.max(self.q[next_state])
            self.q[state, act] += self.learn_rate * (reward + self.gamma * np.max(self.q[next_state, :]) - self.q[state, act])
            if done:
                return t_reward, np.sum(self.q)
            else:
                state = next_state

    def test(self):
        state = self.env.reset()
        for t in range(100000000):
            act = np.argmax(self.q[state])
            next_state, reward, done, info = self.env.step(act)
            if done:
                return reward
            else:
                state = next_state
        return 0.0
