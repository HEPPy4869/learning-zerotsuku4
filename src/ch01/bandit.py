from unittest import result
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)

def main():
    save_doc_dir = '../../docs/ch01/'

    steps = 1000
    epsilon = 0.1
    
    runs = 10

    rewards_fig = plt.figure()
    reward_ax = rewards_fig.add_subplot(111, ylabel='Total reward', xlabel='Steps')
    rate_fig = plt.figure()
    rate_ax = rate_fig.add_subplot(111, ylabel='Rates', xlabel='Steps')
    all_rate_fig = plt.figure()
    all_rate_ax = all_rate_fig.add_subplot(111, ylabel='Rates', xlabel='Steps')

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        total_rewards = []
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward

            total_rewards.append(total_reward)
            rates.append(total_reward / (step + 1))

        all_rate_ax.plot(rates)
        print(total_reward)

    reward_ax.plot(total_rewards)
    rate_ax.plot(rates)
    rewards_fig.savefig(save_doc_dir+'1-12.png')
    rate_fig.savefig(save_doc_dir+'1-13.png')
    all_rate_fig.savefig(save_doc_dir+'1-14.png')


if __name__ == '__main__':
    main()