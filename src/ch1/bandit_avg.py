import numpy as np
import matplotlib.pyplot as plt

from bandit import *

def main():
    save_doc_dir = '../../docs/ch1/'
    
    runs = 200
    steps = 1000
    epsilons = [0.3, 0.1, 0.01]
    all_rates = np.zeros((runs, steps))

    epsilon_fig = plt.figure()
    epsilon_ax = epsilon_fig.add_subplot(111, ylabel='Rates', xlabel='Steps')

    for epsilon in epsilons:
        for run in range(runs):
            bandit = Bandit()
            agent = Agent(epsilon)
            total_reward = 0
            rates = []

            for step in range(steps):
                action = agent.get_action()
                reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step+1))

            all_rates[run] = rates

        avg_rates = np.average(all_rates, axis=0)
        epsilon_ax.plot(avg_rates, label=epsilon)

        if epsilon == 0.1:
            
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Rates', xlabel='Steps')
            ax.plot(avg_rates)
            fig.savefig(save_doc_dir+'1-16.png')
    
    epsilon_ax.legend()
    epsilon_fig.savefig(save_doc_dir+'1-17.png')
    

    


if __name__ == '__main__':
    main()