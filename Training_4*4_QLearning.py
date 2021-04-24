import gym
from gym import wrappers
from Q_Learning import Agent
import time
import numpy as np
import os
import matplotlib.pyplot as plt

# Generate Lists
def generate_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Plot Q-Learning curves
def draw_Q_curve(data_list, iterations, para, name):
    plt.close()

    # Create plot
    plt.figure()
    plt.title("Q-Learning comparison for {} on problem {} (tyang358)".format(para, name))
    plt.xlabel("exploration")
    plt.ylabel(para)

    # Draw lines
    plt.grid()
    plt.plot(iterations, data_list[0], label="lr 0.1, e 0.1", color='green')
    plt.plot(iterations, data_list[1], label="lr 0.1, e 0.3", color='black')
    plt.plot(iterations, data_list[2], label="lr 0.1, e 0.6", color='blue')
    plt.plot(iterations, data_list[3], label="lr 0.1, e 0.9", color='orange')
    plt.plot(iterations, data_list[4], '--', label="lr 0.9, e 0.1", color='green')
    plt.plot(iterations, data_list[5], '--', label="lr 0.9, e 0.3", color='black')
    plt.plot(iterations, data_list[6], '--', label="lr 0.9, e 0.6", color='blue')
    plt.plot(iterations, data_list[7], '--', label="lr 0.9, e 0.9", color='orange')

    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "Q-Learning comparison on problem {} for parameter {} ".format(name, para))
    plt.savefig(image_path, dpi=100)
    return


if __name__ == "__main__":

    rewards_list = []
    times_list = []
    converage_list = []


    epi = 10000
    test_epi = 10000

    env = gym.make("FrozenLake-v0")
    env = wrappers.Monitor(env, './FrozenLake-v0', force=True)


    for learning_rate in [0.1, 0.9]:

        for exploration_rate in [0.1, 0.3, 0.6, 0.9]:
            rewards = []
            times = []
            q_vals = []
            total_r = 0.0

            agent = Agent(env=env, learn_rate=learning_rate, epsilon=exploration_rate)


            for i in range(epi):
                start_time = time.time()
                e_reward, e_q = agent.learn()
                end_time = time.time()

                times.append(end_time - start_time)
                total_r += e_reward
                rewards.append(e_reward)
                q_vals.append(e_q)

            print("total reward is : ", total_r)
            print("average reward: {:.2f}".format(total_r / epi))

            total_r = 0.0
            for i in range(test_epi):
                total_r += agent.test()

            print("total reward is : ", total_r)
            print("average reward: {:.2f}".format(total_r / test_epi))

            size = int(epi / 100)
            chunk_reward = list(generate_list(rewards, size))
            chunk_time = list(generate_list(times, size))
            chunk_q = list(generate_list(q_vals, size))
            reward_av = [sum(chunk) / len(chunk) for chunk in chunk_reward]
            time_av = [sum(chunk) / len(chunk) for chunk in chunk_time]
            converage_at = [sum(chunk) / len(chunk) for chunk in chunk_q]

            rewards_list.append(reward_av)
            times_list.append(time_av)
            converage_list.append(converage_at)

    iterations = [i for i in range(1, epi, 100)]

    draw_Q_curve(rewards_list, iterations, "Average Reward", "FrozenLake 4 * 4")
    draw_Q_curve(times_list, iterations, "Average Time", "FrozenLake 4 * 4")
    draw_Q_curve(converage_list, iterations, "Convergence At", "FrozenLake 4 * 4")



