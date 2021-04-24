import gym
from Q_Learning import Agent
import time
import os
import matplotlib.pyplot as plt

def generate_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


# Plot Q-Learning Curves
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
    plt.plot(iterations, data_list[1], label="lr 0.1, e 0.9", color='black')
    plt.plot(iterations, data_list[2], label="lr 0.9, e 0.1", color='blue')
    plt.plot(iterations, data_list[3], label="lr 0.9, e 0.9", color='orange')

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
    epi = 10000
    test_epi = 10000

    env = gym.make("FrozenLake8x8-v0")

    for learning_rate in [0.1, 0.9]:

        for exploration_rate in [0.1, 0.9]:
            print("learning rate : ", learning_rate)
            print(" exploration rate: ", exploration_rate)
            rewards = []
            times = []
            q_vals = []
            total_r = 0.0

            agent = Agent(env=env, learn_rate=learning_rate, epsilon=exploration_rate)


            for i in range(epi - 1):
                start_time = time.time()
                e_reward, e_q = agent.learn()
                end_time = time.time()

                times.append(end_time - start_time)
                total_r += e_reward
                rewards.append(total_r)

            print("total reward is : ", total_r)

            rewards_list.append(rewards)
            times_list.append(times)


    iterations = [i for i in range(1, epi)]

    draw_Q_curve(rewards_list, iterations, "Average Reward", "FrozenLake 8 * 8")
    draw_Q_curve(times_list, iterations, "Average Time", "FrozenLake 8 * 8")





