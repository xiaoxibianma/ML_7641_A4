import numpy as np
import gym
import time
import os
import matplotlib.pyplot as plt

# ref https://gym.openai.com/envs/FrozenLake-v0/


# action list
action_mapping = {
    0: '\u2190',  # LEFT
    1: '\u2193',  # DOWN
    2: '\u2192',  # RIGHT
    3: '\u2191',  # UP
}

def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


# 2. Setup GYM Env for playing
# we define a faction that will take a GYM environment and plays number of games according to given policy.
def play_episodes(enviorment, n_episodes, policy, random=False):
    """
    This fucntion plays the given number of episodes given by following a policy or sample randomly from action_space.

    Parameters:
        enviorment: openAI GYM object
        n_episodes: number of episodes to run
        policy: Policy to follow while playing an episode
        random: Flag for taking random actions. if True no policy would be followed and action will be taken randomly

    Return:
        wins: Total number of wins playing n_episodes
        total_reward: Total reward of n_episodes
        avg_reward: Average reward of n_episodes

    """
    # initialize wins and total reward
    wins = 0
    total_reward = 0
    # reward_list = []

    # loop over number of episodes to play
    for episode in range(n_episodes):
        # flag to check if the game is finished
        terminated = False
        # reset the enviorment every time when playing a new episode
        state = enviorment.reset()
        # episode_reward = 0
        # i = 0

        while not terminated:
            # i += 1
            # check if the random flag is not true then follow the given policy other wise take random action
            if random:
                action = enviorment.action_space.sample()
            else:
                action = policy[state]

            # take the next step
            next_state, reward, terminated, info = enviorment.step(action)
            # enviorment.render()
            # accumalate total reward
            total_reward += reward
            # episode_reward += reward
            # change the state
            state = next_state

            # if game is over with positive reward then add 1.0 in wins
            if terminated and reward == 1.0:
                wins += 1
        # reward_list.append(episode_reward / i)

    # calculate average reward
    average_reward = total_reward / n_episodes
    return wins, total_reward, average_reward


def extract_policy(env, v, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


# 3. Solve for Value Iteration.
def one_step_lookahead(env, state, V, discount_factor=0.99):
    """
    Helper function to  calculate state-value function

    Arguments:
        env: openAI GYM Enviorment object
        state: state to consider
        V: Estimated Value for each state. Vector of length nS
        discount_factor: MDP discount factor

    Return:
        action_values: Expected value of each action in a state. Vector of length nA
    """

    # initialize vector of action values
    action_values = np.zeros(env.nA)

    # loop over the actions we can take in an enviorment
    for action in range(env.nA):
        # loop over the P_sa distribution.
        for probablity, next_state, reward, info in env.P[state][action]:
            # if we are in state s and take action a. then sum over all the possible states we can land into.
            action_values[action] += probablity * (reward + (discount_factor * V[next_state]))

    return action_values


def update_policy(env, policy, V, discount_factor):
    """
    Helper function to update a given policy based on given value function.

    Arguments:
        env: openAI GYM Enviorment object.
        policy: policy to update.
        V: Estimated Value for each state. Vector of length nS.
        discount_factor: MDP discount factor.
    Return:
        policy: Updated policy based on the given state-Value function 'V'.
    """

    for state in range(env.nS):
        # for a given state compute state-action value.
        action_values = one_step_lookahead(env, state, V, discount_factor)
        # choose the action which maximizez the state-action value.
        policy[state] = np.argmax(action_values)

    return policy


def start_iteration(env, discount_factor=0.999, max_iteration=800):
    """
    Algorithm to solve MPD.

    Arguments:
        env: openAI GYM Enviorment object.
        discount_factor: MDP discount factor.
        max_iteration: Maximum No.  of iterations to run.

    Return:
        V: Optimal state-Value function. Vector of lenth nS.
        optimal_policy: Optimal policy. Vector of length nS.

    """
    # intialize value fucntion
    V = np.zeros(env.nS)
    v_list = []
    time_list = []
    reward_list = []
    time_per_iter = 0

    # iterate over max_iterations
    for i in range(1500):
        #  keep track of change with previous value function
        start_time = time.time()
        prev_v = np.copy(V)
        prev_policy = extract_policy(env, prev_v)
        wins, total_reward, average_reward = play_episodes(env, 100, prev_policy, random=False)
        reward_list.append(average_reward)

        # loop over all states
        for state in range(env.nS):
            # Asynchronously update the state-action value
            # action_values = one_step_lookahead(env, state, V, discount_factor)

            # Synchronously update the state-action value
            action_values = one_step_lookahead(env, state, prev_v, discount_factor)
            # select best action to perform based on highest state-action value
            best_action_value = np.max(action_values)
            # update the current state-value fucntion
            V[state] = best_action_value
        end_time = time.time()
        time_per_iter = end_time - start_time
        v_list.append(np.sum(np.fabs(prev_v - V)))
        time_list.append(time_per_iter)

        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            # if values of 'V' not changing after one iteration
            if (np.all(np.isclose(V, prev_v))):
                print('Value converged at iteration %d' % (i + 1))


    # intialize optimal policy
    optimal_policy = np.zeros(env.nS, dtype='int8')
    # update the optimal polciy according to optimal value function 'V'
    optimal_policy = update_policy(env, optimal_policy, V, discount_factor)

    return V, optimal_policy, v_list, time_list, reward_list


# 4. Solve for Policy Iteration
def policy_eval(env, policy, V, discount_factor):
    """
    Helper function to evaluate a policy.

    Arguments:
        env: openAI GYM Enviorment object.
        policy: policy to evaluate.
        V: Estimated Value for each state. Vector of length nS.
        discount_factor: MDP discount factor.
    Return:
        policy_value: Estimated value of each state following a given policy and state-value 'V'.

    """
    policy_value = np.zeros(env.nS)
    for state, action in enumerate(policy):
        for probablity, next_state, reward, info in env.P[state][action]:
            policy_value[state] += probablity * (reward + (discount_factor * V[next_state]))

    return policy_value


def policy_iteration(env, discount_factor=0.999, max_iteration=1500):


    # intialize the state-Value function
    V = np.zeros(env.nS)

    policy = np.random.randint(0, 4, env.nS)
    policy_prev = np.copy(policy)
    policy_list = []
    time_list = []
    reward_list = []

    for i in range(max_iteration):
        start_time = time.time()
        # evaluate prev policy v
        old_policy_v = policy_eval(env, policy_prev, V, discount_factor)

        # evaluate given policy
        V = policy_eval(env, policy, V, discount_factor)
        # improve policy
        policy = update_policy(env, policy, V, discount_factor)
        wins, total_reward, average_reward = play_episodes(env, 100, policy, random=False)
        reward_list.append(average_reward)

        end_time = time.time()
        time_per_iter = end_time - start_time
        policy_list.append(np.sum(np.fabs(V - old_policy_v)))
        time_list.append(time_per_iter)

        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            if (np.all(np.equal(policy, policy_prev))):
                print('policy converged at iteration %d' % (i + 1))
            policy_prev = np.copy(policy)

    return V, policy, policy_list, time_list, reward_list


def plot_grid_map(policy_v, algorithm):
    title = "State Values for {} algorithm on fronzen grid 4 * 4 (tyang358)".format(algorithm)
    fig, ax = plt.subplots()
    im = ax.imshow(policy_v)

    ax.set_xlim([-0.5, len(policy_v[0]) - 0.5])
    ax.set_ylim([len(policy_v) - 0.5, -0.5])

    for i in range(len(policy_v)):
        for j in range(len(policy_v[0])):
            text = ax.text(j, i, policy_v[i, j], ha="center", va="center", color="b", fontsize=12)

    ax.set_title(title, fontsize=12)
    fig.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', title + '.png')
    plt.savefig(image_path, dpi=100)
    return


def plot_converge_curve(pi_list, vi_list, iters, problem, y_val):

    plt.close()

    plt.figure()
    plt.title("{} plot for PI and VI on {} (tyang358)".format(y_val, problem))
    plt.xlabel("Number of iterations")
    plt.ylabel(y_val)


    plt.grid()
    plt.plot(iters, pi_list, label="PI", color='black')
    plt.plot(iters, vi_list, label="VI", color='orange')
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} plot for PI and VI on {}".format(y_val, problem))
    plt.savefig(image_path, dpi=100)
    return


def plot_episode_reward_curve(vi_list, pi_list, n_episodes, problem):
    plt.close()
    n_episodes = [i for i in range(n_episodes)]

    # Create plot
    plt.figure()
    plt.title("Average reward for episodes for PI and VI on {} (tyang358)".format(problem), size=8)
    plt.xlabel("episodes number")
    plt.ylabel("Reward")

    # Draw lines
    plt.grid()
    plt.plot(n_episodes, pi_list, label="PI", color='blue')
    plt.plot(n_episodes, vi_list, label="VI", color='yellow')
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "Count reward PI VI for {}".format(problem))
    plt.savefig(image_path, dpi=100)
    return


if __name__ == "__main__":

    n_episodes = 1500
    iterations = [i for i in range(1500)]


    # make a 4 * 4 game
    game = gym.make('FrozenLake-v0')

    start_time = time.time()

    opt_V, opt_Policy, v_list, v_time_list, v_reward_list = start_iteration(game.env, max_iteration=1500)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000

    print(f"Time to converge: {elapsed_time: 0.3} ms")
    print('Optimal Value function: ')
    print(opt_V.reshape((4, 4)))
    print('Final Policy: ')
    print(opt_Policy)
    print(' '.join([action_mapping[int(action)] for action in opt_Policy]))
    # plot grid map with state values
    opt_v_ref = np.round(opt_V.copy(), 2)
    plot_grid_map(opt_v_ref.reshape((4, 4)), "PI")

    # Test the PI Algorithim
    game2 = gym.make('FrozenLake-v0')
    start_time = time.time()
    opt_V2, opt_policy2, policy_v_list, policy_time_list, policy_reward_list = policy_iteration(game2.env,
                                                                                                 discount_factor=0.999,
                                                                                                 max_iteration=1500)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000


    print(f"Time to converge: {elapsed_time: 0.3} ms")
    print('Optimal Value function: ')
    print(opt_V2.reshape((4, 4)))
    print('Final Policy: ')
    print(opt_policy2)
    print(' '.join([action_mapping[(action)] for action in opt_policy2]))
    # plot grid map with state values
    opt_v2_ref = np.round(opt_V2.copy(), 2)
    plot_grid_map(opt_v2_ref.reshape((4, 4)),  "VI on 4 * 4")

    size = int(1500 / 50)
    chunk_pi_times = list(chunk_list(policy_time_list, size))
    chunk_vi_times = list(chunk_list(v_time_list, size))
    average_pi_times = [sum(chunk) / len(chunk) for chunk in chunk_pi_times]
    average_vi_times = [sum(chunk) / len(chunk) for chunk in chunk_vi_times]

    plot_converge_curve(policy_v_list, v_list, iterations, "Frozen Lake 4 * 4", "Convergence")
    plot_converge_curve(average_vi_times, average_pi_times, [i for i in range(1, 1500, 30)], "Frozen Lake 4 * 4", "Time")

    plot_episode_reward_curve(v_reward_list, policy_reward_list, n_episodes, "Frozen Lake 4 * 4")


