import gym
import numpy as np
import time
from matplotlib import pyplot as plt
import os

def play_episodes(environment, n_episodes, policy):
    wins = 0
    total_reward = 0

    for episode in range(n_episodes):

        terminated = False
        state = environment.reset()

        while not terminated:

            # Select best action to perform in a current state
            action = np.argmax(policy[state])

            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)

            # Summarize total reward
            total_reward += reward

            # Update current state
            state = next_state

            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                wins += 1

    average_reward = total_reward / n_episodes

    return wins, total_reward, average_reward



def next_state(environment, state, V, discount_factor):
    """
    Helper function to calculate a state-value function.
    :param environment: Initialized OpenAI gym environment object.
    :param state: Agent's state to consider (integer).
    :param V: The value to use as an estimator. Vector of length nS.
    :param discount_factor: MDP discount factor.
    :return: A vector of length nA containing the expected value of each action.
    """

    action_values = np.zeros(environment.nA)

    for action in range(environment.nA):

        for probability, next_state, reward, terminated in environment.P[state][action]:
            action_values[action] += probability * (reward + discount_factor * V[next_state])

    return action_values



def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iter=100):
    """
    Evaluate a policy given a deterministic environment.
    :param policy: Matrix of a size nSxnA, each cell represents a probability of taking action a in state s.
    :param environment: Initialized OpenAI gym environment object.
    :param discount_factor: MDP discount factor. Float in range from 0 to 1.
    :param theta: A threshold of a value function change.
    :param max_iter: Maximum number of iteration to prevent infinite loops.
    :return: A vector of size nS, which represent a value function for each state.
    """

    # Number of evaluation iterations
    evaluation_iterations = 1

    # Initialize a value function for each state as zero
    V = np.zeros(environment.nS)

    # Repeat until value change is below the threshold
    for i in range(int(max_iter)):

        # Initialize a change of value function as zero
        delta = 0

        # Iterate though each state
        for state in range(environment.nS):

            # Initial a new value of current state
            v = 0

            # Try all possible actions which can be taken from this state
            for action, action_probability in enumerate(policy[state]):

                # Evaluate how good each next state will be
                for state_probability, next_state, reward, terminated in environment.P[state][action]:

                    # Calculate the expected value
                    v += action_probability * state_probability * (reward + discount_factor * V[next_state])

            # Calculate the absolute change of value function
            delta = max(delta, np.abs(V[state] - v))

            # Update value function
            V[state] = v

        evaluation_iterations += 1

        # Terminate if value change is insignificant
        if delta < theta:
            print(f'Policy evaluated in {evaluation_iterations} iterations.')
    return V



def policy_iteration(environment, discount_factor=1.0, max_iter=200):
    """
    Policy iteration algorithm to solve MDP.
    :param environment: Initialized OpenAI gym environment object.
    :param discount_factor: MPD discount factor. Float in range from 0 to 1.
    :param max_iter: Maximum number of iterations to prevent infinite loops.
    :return: tuple(policy, V), which consist of an optimal policy matrix and value function for each state.
    """
    # Start with a random policy
    #num states x num actions / num actions
    policy = np.ones([environment.nS, environment.nA]) / environment.nA

    # Initialize counter of evaluated policies
    evaluated_policies = 1

    convergence_list = []
    time_list = []
    iterations = [i for i in range(max_iter)]
    # max_step = 1000

    # Repeat until convergence or critical number of iterations reached
    for i in range(int(max_iter)):
        start_time = time.time()

        stable_policy = True

        # Evaluate current policy
        V = policy_evaluation(policy, environment, discount_factor=discount_factor)



        for state in range(environment.nS):



            current_action = np.argmax(policy[state])


            action_value = next_state(environment, state, V, discount_factor)

            # Select a better action
            best_action = np.argmax(action_value)

            # If action didn't change
            if current_action != best_action:
                stable_policy = True

            # Greedy policy update
            policy[state] = np.eye(environment.nA)[best_action]

        new_V = policy_evaluation(policy, environment, discount_factor=discount_factor)
        convergence_list.append(np.sum(np.fabs(new_V - V)))

        evaluated_policies += 1

        end_time = time.time()
        time_list.append(end_time - start_time)

        if stable_policy:
            print(f'Evaluated {evaluated_policies} policies.')

    return policy, V, time_list, convergence_list



def value_iteration(environment, discount_factor=0.5, theta=1e-9, max_iterations=200):
    """
    Value Iteration algorithm to solve MDP.
    :param environment: Initialized OpenAI environment object.
    :param theta: Stopping threshold. If the value of all states changes less than theta in one iteration - we are done.
    :param discount_factor: MDP discount factor.
    :param max_iterations: Maximum number of iterations that can be ever performed (to prevent infinite loops).
    :return: tuple (policy, V) which contains optimal policy and optimal value function.
    """


    V = np.zeros(environment.nS)

    convergence_list = []
    time_list = []
    iterations = [i for i in range(max_iterations)]

    for i in range(int(max_iterations)):
        start_time = time.time()

        delta = 0


        for state in range(environment.nS):


            action_value = next_state(environment, state, V, discount_factor)


            best_action_value = np.max(action_value)


            delta = max(delta, np.abs(V[state] - best_action_value))


            V[state] = best_action_value

        end_time = time.time()
        time_list.append(end_time - start_time)
        convergence_list.append(delta)

        # Check if we can stop
        if delta < theta:
            print(f'Value-iteration converged at iteration#{i}.')
            # break


    policy = np.zeros([environment.nS, environment.nA])

    for state in range(environment.nS):


        action_value = next_state(environment, state, V, discount_factor)



        best_action = np.argmax(action_value)



        policy[state, best_action] = 1.0

    return policy, V, time_list, convergence_list



def draw_curve(data_list, iters, problem, y_val):

    plt.close()

    # Create plot
    plt.figure()
    plt.title("PI and VI on parameter {} for {} (tyang358)".format(y_val, problem))
    plt.xlabel("iterations")
    plt.ylabel(y_val)


    # Draw lines
    plt.grid()
    plt.plot(iters, data_list[0], label="PI", color='black')
    plt.plot(iters, data_list[1], label="VI", color='orange')
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} plot for PI and VI on {}".format(y_val, problem))
    plt.savefig(image_path, dpi=100)
    return


if __name__ == "__main__":

    iters = 10000

    time_list = []
    convergence_list = []

    environment1 = gym.make('FrozenLake8x8-v0')

    policy, V, times, convergences = policy_iteration(environment1.env)
    time_list.append(times)
    convergence_list.append(convergences)



    # Apply best policy to the real environment
    wins, total_reward, average_reward = play_episodes(environment1, iters, policy)


    print("Average Rewards for policy iteration : ", average_reward)




    # Load a Frozen Lake environment
    environment2 = gym.make('FrozenLake8x8-v0')

    # Search for an optimal policy using policy iteration
    policy, V, times, convergences = value_iteration(environment2.env)
    time_list.append(times)
    convergence_list.append(convergences)

    # Apply best policy to the real environment
    wins, total_reward, average_reward = play_episodes(environment2, iters, policy)

    print("Average Rewards for Value iteration : ", average_reward)


    iterations = [i for i in range(200)]
    draw_curve(time_list, iterations, "Frozen Lake 8x8", "Time")
    draw_curve(convergence_list, iterations, "Frozen Lake 8x8", "Convergence")