from collections import defaultdict
import gymnasium as gym
import numpy as np


### AGENT BUILDING ###

class TaxiAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        
        self.env = env
        
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate

        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon

        self.epsilon_decay = epsilon_decay

        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        # Choose an action using epsilon-greedy strategy
        # It returns one of the six actions
        
        # Exploration (random action) with probability epsilon
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Exploitation (best known action) with probability 1-epsilon
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
    ):
        # Update Q-value based on experience (state, action, reward, next_state)

        # Current estimate: Q(state, action)
        current_q = self.q_values[obs][action]
        
        # The best we could do from the next state (zero if episode terminated, since no future possible rewards)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # Error: how wrong we were
        temporal_difference = target - current_q

        # Update estimate: move toward the target
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        #Track learning progress
        self.training_error.append(temporal_difference)


    def decay_epsilon(self):
        # Reduce exploration rate after each episode
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)




###  ANALYZING TRAINING RESULTS (performance metrics) ###

from matplotlib import pyplot as plt

def plot_results(env, agent, rolling_length=500):

    def get_moving_avgs(arr, window, convolution_mode):
        # Compute moving average to smooth noisy data
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode = convolution_mode
        ) / window

    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )

    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")


    # Episode lengths (how many actions)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )

    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")


    # Training error (how much we are still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )

    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()



### TESTING THE TRAINED AGENT ###

def test_agent(agent, env, num_episodes=1000):
    # Test agent performance without learning or exploration
    total_rewards = []

    #Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # pure exploitation
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
    
    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards)>0)
    average_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f} +/- {std_reward:.3f}")



### MAIN ###

if __name__ == "__main__":
    # --- Environment and Hyperparameters ---
    
    # Create the environment.
    # To run the non-deterministic (stochastic) version, uncomment the line below
    # and adjust the hyperparameters accordingly (e.g., more episodes, slower decay).
    env = gym.make("Taxi-v3")
    #env = gym.make("Taxi-v3", is_rainy=True, fickle_passenger=True)
    
    # Training hyperparameters
    learning_rate = 0.01  # Learning speed - higher = faster but less stable
    n_episodes = 50000  # Number of episodes to practice
    start_epsilon = 1     # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes /2)  # Reduce exploration over time
    final_epsilon = 0.1

    # Wrap the environment to record stats
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    # Creation of agent
    agent = TaxiAgent(
        env= env,
        learning_rate= learning_rate,
        initial_epsilon= start_epsilon,
        epsilon_decay= epsilon_decay,
        final_epsilon= final_epsilon,
    )

    # --- Training Loop ---
    print("--- Starting Training ---")
    from tqdm import tqdm  # Progress bar

    for episode in tqdm(range(n_episodes)):
        # Start a new episode
        obs, info = env.reset()
        done = False

        # Play one complete episode
        while not done:
            
            # Agent chooses action
            action = agent.get_action(obs)

            # Execute action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to the next state
            done = terminated or truncated
            obs = next_obs
        
        # Reduce exploration rate
        agent.decay_epsilon()

    print("--- Training Finished ---")

    # --- Analysis ---
    plot_results(env, agent)
    test_agent(agent, env)

    env.close()