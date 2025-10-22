import argparse
import random
from collections import deque, namedtuple
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path # Use pathlib for robust path handling

# Use a GPU if available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the structure for storing experiences
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'terminated'))

class ReplayBuffer:
    """A fixed-size buffer to store experience tuples."""

    def __init__(self, capacity):
        """
        Initialize a ReplayBuffer.
        
        Args:
            capacity (int): The maximum size of the buffer.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """Deep Q-Network model."""

    def __init__(self, n_observations, n_actions):
        """
        Initialize the neural network.

        Args:
            n_observations (int): The size of the state space (500 for Taxi).
            n_actions (int): The number of possible actions (6 for Taxi).
        """
        super(DQN, self).__init__()
        # Changed the hidden layers to 64 and 32 units
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        """Defines the forward pass of the network."""
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    """The agent that interacts with and learns from the environment."""

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
    ):
        self.env = env
        self.n_actions = env.action_space.n
        self.n_observations = env.observation_space.n
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.batch_size = batch_size

        # Initialize the Policy network
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # Define the loss function here (using = for assignment)
        self.criterion = nn.SmoothL1Loss() # Huber loss
        
        self.training_steps_done = 0
        self.training_loss = []

    def get_action(self, state: int) -> int:
        """Choose an action using an epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                # One-hot encode the state
                state_tensor = torch.zeros(self.n_observations, device=device)
                state_tensor[state] = 1.0
                # Get Q-values from the policy network
                q_values = self.policy_net(state_tensor.unsqueeze(0))
                # Choose the action with the highest Q-value
                return q_values.max(1)[1].item()

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def learn(self):
        """Update the policy network using a batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # --- Prepare the batch for the network ---
        
        # One-hot encode states and next_states
        state_batch = torch.zeros(self.batch_size, self.n_observations, device=device)
        next_state_batch = torch.zeros(self.batch_size, self.n_observations, device=device)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        
        for i, s in enumerate(batch.state):
            state_batch[i, s] = 1.0
        
        next_state_indices = [s for s in batch.next_state if s is not None]
        if len(next_state_indices) > 0:
            next_state_batch[non_final_mask] = torch.eye(self.n_observations, device=device)[next_state_indices]

        action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=device)
        
        # --- Compute Q-values ---

        # Q(s_t, a) - The Q-values for the actions that were actually taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # V(s_{t+1}) - The maximum Q-value for the next state, calculated by the SAME policy network
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.policy_net(next_state_batch[non_final_mask]).max(1)[0]
        
        # Expected Q-values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # --- Compute Loss and update the policy network ---
        
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.training_loss.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.training_steps_done += 1

def plot_results(env: gym.Env, agent: DQNAgent, rolling_length: int = 100, filename: str = None):
    """Plot training results and optionally save to a file."""
    # Changed to 3 subplots and adjusted figure size
    fig, axs = plt.subplots(ncols=3, figsize=(18, 5))
    
    def get_moving_avgs(arr, window):
        return np.convolve(np.array(arr), np.ones(window), mode="valid") / window

    # Plot 1: Episode Rewards
    axs[0].set_title("Episode Rewards")
    reward_moving_average = get_moving_avgs(env.return_queue, rolling_length)
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel(f"Average Reward (over {rolling_length} episodes)")

    # Plot 2: Episode Lengths (NEW PLOT)
    axs[1].set_title("Episode Lengths")
    length_moving_average = get_moving_avgs(env.length_queue, rolling_length)
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel(f"Average Length (over {rolling_length} episodes)")

    # Plot 3: Training Loss (moved to the third position)
    axs[2].set_title("Training Loss")
    loss_moving_average = get_moving_avgs(agent.training_loss, rolling_length * 10) # Smoother loss
    axs[2].plot(range(len(loss_moving_average)), loss_moving_average)
    axs[2].set_xlabel("Training Step")
    axs[2].set_ylabel(f"Average Loss (over {rolling_length*10} steps)")
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"\nPlot saved to {filename}")
    
    plt.show()
    plt.close(fig) # Close the figure to free memory

def test_agent(agent: DQNAgent, env: gym.Env, num_episodes: int = 1000, filename: str = None):
    """Test the agent's performance and optionally save results to a file."""
    total_rewards = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Turn off exploration

    print(f"\n--- Testing over {num_episodes} episodes ---")
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    
    agent.epsilon = old_epsilon # Restore epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {avg_reward:.3f} +/- {std_reward:.3f}")
    
    # Save results to the specified file
    if filename:
        with open(filename, 'a') as f:
            f.write("\n--- Test Results ---\n")
            f.write(f"Tested over {num_episodes} episodes.\n")
            f.write(f"Win Rate: {win_rate:.1%}\n")
            f.write(f"Average Reward: {avg_reward:.3f} +/- {std_reward:.3f}\n")
        print(f"Test results appended to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Taxi-v3.")
    parser.add_argument(
        "--variant", 
        type=str, 
        default="deterministic", 
        choices=["deterministic", "stochastic"],
        help="Choose the environment variant."
    )
    args = parser.parse_args()

    configs = {
        "deterministic": {
            "n_episodes": 5000,
            "learning_rate": 0.0005,
            "start_epsilon": 1.0,
            "final_epsilon": 0.01,
            "env_kwargs": {"is_rainy": False, "fickle_passenger": False}
        },
        "stochastic": {
            "n_episodes": 2500,
            "learning_rate": 1e-4,
            "start_epsilon": 1.0,
            "final_epsilon": 0.05,
            "env_kwargs": {"is_rainy": True, "fickle_passenger": True}
        }
    }
    
    config = configs[args.variant]
    n_episodes = config["n_episodes"]
    epsilon_decay = config["start_epsilon"] / (n_episodes * 0.8)

    # --- File Setup for Logging ---
    # Define the main results directory
    main_results_dir = Path("results")

    # Create a unique name and subdirectory for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"DQN_{args.variant}_{timestamp}"
    run_dir = main_results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths inside the new unique subdirectory
    plot_filename = run_dir / "plot.png"
    results_filename = run_dir / "results.txt"

    # --- Agent and Environment Setup ---
    env = gym.make("Taxi-v3", **config["env_kwargs"])
    # Use the older 'buffer_length' argument for compatibility
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    
    agent = DQNAgent(
        env=env,
        learning_rate=config["learning_rate"],
        initial_epsilon=config["start_epsilon"],
        epsilon_decay=epsilon_decay,
        final_epsilon=config["final_epsilon"],
    )

    # --- Log Hyperparameters to File ---
    with open(results_filename, 'w') as f:
        f.write(f"--- Hyperparameters for run on {timestamp} ---\n")
        f.write(f"Variant: {args.variant}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Learning Rate: {agent.lr}\n")
        f.write(f"Start Epsilon: {config['start_epsilon']}\n")
        f.write(f"Final Epsilon: {agent.final_epsilon}\n")
        f.write(f"Epsilon Decay: {agent.epsilon_decay}\n")
        f.write(f"Discount Factor (Gamma): {agent.gamma}\n")
        f.write(f"Replay Buffer Size: {len(agent.replay_buffer.memory)}\n")
        f.write(f"Batch Size: {agent.batch_size}\n")
        f.write(f"Network Architecture: {agent.policy_net}\n")


    print(f"--- Running {args.variant.capitalize()} Variant on {device} ---")
    print(f"Results will be saved to '{run_dir}'")
    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # If terminated, next_state is None for our buffer logic
            real_next_state = next_state if not terminated else None
            
            agent.replay_buffer.push(state, action, real_next_state, reward, terminated)
            
            state = next_state
            
            agent.learn()
        
        agent.decay_epsilon()

    print("--- Training Finished ---")

    plot_results(env, agent, filename=plot_filename)
    test_agent(agent, env, filename=results_filename)

    env.close()

