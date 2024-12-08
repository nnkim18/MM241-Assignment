import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2033766 import *

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    # render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    # Reset the environment
    # observation, info = env.reset(seed=42)
    #
    # # Test GreedyPolicy
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #
    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1
    #
    # # Reset the environment
    observation, info = env.reset(seed=42)
    #
    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #
    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # Uncomment the following code to test your policy
    # # Reset the environment
    print("student")
    observation, info = env.reset(seed=42)
    action_size = len(observation["products"]) * len(observation["stocks"])
    ql_policy = QLearningPolicy(action_size, num_stocks=len(observation["stocks"]))
    ep = 0
    while ep < NUM_EPISODES:
        action = ql_policy.get_action(observation, info)
        next_observation, reward, terminated, truncated, info = env.step(action)
        current_state = ql_policy._extract_state(observation)
        next_state = ql_policy._extract_state(next_observation)

        # Update Q-table
        ql_policy.learn(
            state=current_state,
            action=action["stock_idx"],  # Assuming stock_idx represents the action
            reward=reward,
            next_state=next_state,
            done=terminated or truncated,
        )

        # Update observation
        observation = next_observation
        if terminated or truncated or info['filled_ratio'] == 1.0:
            print(f"Episode {ep + 1} finished. Info: {info}")
            observation, info = env.reset(seed=ep)
            ep += 1

env.close()
