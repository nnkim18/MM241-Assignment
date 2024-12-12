import copy
import os.path
from PIL import Image

import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2033766 import *
import student_submissions.s2210xxx.utils as ut

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="rgb_array",  # Comment this line to disable rendering
)
NUM_EPISODES = 10

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
    # observation, info = env.reset(seed=42)
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
    ml_model = ut.MLModel()
    observation, info = env.reset(seed=42)
    number_of_product = ut.calculate_remaining_products(observation)
    # if number_of_product < len(observation["stocks"]):
    #     number_of_product *= 3
    print(number_of_product)
    action_size = number_of_product * len(observation["stocks"])
    num_stocks = len(observation["stocks"])
    max_quantity = max(product["quantity"] for product in observation["products"])
    ql_policy = QLearningPolicy(action_size, num_stocks=num_stocks, max_product_quantity=max_quantity)
    os.makedirs("episode_images", exist_ok=True)

    # Load the model if available
    ml_model.load_model(ql_policy)

    ep = 0
    while ep < NUM_EPISODES:
        action_size = ut.calculate_remaining_products(observation) * len(observation["stocks"])
        if action_size > ql_policy.action_size:
            ql_policy.action_size = action_size
        current_state = ql_policy._extract_state(observation)
        action = ql_policy.get_action(observation, info)
        previous_observation = copy.deepcopy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        next_state = ql_policy._extract_state(observation)
        # Update Q-table
        ql_policy.learn(
            state=current_state,
            action=action["stock_idx"],  # Assuming stock_idx represents the action
            observation=previous_observation,
            next_observation=observation,
            reward=reward,
            next_state=next_state,
            done=terminated or truncated,
        )
        if terminated or truncated:
            print(f"Episode {ep + 1} finished. Info: {info}")
            print(f"Epsilon {ql_policy.epsilon}")
            number_of_used_stock = ut.get_number_of_used_stocks(observation['stocks'])
            frame = env.render()
            if frame is not None:  # Ensure frame is returned
                img = Image.fromarray(frame)
                img.save(f"episode_images/episode_{ep + 1}.png")
            print(f"Number of used stock {number_of_used_stock}")
            observation, info = env.reset(seed=ep)
            number_of_product = ut.calculate_remaining_products(observation)
            print(f"Number of products of {ep + 1}: {number_of_product}")
            ep += 1

    # Save the trained model
    ml_model.save_model(ql_policy)

env.close()
