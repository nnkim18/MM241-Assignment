from CuttingStockEnv import CuttingStockEnv
import numpy as np
from scipy.optimize import linprog, milp
from scipy.ndimage import label
import gym_cutting_stock
import gymnasium as gym
from heuristic_policy import CPolicy
from time import time, sleep

def simplex_method(stock_sizes, demand, item_sizes, stock):
    num_stocks = len(stock_sizes)		#so luong stock
    num_items = len(demand)		# so luong item yeu cau
    A = np.zeros((num_items + num_stocks, num_stocks))
    array_zero = np.zeros(num_stocks)
    b = np.array(demand)
    b = np.concatenate((b, array_zero), axis = 0)
    # Định nghĩa hàm dếm sô lượng từng item
    def num_item_per_stock(i, stock):
        item_mask = (stock == i)
        _, num_features = label(item_mask)
        return num_features
        
    for j in range(num_stocks):
        for i in range(num_items): 
            A[i, j] = num_item_per_stock(i + 1, stock[j])
 
    for i in range(num_stocks):
        A[i + num_items, i] = 1
    c = np.ones(num_stocks)  # Hàm mục tiêu
    # Giai
    result = linprog(c, integrality=1, A_ub=-A, b_ub=-b)

    # In ra kết quả
    if result.success:
        return result.x
    else:
        return None
def activity():
    pass

# env = gym.make(
#     "gym_cutting_stock/CuttingStock-v0",
#     render_mode="human",  # Comment this line to disable rendering
# )
env =CuttingStockEnv(
    #  render_mode="human"
                     )
NUM_EPISODES = 1

if __name__ == "__main__":
    # Reset the environment
    observation, info = env.reset(seed=42)
    # Khoi tao cac tham so cho simplex 
    demand = []
    item_sizes = []
    for item in observation["products"]:
        demand.append(item["quantity"])
        item_sizes.append(item["size"])

    sp_policy = CPolicy()
    ep = 0
    action_chain = dict()
    start_time = time()
    while ep < NUM_EPISODES:
        action = sp_policy.get_action(observation, info)
        # luu action vao
        if action['stock_idx'] not in action_chain:
            action_chain[action['stock_idx']] = [(action['size'], action['position'])]
        else:
            action_chain[action['stock_idx']].append((action['size'], action['position']))

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            sleep(2)
            # observation, info = env.reset(seed=ep)
            num_stocks = len(observation["stocks"])
            num_items = len(observation["products"])
            print("num stocks: ", num_stocks, "Num items:", num_items)
            print("-------------------------\n")

            stock_sizes = []

            for stock in observation["stocks"]:
                stock_width = np.sum(np.any(stock != -2, axis=1))
                stock_height = np.sum(np.any(stock != -2, axis=0))
                stock_sizes.append((stock_width, stock_height))
            print(f"Demand {demand}")
            print("Simplex Interger: ")
            solution = simplex_method(stock_sizes, demand, item_sizes, observation["stocks"])
            print(solution)
            # thuc hien simplex
            env.render_mode="human"
            observation, info = env.reset(seed = 42)
            if solution is not None:
                for index, x in enumerate(solution):
                    print(x)
                    if(x >= 0.5):
                        for j in range(int(x)):
                            for k in range(len(action_chain[index])):
                                action = {"stock_idx": index, "size": action_chain[index][k][0], "position": action_chain[index][k][1]}
                                print(f"size {action_chain[index][k][0]}, position: {action_chain[index][k][1]}")
                                observation, reward, terminated, truncated, info = env.step(action)
            else:
                print("No solution")
            end_time = time()
            print("Simplex method: ", info, "time: ", round(end_time-start_time,2), "second")
            #
            print("So luong mieng stock da cat", env.cutted_stocks)
            print("total stock", np.sum(env.cutted_stocks[env.cutted_stocks > 0]))
            ep += 1
    
env.close()