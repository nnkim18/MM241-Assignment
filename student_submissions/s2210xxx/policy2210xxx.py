import copy
import random

import numpy as np

from policy import Policy
from . import utils as ut


class Rectangle:
    def __init__(self, width, height, quantity):
        self.width = width
        self.height = height
        self.quantity = quantity

        
class Policy2033766_2033528(Policy):
    def __init__(self, policy_id=1):
        super().__init__()
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        if self.policy_id == 2:
            self.num_stocks = 100
            self.max_product_quantity = 15
            self.state_size = (self.num_stocks + 10) * self.max_product_quantity
            self.action_size = 3000
            self.learning_rate = 0.1
            self.discount_factor = 0.99
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
            self.q_table = np.zeros(1)
            ml_model = ut.MLModel()
            ml_model.load_model(self)

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.get_action_wang(observation, info)
        else:
            return self.get_action_qtable(observation, info)

    def get_action_wang(self, observation, info):
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        list_prods = observation["products"]

        rectangles = [Rectangle(prod["size"][0], prod["size"][1], prod["quantity"]) for prod in list_prods if
                      prod["quantity"] > 0]
        # Sắp xếp các hình chữ nhật theo diện tích giảm dần
        rectangles.sort(key=lambda x: x.width * x.height, reverse=True)

        stock_areas = [(i, self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[0]) for i, stock in
                       enumerate(observation["stocks"])]
        stock_areas.sort(key=lambda x: x[1], reverse=True)
        stocks = observation["stocks"]
        # Pick a product that has quality > 0
        # Loop through all stocks
        for idx, stock_area in stock_areas:
            solution = self.wang_algorithm(idx, stocks[idx], stock_area, rectangles)
            if solution is None:
                continue
            return solution

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    @staticmethod
    def calculate_waste(stock, position, prod_size):
        x, y = position
        width, height = prod_size
        stock[x: x + width, y: y + height] = 0
        return (stock == -1).sum()

    def wang_algorithm(self, stock_idx, selected_stock, stock_area, rectangles):
        stock_w, stock_h = self._get_stock_size_(selected_stock)
        rejection_parameter = 1
        all_solution = []
        for rectangle in rectangles:
            rep_stock = copy.deepcopy(selected_stock)
            prod_w, prod_h = rectangle.width, rectangle.height
            prod_size = (prod_w, prod_h)
            is_placed = False
            if stock_w >= prod_w and stock_h >= prod_h:
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(selected_stock, (x, y), prod_size):
                            waste = self.calculate_waste(rep_stock, (x, y), prod_size)
                            if waste < rejection_parameter * stock_w * stock_h:
                                is_placed = True
                                all_solution.append((waste, [int(x), int(y)], prod_size))
                                break
                    if is_placed:
                        break
            if is_placed:
                continue
            if stock_w >= prod_h and stock_h >= prod_w:
                pos_x, pos_y = None, None
                for x in range(stock_w - prod_h + 1):
                    for y in range(stock_h - prod_w + 1):
                        if self._can_place_(selected_stock, (x, y), prod_size[::-1]):
                            waste = self.calculate_waste(rep_stock, (x, y), prod_size)
                            if waste < rejection_parameter * stock_w * stock_h:
                                is_placed = True
                                all_solution.append((waste, [int(x), int(y)], prod_size))
                                break
                    if is_placed:
                        break

        if len(all_solution) == 0:
            return

        all_solution.sort(key=lambda x: x[0], reverse=True)
        best_solution = all_solution[0]
        (best_prod_w, best_prod_h) = best_solution[2]
        can_replace = False
        for x in range(stock_w - best_prod_w + 1):
            for y in range(stock_h - best_prod_h + 1):
                if self._can_place_(selected_stock, (x, y), best_solution[2]):
                    can_replace = True
        if not can_replace:
            return
        return {"stock_idx": stock_idx, "size": best_solution[2], "position": best_solution[1]}

    def get_action_qtable(self, observation, info):
        state = self._extract_state(observation)
        if random.random() < self.epsilon:
            action = self._random_valid_action(observation)
        else:
            action = np.argmax(self.q_table[state])
        if action is None:  # Handle case where no valid action is found
            print("Warning: No valid action available during get_action.")

        return self._action_to_env(action, observation)

    def _extract_state(self, observation):
        stock_fill_levels = [min(int(np.sum(stock) / stock.size * 10), 10) for stock in observation["stocks"]]
        product_quantities = [min(prod["quantity"], self.max_product_quantity) for prod in observation["products"]]
        state_vector = stock_fill_levels + product_quantities
        state_hash = sum(x * (10 ** i) for i, x in enumerate(state_vector))
        return state_hash % self.state_size

    def _action_to_env(self, action, observation):
        stock_idx = action // len(observation["products"])
        product_idx = action % len(observation["products"])
        prod_size = observation["products"][product_idx]["size"]
        if action is None:
            return {"stock_idx": stock_idx, "size": prod_size, "position": (0, 0)}
        stock = observation["stocks"][stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        for pos_x in range(stock_w - prod_size[0] + 1):
            for pos_y in range(stock_h - prod_size[1] + 1):
                if self._can_place_(stock, (pos_x, pos_y), prod_size):
                    return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
                # if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                #     return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (pos_x, pos_y)}
        return {"stock_idx": stock_idx, "size": prod_size, "position": (0, 0)}

    def _random_valid_action(self, observation):
        stocks_ratios = ut.compute_filled_ratio(observation["stocks"])
        sorted_stock_indices = np.argsort(stocks_ratios)[::-1]
        for stock_idx in sorted_stock_indices:
            stock = observation["stocks"][stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            for product_idx, product in enumerate(observation["products"]):
                if product["quantity"] == 0:
                    continue

                prod_size = product["size"]
                for pos_x in range(stock_w - prod_size[0] + 1):
                    for pos_y in range(stock_h - prod_size[1] + 1):
                        if self._can_place_(stock, (pos_x, pos_y), prod_size):
                            # Construct the action index and return
                            return stock_idx * len(observation["products"]) + product_idx
                        # if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                        #     # Construct the action index and return
                        #     return stock_idx * len(observation["products"]) + product_idx

        print("No valid actions available. Returning default action.")
        return None


class QLearningPolicy(Policy):
    """
    This class only used in the training state
    """
    def __init__(self, action_size, num_stocks=100, max_product_quantity=15):
        super().__init__()
        self.num_stocks = num_stocks
        self.max_product_quantity = max_product_quantity
        self.state_size = (num_stocks + 1) * max_product_quantity
        self.action_size = action_size
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.q_table = np.zeros((self.state_size, action_size))

    def get_action(self, observation, info):
        state = self._extract_state(observation)
        if random.random() < self.epsilon:
            action = self._random_valid_action(observation)
        else:
            action = np.argmax(self.q_table[state])
        if action is None:  # Handle case where no valid action is found
            print("Warning: No valid action available during get_action.")

        return self._action_to_env(action, observation)

    @staticmethod
    def compute_intermediate_reward(observation, next_observation):
        stocks_ratios_before = ut._compute_filled_ratio(observation["stocks"])
        stocks_ratios_after = ut._compute_filled_ratio(next_observation["stocks"])

        # Reward for increasing the filled ratio
        reward = 0
        for i in range(len(stocks_ratios_before)):
            if stocks_ratios_after[i] > stocks_ratios_before[i]:
                reward += (stocks_ratios_after[i] - stocks_ratios_before[i]) * 10

        return reward

    def learn(self, state, action, observation, next_observation, reward, next_state, done):
        intermediate_reward = self.compute_intermediate_reward(observation, next_observation)
        stocks_ratios_before = ut._compute_filled_ratio(observation["stocks"])
        stocks_ratios_after = ut._compute_filled_ratio(next_observation["stocks"])
        stock_idx = action // len(observation["products"])
        if stocks_ratios_after[stock_idx] > 0 and stocks_ratios_before[stock_idx] == 0:
            intermediate_reward -= 5  # Penalty for opening a new stock
        total_reward = reward + intermediate_reward

        # Update Q-value
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = total_reward + self.discount_factor * self.q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _extract_state(self, observation):
        stock_fill_levels = [min(int(np.sum(stock) / stock.size * 10), 10) for stock in observation["stocks"]]
        product_quantities = [min(prod["quantity"], self.max_product_quantity) for prod in observation["products"]]
        state_vector = stock_fill_levels + product_quantities
        state_hash = sum(x * (10 ** i) for i, x in enumerate(state_vector))
        return state_hash % self.state_size

    def _action_to_env(self, action, observation):
        stock_idx = action // len(observation["products"])
        product_idx = action % len(observation["products"])
        prod_size = observation["products"][product_idx]["size"]
        if action is None:
            return {"stock_idx": stock_idx, "size": prod_size, "position": (0, 0)}
        stock = observation["stocks"][stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        for pos_x in range(stock_w - prod_size[0] + 1):
            for pos_y in range(stock_h - prod_size[1] + 1):
                if self._can_place_(stock, (pos_x, pos_y), prod_size):
                    return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
                # if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                #     return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (pos_x, pos_y)}
        return {"stock_idx": stock_idx, "size": prod_size, "position": (0, 0)}

    def _random_valid_action(self, observation):
        stocks_ratios = ut._compute_filled_ratio(observation["stocks"])
        sorted_stock_indices = np.argsort(stocks_ratios)[::-1]
        for stock_idx in sorted_stock_indices:
            stock = observation["stocks"][stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            for product_idx, product in enumerate(observation["products"]):
                if product["quantity"] == 0:
                    continue

                prod_size = product["size"]
                for pos_x in range(stock_w - prod_size[0] + 1):
                    for pos_y in range(stock_h - prod_size[1] + 1):
                        if self._can_place_(stock, (pos_x, pos_y), prod_size):
                            # Construct the action index and return
                            return stock_idx * len(observation["products"]) + product_idx
                        # if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                        #     # Construct the action index and return
                        #     return stock_idx * len(observation["products"]) + product_idx

        print("No valid actions available. Returning default action.")
        return None