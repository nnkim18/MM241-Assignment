import time
import random

from scipy.optimize import linprog
import numpy as np
from policy import Policy
from . import utils as ut


class Policy2033766(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        """
        Determine the cutting action for each product, iteratively reducing demand.
        """
        stocks = observation["stocks"]
        products = observation["products"]

        # Extract dimensions and demand for products
        product_sizes = [prod["size"] for prod in products]
        product_demands = [prod["quantity"] for prod in products]

        # Iterate through products
        for prod_idx, demand in enumerate(product_demands):
            if demand <= 0:
                continue  # Skip fulfilled demands

            best_action = self._allocate_product(stocks, product_sizes[prod_idx])
            if best_action:
                return best_action

        # If no valid action is found, return a failure action
        print("No valid action found.")
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def _allocate_product(self, stocks, product_size):
        """
        Find the best stock and position for a single product using LP.
        """
        cutting_patterns = []
        stocks_ratios = self._compute_filled_ratio(stocks)
        best_stock_idxs = np.argsort(stocks_ratios)
        print(f"filled_ratio {stocks_ratios[best_stock_idxs[0]]}")
        for stock_idx in best_stock_idxs[0: best_stock_idxs.shape[0]//2]:
            stock = stocks[stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)

            # Check feasible positions for the product
            for pos_x in range(stock_w - product_size[0] + 1):
                for pos_y in range(stock_h - product_size[1] + 1):
                    if self._can_place_(stock, (pos_x, pos_y), product_size):
                        cutting_patterns.append((stock_idx, pos_x, pos_y, product_size))
                    elif self._can_place_(stock, (pos_x, pos_y), product_size[::-1]):
                        cutting_patterns.append((stock_idx, pos_x, pos_y, product_size[::-1]))

        # No valid cutting patterns for this product
        if not cutting_patterns:
            return None

        # Prepare LP to choose the best pattern
        num_patterns = len(cutting_patterns)
        c = np.ones(num_patterns)  # Minimize placements
        A = np.ones((1, num_patterns))  # All patterns satisfy one unit of demand
        b = np.array([1])  # Allocate one unit at a time

        # Solve LP
        bounds = [(0, 1) for _ in range(num_patterns)]  # Binary bounds for patterns
        result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")

        if result.success:
            # Select the best pattern
            best_pattern_idx = np.argmax(result.x)
            stock_idx, pos_x, pos_y, chosen_size = cutting_patterns[best_pattern_idx]

            # Return the chosen action
            return {
                "stock_idx": stock_idx,
                "size": chosen_size,
                "position": (pos_x, pos_y),
            }

        return None

    def _find_position(self, stock, prod_size):
        """
        Find the first position to place the product in the stock, considering rotation.
        """
        stock_w, stock_h = self._get_stock_size_(stock)

        for x in range(stock_w - prod_size[0] + 1):
            for y in range(stock_h - prod_size[1] + 1):
                for rotated_size in [prod_size, prod_size[::-1]]:
                    if self._can_place_(stock, (x, y), rotated_size):
                        return x, y, rotated_size
        return None, None, None

    @staticmethod
    def _compute_filled_ratio(stocks):
        """
        Compute the current filled ratio of a stock.
        """
        usable_area = np.sum(np.array(stocks) == -1, axis=(1, -1))
        filled_stock_idxs = usable_area == 0
        un_filled_stock_idxs = usable_area != 0
        filled_area = np.sum(np.array(stocks) > -1, axis=(1, -1))
        filled_ratios = np.zeros(len(stocks))
        filled_ratios[filled_stock_idxs] = 1.0
        filled_ratios[un_filled_stock_idxs] = filled_area[un_filled_stock_idxs] / usable_area[un_filled_stock_idxs]
        return filled_ratios


class QLearningPolicy(Policy):
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
