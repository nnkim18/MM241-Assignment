import time
import random

from scipy.optimize import linprog
import numpy as np
from policy import Policy


class Policy2033766(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        """
        Determine the cutting action using LP to minimize waste.
        """
        stocks = observation["stocks"]
        products = observation["products"]

        # Extract dimensions and demand for products
        product_sizes = [prod["size"] for prod in products]
        product_demands = [prod["quantity"] for prod in products]

        # Pre-generate feasible cutting patterns
        cutting_patterns = []
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod_idx, (prod_w, prod_h) in enumerate(product_sizes):
                if stock_w >= prod_w and stock_h >= prod_h:
                    max_count = (stock_w // prod_w) * (stock_h // prod_h)
                    cutting_patterns.append((stock_idx, prod_idx, max_count))
                elif stock_w >= prod_h and stock_h >= prod_w:
                    max_count = (stock_w // prod_h) * (stock_h // prod_w)
                    cutting_patterns.append((stock_idx, prod_idx, max_count))
        # Prepare LP components
        num_patterns = len(cutting_patterns)
        num_products = len(products)

        c = np.ones(num_patterns)  # Minimize the number of stocks used
        A = np.zeros((num_products, num_patterns))
        b = np.array(product_demands)

        for j, (stock_idx, prod_idx, max_count) in enumerate(cutting_patterns):
            A[prod_idx, j] = max_count  # Add pattern feasibility

        # Solve LP problem
        bounds = [(0, None) for _ in range(num_patterns)]  # Relax to continuous
        result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")

        # Handle solution
        if result.success:
            # Find the most effective cutting pattern for the next action
            pattern_indexes = np.argsort(result.x)[::-1]
            for idx in pattern_indexes:
                stock_idx, prod_idx, _ = cutting_patterns[idx]
                prod_w, prod_h = product_sizes[prod_idx]
                stock = stocks[stock_idx]
                pos_x, pos_y, new_pro_size = self._find_position(stock, (prod_w, prod_h))
                if None not in [pos_x, pos_y]:
                    prod_w, prod_h = new_pro_size
                    return {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (pos_x, pos_y)}
        else:
            print("LP failed to find a solution.")
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

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


class QLearningPolicy(Policy):
    def __init__(self, action_size, num_stocks=100, stock_usage_levels=10, max_product_quantity=10):
        self.num_stocks = num_stocks
        self.stock_usage_levels = stock_usage_levels
        self.max_product_quantity = max_product_quantity

        self.state_size = num_stocks * stock_usage_levels * max_product_quantity  # Adjust based on observation
        self.action_size = action_size
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Initialize Q-table
        self.q_table = np.zeros((self.state_size, action_size))

    def get_action(self, observation, info):
        # Extract the current state
        state = self._extract_state(observation)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)  # Explore
        else:
            action = np.argmax(self.q_table[state])  # Exploit
        return self._action_to_env(action, observation)

    def learn(self, state, action, reward, next_state, done):
        # Update Q-value using the Q-learning formula
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _extract_state(self, observation):
        # Encode stocks into discrete levels of usage
        stock_levels = [
            int(np.sum(stock) / (stock.shape[0] * stock.shape[1]) * self.stock_usage_levels)
            for stock in observation["stocks"]
        ]

        # Encode products into remaining quantities
        product_quantities = [
            min(prod["quantity"], self.max_product_quantity)
            for prod in observation["products"]
        ]

        # Combine stock levels and product quantities
        state_vector = stock_levels + product_quantities

        # Hash the combined state to a discrete value
        state_hash = hash(tuple(state_vector)) % self.state_size
        return state_hash

    def _action_to_env(self, action, observation):
        # Map action index to stock and product placement
        stock_idx = action // len(observation["products"])
        product_idx = action % len(observation["products"])
        prod_size = observation["products"][product_idx]["size"]
        pos_x, pos_y = 0, 0  # Placeholder logic for position selection
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}