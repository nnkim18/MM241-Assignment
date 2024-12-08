import time

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
        # s = time.time()
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod_idx, (prod_w, prod_h) in enumerate(product_sizes):
                # pos_x, pos_y, new_prod_size = self._find_position(stock, (prod_w, prod_h))
                # if None in [pos_x, pos_y, new_prod_size]:
                #     continue
                if stock_w >= prod_w and stock_h >= prod_h:
                    max_count = (stock_w // prod_w) * (stock_h // prod_h)
                    cutting_patterns.append((stock_idx, prod_idx, max_count))
                elif stock_w >= prod_h and stock_h >= prod_w:
                    max_count = (stock_w // prod_h) * (stock_h // prod_w)
                    cutting_patterns.append((stock_idx, prod_idx, max_count))
        # print(f"innit parameters {time.time() - s} - s")
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
        s = time.time()
        result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
        # print(f"Solving time {time.time() - s }")

        # Handle solution
        if result.success:
            # Find the most effective cutting pattern for the next action
            pattern_indexes = np.argsort(result.x)[::-1]
            for idx in pattern_indexes:
                stock_idx, prod_idx, _ = cutting_patterns[idx]
                prod_w, prod_h = product_sizes[prod_idx]

                # Find first available position in stock
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

