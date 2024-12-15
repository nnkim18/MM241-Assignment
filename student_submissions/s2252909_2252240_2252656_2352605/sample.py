from scipy.optimize import linprog
import numpy as np
from policy import Policy, RandomPolicy

class Policy2210xxx(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        # Step 1: Solve ILP for initial allocation
        products = observation["products"]
        stocks = observation["stocks"]

        # Ensure stocks are dictionaries with a "placed" key
        stocks = [{"grid": stock, "placed": []} if not isinstance(stock, dict) else stock for stock in stocks]

        allocation = self.solve_ilp_allocation(products, stocks)

        if allocation is None:
            # Fall back to random policy if ILP fails
            print("ILP Allocation failed, falling back to random policy.")
            return RandomPolicy().get_action(observation, info)

        # Step 2: Use Greedy method for placement
        for j, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock["grid"])

            for i, product in enumerate(products):
                if allocation[i, j] == 1:  # Product i is allocated to stock j
                    prod_w, prod_h = product["size"]

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                # Place the product and return the action
                                self._place_(stock, (x, y), (prod_w, prod_h))
                                return {
                                    "stock_idx": j,
                                    "size": (prod_w, prod_h),
                                    "position": (x, y),
                                }

        # If no placement is possible, return dummy action
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def solve_ilp_allocation(self, products, stocks):
        num_products = len(products)
        num_stocks = len(stocks)

        # Decision variables
        num_vars = num_products * num_stocks
        c = [1] * num_vars  # Minimize stock usage

        # Constraints
        A_eq = []
        b_eq = []

        # Constraint 1: Each product must be placed in exactly one stock
        for i in range(num_products):
            row = [1 if j // num_stocks == i else 0 for j in range(num_vars)]
            A_eq.append(row)
            b_eq.append(1)

        # Constraint 2: Products must fit in stock dimensions
        A_ub = []
        b_ub = []
        for j, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock["grid"])
            for i, product in enumerate(products):
                prod_w, prod_h = product["size"]
                if prod_w <= stock_w and prod_h <= stock_h:
                    # Valid combination; ensure product fits within stock dimensions
                    valid_row = [1 if k == i * num_stocks + j else 0 for k in range(num_vars)]
                    A_ub.append(valid_row)
                    b_ub.append(1)
                else:
                    # Invalid combination; set x[i, j] = 0
                    invalid_row = [1 if k == i * num_stocks + j else 0 for k in range(num_vars)]
                    A_ub.append(invalid_row)
                    b_ub.append(0)

        # Bounds
        bounds = [(0, 1) for _ in range(num_vars)]

        # Solve ILP
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if result.success:
            return np.round(result.x).reshape((num_products, num_stocks))
        else:
            return None

    def _get_stock_size_(self, stock):
        st_w = np.sum(np.any(stock != -2, axis=1))
        st_h = np.sum(np.any(stock != -2, axis=0))
        return st_w, st_h

    def _can_place_(self, stock, position, size):
        x, y = position
        w, h = size
        stock_w, stock_h = self._get_stock_size_(stock["grid"])

        # Check bounds
        if x + w > stock_w or y + h > stock_h:
            return False

        # Check for overlap with already placed products
        for placed in stock["placed"]:
            px, py, pw, ph = placed["position"][0], placed["position"][1], placed["size"][0], placed["size"][1]
            if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                return False

        return True

    def _place_(self, stock, position, size):
        if "placed" not in stock:
            stock["placed"] = []
        stock["placed"].append({"position": position, "size": size})
