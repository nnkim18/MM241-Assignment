from policy import Policy
import numpy as np
import random


class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy = None
        if policy_id == 1:
            self.policy = BranchAndBoundPolicy()
        elif policy_id == 2:
            self.policy = DynamicProgrammingPolicy()

    def get_action(self, observation, info):
        # Student code here
        if self.policy is not None:
            return self.policy.get_action(observation, info)
        else:
            raise NotImplementedError("Policy is not implemented yet.")


class BranchAndBoundPolicy(Policy):
    def __init__(self):
        self.initialize = True
        self.min_wasted_space = float("inf")
        self.best_solution = []
        self.current_solution = []
        self.cnt = 0
        self.stock_idx = 0

    def get_action(self, observation, info):
        list_prods = observation["products"]
        list_stocks = observation["stocks"]
        n = len(list_stocks)

        if self.stock_idx >= n:
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

        if self.initialize:
            self.check = [0 for _ in range(n)]

        stock = list_stocks[self.stock_idx]
        list_prods = sorted(list_prods, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)

        if self.initialize:
            self.branch_and_bound(list_prods, stock, 0, self.stock_idx)

        self.initialize = False
        self.current_solution = []

        if len(self.best_solution) == 0:
            temp = {"stock_idx": self.stock_idx, "size": (0, 0), "position": (0, 0)}
            self.stock_idx += 1
            self.cnt = 0
            self.min_wasted_space = float("inf")
            self.initialize = True
            return temp

        if self.cnt < len(self.best_solution):
            temp = self.best_solution[self.cnt]
            self.cnt += 1
            if self.cnt == len(self.best_solution):
                self.cnt = 0
                self.best_solution = []
                self.stock_idx += 1
                self.min_wasted_space = float("inf")
                self.initialize = True
            return temp

    def branch_and_bound(self, list_prods, stock, idx, stock_idx):
        if idx == len(list_prods):
            wasted_space = self.calculate_wasted_space(stock)
            if wasted_space < self.min_wasted_space:
                self.min_wasted_space = wasted_space
                self.best_solution = self.current_solution[:]
            return

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size

                if stock_w < prod_w or stock_h < prod_h:
                    continue

                for i in range(stock_w - prod_w + 1):
                    for j in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (i, j), prod_size):
                            self.place_product(stock, (i, j), prod_size)
                            prod["quantity"] -= 1
                            self.current_solution.append({"stock_idx": stock_idx, "size": prod_size, "position": (i, j)})

                            self.branch_and_bound(list_prods, stock, idx + 1, stock_idx)

                            self.current_solution.pop()
                            self.remove_product(stock, (i, j), prod_size)
                            prod["quantity"] += 1

    def calculate_wasted_space(self, stock):
        return np.sum(stock == -1)

    def place_product(self, stock, position, prod_size):
        x, y = position
        w, h = prod_size
        stock[x : x + w, y : y + h] = 1

    def remove_product(self, stock, position, prod_size):
        x, y = position
        w, h = prod_size
        stock[x : x + w, y : y + h] = -1

    def _get_stock_size_(self, stock):
        return stock.shape

    def _can_place_(self, stock, position, prod_size):
        x, y = position
        w, h = prod_size
        if x + w > stock.shape[0] or y + h > stock.shape[1] or x < 0 or y < 0:
            return False
        return np.all(stock[x : x + w, y : y + h] == -1)

class DynamicProgrammingPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        stocks = observation['stocks']  
        products = observation['products']  
        
        sorted_products = sorted(
            [p for p in products if p['quantity'] > 0],
            key=lambda x: x['size'][0] * x['size'][1],
            reverse=True
        )
        
        for product in sorted_products:
            product_height, product_width = product['size']
            demand = product['quantity']
            
            for stock_idx, stock in enumerate(stocks):
                stock_height, stock_width = stock.shape
                
                dp = np.zeros((stock_height + 1, stock_width + 1), dtype=bool)
                dp[0][0] = True  
                
                for i in range(stock_height):
                    for j in range(stock_width):
                        if stock[i, j] == -1:  
                            if i > 0:
                                dp[i][j] = dp[i][j] or dp[i - 1][j]
                            if j > 0:
                                dp[i][j] = dp[i][j] or dp[i][j - 1]
                
                for rotation in [(product_height, product_width), (product_width, product_height)]:
                    rotated_height, rotated_width = rotation
                    
                    for x in range(stock_height - rotated_height + 1):
                        for y in range(stock_width - rotated_width + 1):
                            if self._can_place(stock, (x, y), (rotated_height, rotated_width)):
                                self._place_product(stock, (x, y), (rotated_height, rotated_width))
                                demand -= 1
                                action = {
                                    "stock_idx": stock_idx,
                                    "size": (rotated_height, rotated_width),
                                    "position": (x, y),
                                }
                                return action
        
        
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def _can_place(self, stock, position, size):
        
        x, y = position
        height, width = size
        stock_height, stock_width = stock.shape
        
        if x + height > stock_height or y + width > stock_width:
            return False
        
        for i in range(height):
            for j in range(width):
                if stock[x + i, y + j] != -1:
                    return False
        return True

    def _place_product(self, stock, position, size):
        x, y = position
        height, width = size
        
        for i in range(height):
            for j in range(width):
                stock[x + i, y + j] = 1

