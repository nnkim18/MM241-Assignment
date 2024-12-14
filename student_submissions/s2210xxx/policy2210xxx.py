from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.policy = VeryGreedy()  # Khởi tạo BFPolicy
        elif policy_id == 2:
            self.policy = ColumnGeneration()

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1 and self.policy:
            return self.policy.get_action(observation, info)

        elif self.policy_id == 2:
           return self.policy.get_action(observation, info)
        return None


    # Student code here
    # You can add more functions if needed
class VeryGreedy(Policy):
    def __init__(self):
        self.sorted_stocks_index = np.array([])
        self.sorted_products = []
        self.counter = 0

    def sort_stock_product(self, stock_array, prod_array):
        stock_areas = [self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1] for stock in stock_array]
        self.sorted_stocks_index = np.argsort(stock_areas)[::-1]
        self.sorted_products = sorted(prod_array, key=lambda prod: prod['size'][0] * prod['size'][1], reverse=True)

    def get_action(self, observation, info):
        if self.counter == 0:
            self.sort_stock_product(observation['stocks'], observation['products'])
        self.counter += 1

        if self.sorted_products[-1]['quantity'] == 1:
            self.counter = 0

        selected_product_size = [0, 0]
        selected_stock_idx = -1
        position_x, position_y = 0, 0

        for product in self.sorted_products:
            if product["quantity"] > 0:
                selected_product_size = product["size"]

                for stock_idx in self.sorted_stocks_index:
                    stock_width, stock_height = self._get_stock_size_(observation['stocks'][stock_idx])
                    product_width, product_height = selected_product_size

                    if stock_width < product_width or stock_height < product_height:
                        continue

                    position_x, position_y = None, None
                    for x in range(stock_width - product_width + 1):
                        for y in range(stock_height - product_height + 1):
                            if self._can_place_(observation['stocks'][stock_idx], (x, y), selected_product_size):
                                position_x, position_y = x, y
                                break

                        if position_x is not None and position_y is not None:
                            break

                    if position_x is not None and position_y is not None:
                        selected_stock_idx = stock_idx
                        break

                if position_x is not None and position_y is not None:
                    break

        return {
            "stock_idx": selected_stock_idx,
            "stock_size": self._get_stock_size_(observation['stocks'][selected_stock_idx]),
            "size": selected_product_size,
            "position": (position_x, position_y)
        }








