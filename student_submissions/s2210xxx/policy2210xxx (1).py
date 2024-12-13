import random
from abc import abstractmethod
import numpy as np
from policy import Policy

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        list_prods = observation["products"]
        sorted_prods = sorted(list_prods, key=lambda prod: prod["size"][1], reverse=True)

        stocks = observation["stocks"]

        action = {"size": None, "position": None, "stock_idx": None}  

        last_used_stock_idx = -1
        current_stock = None

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                placed = False

                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w >= prod_w and stock_h >= prod_h:
                        for _ in range(100):  
                            pos_x = random.randint(0, stock_w - prod_w)
                            pos_y = random.randint(0, stock_h - prod_h)

                            if self._can_place_(stock, (pos_x, pos_y), prod_size):
                                action["size"] = prod_size 
                                action["position"] = (pos_x, pos_y) 
                                action["stock_idx"] = stock_idx  
                                placed = True
                                break
                    if placed:
                        break

        if not placed:
            action["size"] = None
            action["position"] = None
            action["stock_idx"] = None

        return action