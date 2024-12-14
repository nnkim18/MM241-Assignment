import numpy as np
from policy import Policy
from scipy.optimize import linprog


class Policy2110513(Policy):
    def __init__(self, policy_id=1):
        super().__init__()
        self.list_prods = []
        self.list_stocks = []
        self.cuts = []
        self.decision_algorithm = policy_id
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            # clear all the variables
            self.list_prods = []
            self.list_stocks = []
            self.cuts = []
            self.firstIte = True
            self.decision_algorithm = policy_id
            # make the trigger for the first iteration
            pass
        elif policy_id == 2:
            # clear all the variables
            self.firstIte = True
            self.secIte = True  
            self.list_prods = None
            self.list_stocks = []
            self.cuts = []
            self.decision_algorithm = policy_id
            # make the trigger for the first iteration
            pass

    def get_action(self, observation, info):
        if self.decision_algorithm == 1:
            if self.firstIte or len(self.cuts) == 0:
                temp_prod = []
                for i, prod in enumerate(observation["products"]):
                    prod_size = prod["size"][:]
                    prod_quantity = prod["quantity"]

                    if prod_quantity > 0:
                        temp_prod.append({
                        "size": prod_size,
                        "quantity": prod_quantity
                    })

                self.list_prods = sorted(
                    temp_prod, 
                    key=lambda prod: (prod["size"][0], prod["size"][1]),  # Sort by width and length
                    reverse=True
                )

                list_stock = []
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    temp_stk = {
                        "stock_idx": i,
                        "stock_w": stock_w,
                        "stock_h": stock_h,
                        "arr": stock.copy()
                    }
                    list_stock.append(temp_stk)
                self.list_stocks = sorted(list_stock, key=lambda x: (x["stock_w"], x["stock_h"]), reverse=True)
                # print(self.list_stocks)
                self.cuts = self.greedy()
                # print(self.cuts)
                self.firstIte = False

            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0

            if (len(self.cuts) > 0):
                cut = self.cuts.pop(0)
                stock_idx = cut["stock"]
                prod_size = cut["size"]
                pos_x = cut["x"]
                pos_y = cut["y"]

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        
        elif self.decision_algorithm == 2:
            list_prods = observation["products"]
            stocks = observation["stocks"]

            # Arrange stocks in descending order of area   
            stock_areas = [self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1] for stock in stocks]
            sorted_stocks = sorted(enumerate(stocks), key=lambda x: stock_areas[x[0]], reverse=True)

            for stock_idx, stock in sorted_stocks:
                stock_w, stock_h = self._get_stock_size_(stock)
                
                # Filter out suitable products
                valid_prods = []
                for prod_idx, prod in enumerate(list_prods):
                    if prod["quantity"] > 0:
                        prod_w, prod_h = prod["size"]
                        if prod_w <= stock_w and prod_h <= stock_h:
                            valid_prods.append((prod_idx, prod))

                if not valid_prods:
                    continue

                # Make parameter for Knapsack problem
                c = [-prod["quantity"] for _, prod in valid_prods]  # Maximize product quantity
                A = []  
                b = []

                # Constraint about area
                area_constraint = [prod["size"][0] * prod["size"][1] for _, prod in valid_prods]
                A.append(area_constraint)
                b.append(stock_w * stock_h)

                # Constraint about width
                width_constraint = [prod["size"][0] for _, prod in valid_prods]
                A.append(width_constraint)
                b.append(stock_w)

                # Constraint about height
                height_constraint = [prod["size"][1] for _, prod in valid_prods]
                A.append(height_constraint)
                b.append(stock_h)

                # Constraint about maximum quantity of product can be selected
                bounds = [(0, prod["quantity"]) for _, prod in valid_prods]

                # Solve Knapsack problem by Linear Programming method
                res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

                if res.success:
                    solution = res.x
                    
                    # sort prod
                    selected_prods = []
                    for i, (prod_idx, prod) in enumerate(valid_prods):
                        if solution[i] > 0:
                            selected_prods.append((prod_idx, prod, solution[i]))

                    # sort prod
                    selected_prods.sort(key=lambda x: x[1]["size"][0] * x[1]["size"][1], reverse=True)
                    
                    # đặt vào stock
                    for prod_idx, prod, quantity in selected_prods:
                        prod_size = prod["size"]
                        for x in range(stock_w - prod_size[0] + 1):
                            for y in range(stock_h - prod_size[1] + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)
                                    }

            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            
    # Student code here
    # You can add more functions if needed
    
    ############################# ALGORITHM 01 #############################

    def greedy(self):
        # Run greedy algorithm
        # print (self.list_prods)
        print("Processing greedy algorithm")
        local_cuts = []
        for stock in self.list_stocks:
            stock_w, stock_h = self._get_stock_size_(stock["arr"])
            pos_x, pos_y = None, None
            prod_size = [0, 0]
            for y in range(stock_h):    
                for x in range(stock_w):
                    for prod in self.list_prods:
                        if prod["quantity"] > 0:
                            prod_size = prod["size"]

                            if (stock_w - x < prod_size[0] or stock_h - y < prod_size[1]):
                                continue   ## Skip if the product is too big for the stock
                            if self._can_place_(stock["arr"], (x, y), prod_size):
                                prod["quantity"] -= 1
                                pos_x, pos_y = x, y
                                stock["arr"][x:x + prod_size[0], y:y + prod_size[1]] = 3
                                local_cuts.append({
                                    "stock": stock["stock_idx"],
                                    "size": prod_size,
                                    "x": pos_x,
                                    "y": pos_y
                                })
                                break
        # print(local_cuts)
        if len(local_cuts) == 0:
            local_cuts.append({
                "stock": -1,
                "size": [0, 0],
                "x": 0,
                "y": 0
            })
        return local_cuts