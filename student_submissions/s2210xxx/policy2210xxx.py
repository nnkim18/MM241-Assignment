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


class ColumnGeneration(Policy):
    def __init__(self):
        self.current_patterns = []
        self.dual_values = []
        self.demands = None
        self.stock_size = None
        self.num_products = 0
        self.epsilon = 1e-6
        self.action_queue = []
        self.master_solution = None
        self.stock_idx = -1
        self.check_100 = False
        self.last_check = False
        self.stock_placed = []

    def _get_stock_size_(self, stock):
        stock = np.atleast_2d(stock)
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, size):
        xi, yi = position
        w, h = size
        stock_w, stock_h = stock.shape
        if xi + w > stock_w or yi + h > stock_h:
            return False
        return np.all(stock[xi:xi+w, yi:yi+h] == -1)

    def _solve_master_problem(self, demand_vector):
        num_patterns = len(self.current_patterns)
        cvec = np.ones(num_patterns)
        A_eq = np.zeros((self.num_products, num_patterns))

        for j, pattern in enumerate(self.current_patterns):
            A_eq[:, j] = pattern['counts']

        result = linprog(cvec, A_eq=A_eq, b_eq=demand_vector, method='highs', bounds=(0, None))

        if result.success:
            self.dual_values = result["eqlin"]["marginals"]
            self.master_solution = result["x"]
        else:
            raise ValueError("Master problem did not converge.")

    def _solve_subproblem(self):
        unit_values = [
            (self.dual_values[i] / (product["size"][0] * product["size"][1]) if product["size"][0] * product["size"][1] > 0 else 0, i)
            for i, product in enumerate(self.demands)
        ]
        unit_values.sort(key=lambda x: x[0], reverse=True)

        stock = np.full(self.stock_size, fill_value=-1, dtype=int)
        counts = np.zeros(self.num_products, dtype=int)
        placements = []

        for unit_value, i in unit_values:
            product = self.demands[i]
            w_i, h_i = product["size"]
            quantity_i = product["quantity"]
            orientations = [(w_i, h_i), (h_i, w_i)] if w_i != h_i else [(w_i, h_i)]

            for _ in range(quantity_i - counts[i]):
                placed = False
                for w, h in orientations:
                    for xi in range(self.stock_size[0] - w + 1):
                        for yi in range(self.stock_size[1] - h + 1):
                            if self._can_place_(stock, (xi, yi), (w, h)):
                                stock[xi:xi+w, yi:yi+h] = i
                                placements.append((i, xi, yi, w, h))
                                counts[i] += 1
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break
                if not placed:
                    break

        reduced_cost = 1 - np.dot(self.dual_values, counts)
        return {
            'counts': counts,
            'placements': placements,
            'unit_values': unit_values
        } if reduced_cost < -self.epsilon else None

    def get_action(self, observation, info):
        while True:
            if self.action_queue:
                return self.action_queue.pop(0)

            # Adjust stock index
            self._adjust_stock_idx()

            self.current_patterns.clear()
            stock_list = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1], reverse=False)
            self.demands = observation["products"]
            self.num_products = len(self.demands)
            demands_vector = np.array([product["quantity"] for product in self.demands])

            # Initialize patterns if empty
            if not self.current_patterns:
                self._initialize_patterns()

            # Kiểm tra xem stock_list có phần tử không
            if len(stock_list) == 0:
                return {'stock_idx': -1, 'size': [0, 0], 'position': [0, 0]}

            # Giới hạn stock_idx trong phạm vi hợp lệ
            self.stock_idx = min(self.stock_idx, len(stock_list) - 1)

            # Kiểm tra nếu stock_idx hợp lệ
            if self.stock_idx < 0 or self.stock_idx >= len(stock_list):
                return {'stock_idx': -1, 'size': [0, 0], 'position': [0, 0]}

            self.stock_size = self._get_stock_size_(stock_list[self.stock_idx][1])

            stockidxreturn = stock_list[self.stock_idx][0]

            while True:
                self._solve_master_problem(demands_vector)
                new_pattern = self._solve_subproblem()
                if new_pattern is None:
                    break
                self.current_patterns.append(new_pattern)

            x = self.master_solution
            if x is None or x.size == 0:
                return {'stock_idx': -1, 'size': [0, 0], 'position': [0, 0]}

            Area_array = np.array([sum(self.demands[j]["size"][0] * self.demands[j]["size"][1] * self.current_patterns[i]["counts"][j] for j in range(self.num_products)) for i in range(len(self.current_patterns))])

            pattern_idx = np.argmax(Area_array)
            a = Area_array[pattern_idx] / (self.stock_size[0] * self.stock_size[1])
            selected_pattern = self.current_patterns[pattern_idx]
            placements = selected_pattern["placements"]
            self.action_queue.clear()

            if not self._should_generate_action(a, placements):
                continue

            self.stock_placed.append(self.stock_idx)
            for placement in placements:
                i, xi, yi, w, h = placement
                self.action_queue.append({
                    "stock_idx": stockidxreturn,
                    "size": [w, h],
                    "position": [xi, yi]
                })

            if self.action_queue:
                return self.action_queue.pop(0)
            else:
                return {'stock_idx': -1, 'size': [0, 0], 'position': [0, 0]}

    def _adjust_stock_idx(self):
        if self.stock_idx in self.stock_placed and not self.last_check and self.check_100:
            self.stock_idx -= 1
        elif self.stock_idx in self.stock_placed and self.last_check and self.check_100:
            self.stock_idx += 1

        if self.stock_idx == 99:
            self.check_100 = True
        if not self.check_100:
            self.stock_idx += 1
        elif self.check_100:
            self.stock_idx = self.stock_idx + 1 if self.last_check else self.stock_idx - 1

    def _initialize_patterns(self):
        for i in range(self.num_products):
            counts = np.zeros(self.num_products, dtype=int)
            counts[i] = 1
            placements = [(i, 0, 0, self.demands[i]["size"][0], self.demands[i]["size"][1])]
            self.current_patterns.append({
                'counts': counts,
                'placements': placements,
                'unit_values': [0]
            })

    def _should_generate_action(self, a, placements):
        if a < 0.9 and not self.check_100 and not self.last_check:
            return False
        elif a < 0.8 and self.check_100 and not self.last_check:
            return False
        return bool(placements)






