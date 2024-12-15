from s2210xxx.Policy2313221_2311269_2313765_2313446_2213989 import Policy
import numpy as np


class Policy2313221(Policy):  # Ngẫu nhiên từ trên xuống dưới, trái qua phải, lấy theo thứ tự, nhưng chỉ lấy stock mới khi stock đầu đã đầy
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = None, None

        # Duyệt qua danh sách sản phẩm để chọn sản phẩm có số lượng > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Duyệt qua các kho (stocks)
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # Nếu sản phẩm không thể vừa kho, bỏ qua
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Áp dụng thuật toán Bottom-Left để tìm vị trí khả dụng
                    best_x, best_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                if best_x is None or (y < best_y or (y == best_y and x < best_x)):
                                    best_x, best_y = x, y

                    if best_x is not None and best_y is not None:
                        pos_x, pos_y = best_x, best_y
                        stock_idx = i
                        break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

class Policy2311269(Policy):  # Lấy product ngẫu nhiên từ trên xuống dưới, trái qua phải, lấy stock theo thứ tự
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stock_idx = -1
        pos_x, pos_y = None, None

        # Duyệt qua từng kho theo thứ tự
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Duyệt qua từng sản phẩm
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    # Kiểm tra nếu sản phẩm có thể đặt vào kho
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Áp dụng thuật toán Bottom-Left để tìm vị trí phù hợp
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                # Nếu đặt được sản phẩm, trả về hành động
                                return {
                                    "stock_idx": i,
                                    "size": prod_size,
                                    "position": (x, y),
                                }
                            

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    

class Policy2313765(Policy):  # Lấy product từ lớn đến nhỏ, trái qua phải, lấy theo thứ tự
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        # Sắp xếp danh sách sản phẩm theo kích thước giảm dần (diện tích)
        sorted_prods = sorted(
            list_prods,
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )

        stock_idx = -1
        pos_x, pos_y = None, None

        # Duyệt qua từng kho theo thứ tự
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Duyệt qua từng sản phẩm (đã được sắp xếp từ lớn đến nhỏ)
            for prod in sorted_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    # Kiểm tra nếu sản phẩm có thể đặt vào kho
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Áp dụng thuật toán Bottom-Left để tìm vị trí phù hợp
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                # Nếu đặt được sản phẩm, trả về hành động
                                # print({
                                #     "stock_idx": i,
                                #     "size": prod_size,
                                #     "position": (x, y),
                                # })
                                return {
                                    "stock_idx": i,
                                    "size": prod_size,
                                    "position": (x, y),
                                }         
        #print({"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)})
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
class Policy2313446(Policy): # theo file 2023 (thuật toán Column Generation) 
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        # 1. Tìm các mẫu cắt khả thi (Column Generation)
        patterns = []
        for i, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod in list_prods: 
                if prod["quantity"] > 0:
                    prod_w, prod_h = prod["size"]
                    if stock_w >= prod_w and stock_h >= prod_h:
                        position = self._find_position(stock, prod["size"])
                        if position is not None:
                            patterns.append({
                                "stock_idx": i,
                                "size": prod["size"],
                                "position": position,
                                "quantity": prod["quantity"],
                            })

        # 2. Tính toán giá trị hàm mục tiêu (Dynamic Programming - Knapsack)
        best_pattern = None
        max_value = float('-inf')

        for pattern in patterns:
            stock_idx = pattern["stock_idx"]
            size = pattern["size"]
            pos_x, pos_y = pattern["position"]

            # Ước lượng giá trị (lượng sản phẩm được cắt)
            value = size[0] * size[1] * pattern["quantity"]  # Diện tích x số lượng
            if value > max_value:
                max_value = value
                best_pattern = pattern

        # 3. Trả về hành động tối ưu
        if best_pattern:
            return {
                "stock_idx": best_pattern["stock_idx"],
                "size": best_pattern["size"],
                "position": best_pattern["position"]
            }
        else:
            # Không tìm thấy mẫu khả thi
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_position(self, stock, prod_size):
        """Tìm vị trí khả thi để đặt sản phẩm trong kho."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None



class Policy2213989(Policy): # áp dụng Greedy, Heuristic và Bottom-Left 
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation.get("products", [])
        stocks = observation.get("stocks", [])

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        #Duyệt qua mẫu
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Tìm stock vừa
                for i, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Skip stock nếu k vừa
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # tìm vị trí phù hợp
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                stock_idx = i
                                pos_x, pos_y = x, y
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        # nếu không trả về mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}


