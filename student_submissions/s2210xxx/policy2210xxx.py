from policy import Policy
import numpy as np




class Policy2210xxx(Policy):  # Lấy product từ lớn đến nhỏ, trái qua phải, lấy theo thứ tự
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
    
