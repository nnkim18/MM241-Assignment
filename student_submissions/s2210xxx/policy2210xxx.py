from policy import Policy
import numpy as np

class Policy2210xxx(Policy):
    def __init__(self):
        # Student code here
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        # Sắp xếp sản phẩm theo diện tích giảm dần
        list_prods = sorted(
            list_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True
        )

        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Khởi tạo bảng DP
            dp = [[{"num_products": 0, "used_area": 0, "placements": []}
                   for _ in range(stock_h + 1)] for _ in range(stock_w + 1)]

            # Duyệt qua từng sản phẩm
            for prod in list_prods:
                prod_width, prod_height = prod["size"]
                if prod["quantity"] > 0:
                    # Duyệt qua bảng DP từ lớn đến nhỏ
                    for w in range(stock_w, prod_width - 1, -1):
                        for h in range(stock_h, prod_height - 1, -1):
                            if (
                                dp[w - prod_width][h - prod_height]["num_products"] + 1
                                > dp[w][h]["num_products"]
                                and self._can_place_(stock, (w - prod_width, h - prod_height), prod["size"])
                            ):
                                dp[w][h]["num_products"] = (
                                    dp[w - prod_width][h - prod_height]["num_products"] + 1
                                )
                                dp[w][h]["used_area"] = (
                                    dp[w - prod_width][h - prod_height]["used_area"]
                                    + (prod_width * prod_height)
                                )
                                dp[w][h]["placements"] = (
                                    dp[w - prod_width][h - prod_height]["placements"]
                                    + [(prod, (w - prod_width, h - prod_height))]
                                )

            # Tìm vị trí tốt nhất trong bảng DP
            max_products = 0
            best_placement = None
            for w in range(stock_w + 1):
                for h in range(stock_h + 1):
                    if dp[w][h]["num_products"] > max_products:
                        max_products = dp[w][h]["num_products"]
                        best_placement = dp[w][h]["placements"]

            # Nếu tìm thấy cách xếp
            if best_placement:
                # Đặt các sản phẩm vào vị trí đã chọn
                for placement in best_placement:
                    prod, pos = placement
                    if prod["quantity"] > 0:
                        prod["quantity"] -= 1
                        return {
                            "stock_idx": stock_idx,
                            "size": prod["size"],
                            "position": pos,
                        }

        # Nếu không tìm thấy vị trí hợp lệ
        return {"stock_idx": -1, "size": [0, 0], "position": (None, None)}