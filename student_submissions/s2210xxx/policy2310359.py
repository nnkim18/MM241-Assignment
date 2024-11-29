from policy import Policy


class Policy2310359(Policy):
    def __init__(self):
        # Student code here
        pass

    def get_action(self, observation, info):
        # Student code here
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in list_prods:
            if (prod["quantity"] > 0):
                prod_size = prod["size"]

                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_h or stock_h < prod_h:
                        continue

                    if not self._can_place_(stock, (0, 0), prod_size):
                        continue

                    return {"stock_idx": i, "size": prod_size, "position": (0, 0)}
        pass

    # Student code here
    # You can add more functions if needed
