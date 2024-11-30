from policy import Policy


class Policy2310359(Policy):
    def __init__(self):
        # Student code here
        self.skylines = []
        pass

    def get_action(self, observation, info):
        # Student code here

        if len(self.skylines) == 0:
            self.init_skyline_set(observation["stocks"])
            print(self.skylines)

        ### this makes code run normal

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

    class SkylinePart():
        def __init__(self, x_start, x_end, height) -> None:
            self.x_start = x_start
            self.x_end = x_end
            self.height = height

        def _intersect_(self, prod_size, prod_pos) -> bool:
            # intersect here means positive common areas, which makes this invalid

            prod_x_start, prod_y = prod_pos
            prod_w, prod_h = prod_size
            prod_x_end = prod_x_start + prod_w

            if (prod_x_end <= self.x_start or prod_x_start >= self.x_end): return False

            if (prod_y >= self.height): return False
            
            return True


    #####################


    # skyline is a set of 3-int tuple (x_start, x_end, height)
    def init_skyline_set(self, stocks):
        # this will initialize the skyline set
        for i, stock in enumerate(stocks):
            self.skylines += [[Policy2310359.SkylinePart(0, self._get_stock_size_(stock), 0)]]

    def local_waste_calc(self, product, stock, stock_idx):
        # Current idea: 
        # Each time we add the product to make smallest local waste
        # This function calculates local waste
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = product["size"]

        WONT_USE = 1e18
        current_waste = WONT_USE

        # First, we try placing on the leftmost of each part of skyline
        for part_idx, part in enumerate(self.skylines[stock_idx]):
            # now have to check validness
            pos_x, pos_y = part.x_start, part.height

            # check if product in stock
            if not (pos_x + prod_w <= stock_w and pos_y + prod_h <= prod_h): continue

            # Must not intersect other part
            ok = True
            for other_part in self.skylines[part_idx]:
                if other_part.insersect(product["size"], (pos_x, pos_y)):
                    ok = False
                    break
            
            if not ok: continue

            waste_area = 0

