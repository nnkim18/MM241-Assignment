from policy import Policy


class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

        # Initialize level-related variables
        self.level_pos = 0
        self.level_h = 0
        self.is_new_level = True

        # Processed products' indices
        self.processed_prods = []

    def _get_sorted_stocks_indices(self, stocks) -> tuple[int]:
        # Sort the stocks in the ascending order of the stock's height
        # Return the list of stock indices
        indices = sorted(range(len(stocks)),
                         key=lambda i: self._get_stock_size_(stocks[i])[1])
        return tuple(indices)

    def _get_sorted_products_indices(self, products) -> tuple[int]:
        # Sort the products in the descending order of the product's height
        # Return the list of product indices
        indices = sorted(range(len(products)),
                         key=lambda i: products[i]["size"][1], reverse=True)
        return tuple(indices)

    def get_action(self, observation, info):
        # Sorting the stocks
        stocks = observation["stocks"]
        sorted_stocks_indices = self._get_sorted_stocks_indices(stocks)

        # Sorting the products
        products = observation["products"]
        sorted_products_indices = self._get_sorted_products_indices(products)

        # Main Algorithm
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        i = 0
        while i < len(sorted_stocks_indices):
            stock = stocks[sorted_stocks_indices[i]]
            stock_w, stock_h = self._get_stock_size_(stock)

            # Choose a product that can be placed in the stock
            pos_x, pos_y = None, None
            j = 0
            while j < len(sorted_products_indices):
                prod_idx = sorted_products_indices[j]
                prod = products[prod_idx]
                prod_size = prod["size"]
                prod_w, prod_h = int(prod_size[0]), int(prod_size[1])

                if prod["quantity"] == 0:
                    j += 1
                    continue

                # Check if the product can be placed in the stock
                if prod_w <= stock_w and prod_h <= stock_h and prod_idx not in self.processed_prods:
                    # Check if it is a new level
                    if self.is_new_level:
                        self.level_h = prod_h
                        self.is_new_level = False

                    # Check if the product can be placed in the current level
                    if prod_h + self.level_pos <= self.level_h + self.level_pos:
                        for x in range(stock_w - prod_w + 1):
                            if self._can_place_(stock, (x, self.level_pos), prod_size):
                                pos_x, pos_y = x, self.level_pos
                                self.processed_prods.append(prod_idx)
                                break
                    else:  # Move to the next level
                        j = 0  # Reset the product index
                        self.level_pos += self.level_h
                        self.is_new_level = True
                        self.processed_prods = []  # Reset the processed products
                        continue

                if pos_x is not None and pos_y is not None:
                    break

                j += 1

            # If a product is found, break the loop
            if pos_x is not None and pos_y is not None:
                stock_idx = sorted_stocks_indices[i]
                break

            # If can't create a new level, move to the next stock
            if self.level_h + self.level_pos < stock_h:
                self.level_pos += self.level_h
                self.is_new_level = True
                self.processed_prods = []
                continue

            # Reset the level-related variables for the next stock
            self.level_pos = 0
            self.level_h = 0
            self.is_new_level = True
            self.processed_prods = []
            i += 1

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
