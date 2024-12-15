from policy import Policy
import numpy as np


class Policy2210404_2310334_2211366_2113971_2311849(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._bld(observation)
        elif self.policy_id == 2:
            return self._nfdh(observation)

    def _bld(self, observation):
        """
        Implements the Bottom-Left-Decreasing (BLD) algorithm.
        This places products into stocks by sorting them in descending size order
        and placing them from the bottom-left corner upwards.
        """
        # Sort products by their largest dimension in descending order
        list_prods = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda p: max(p["size"]),  # Sort by the larger dimension
            reverse=True
        )

        # Iterate over each product
        for prod in list_prods:
            prod_size = prod["size"]

            # Check each stock for placement
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Try to place the product in its current orientation
                for x in range(stock_w - prod_size[0] + 1):
                    for y in range(stock_h - prod_size[1] + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            prod["quantity"] -= 1  # Deduct the product quantity
                            return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}

                # Try to place the product in rotated orientation
                prod_size_rotated = prod_size[::-1]
                for x in range(stock_w - prod_size_rotated[0] + 1):
                    for y in range(stock_h - prod_size_rotated[1] + 1):
                        if self._can_place_(stock, (x, y), prod_size_rotated):
                            prod["quantity"] -= 1  # Deduct the product quantity
                            return {"stock_idx": stock_idx, "size": prod_size_rotated, "position": (x, y)}

        # If no placement is possible, return an invalid action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _nfdh(self, observation):
        """
        Implements the Next-Fit-Decreasing-Height (NFDH) algorithm.
        This packs products into stocks row by row while minimizing trim loss.
        """
        list_prods = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda p: p["size"][1],  # Sort by height
            reverse=True
        )

        for stock_idx, stock in enumerate(observation["stocks"]):  # Process one stock at a time
            stock_w, stock_h = self._get_stock_size_(stock)

            # Initialize placement variables
            current_x, current_y = 0, 0  # Start from the top-left corner
            max_row_height = 0  # Track the maximum height of the current row

            for prod in list_prods:
                if prod["quantity"] == 0:
                    continue  # Skip products with zero quantity

                prod_size = prod["size"]

                while current_y + prod_size[1] <= stock_h:  # Ensure we stay within the stock height
                    if current_x + prod_size[0] <= stock_w:  # Check if product fits horizontally
                        if self._can_place_(stock, (current_x, current_y), prod_size):
                            prod["quantity"] -= 1  # Deduct the quantity of this product
                            return {"stock_idx": stock_idx, "size": prod_size, "position": (current_x, current_y)}
                        current_x += prod_size[0]  # Move to the right
                    else:  # Start a new row
                        current_y += max_row_height
                        current_x = 0
                        max_row_height = 0  # Reset row height

                    max_row_height = max(max_row_height, prod_size[1])  # Update row height

                # Try rotated placement
                prod_size_rotated = prod_size[::-1]
                current_x, current_y = 0, 0  # Reset position variables
                max_row_height = 0

                while current_y + prod_size_rotated[1] <= stock_h:
                    if current_x + prod_size_rotated[0] <= stock_w:
                        if self._can_place_(stock, (current_x, current_y), prod_size_rotated):
                            prod["quantity"] -= 1  # Deduct the quantity of this product
                            return {"stock_idx": stock_idx, "size": prod_size_rotated, "position": (current_x, current_y)}
                        current_x += prod_size_rotated[0]
                    else:  # Start a new row
                        current_y += max_row_height
                        current_x = 0
                        max_row_height = 0

                    max_row_height = max(max_row_height, prod_size_rotated[1])  # Update row height

        # If no placement is possible, return an invalid action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _get_stock_size_(self, stock):
        """
        Returns the dimensions (width, height) of the stock.
        """
        stock_h, stock_w = stock.shape  # Access shape of the stock
        return stock_w, stock_h  # Return width and height

def _can_place_(self, stock, position, prod_size):
    """
    Checks if a product can be placed at the given position in the stock.
    """
    x, y = position
    prod_w, prod_h = prod_size

    # Ensure the product is within bounds
    if x + prod_w > stock.shape[1] or y + prod_h > stock.shape[0]:
        return False

    # Check if all cells in the placement area are free
    region = stock[y:y + prod_h, x:x + prod_w]
    if np.any(region != 0):
        return False

    return True
