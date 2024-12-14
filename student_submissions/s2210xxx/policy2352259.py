import numpy as np
from policy import Policy
class Policy2352259(Policy):
    def __init__(self, policy_id=1):
        self.policy_id = policy_id
        self.skylines = {}
        self.used_stocks = set()
        self.stock_size_cache = {}
        self.filled_area = 0
        self.last_observation = None
    
    def get_action(self, observation, info):
        if (self.last_observation is None or 
            len(observation["products"]) != len(self.last_observation["products"]) or
            any(p1["quantity"] != p2["quantity"] 
                for p1, p2 in zip(observation["products"], self.last_observation["products"]))):
            self.skylines = {}
            self.used_stocks = set()
            self.stock_size_cache = {}
            self.filled_area = 0
        
        self.last_observation = observation.copy()
        
        if self.policy_id == 1:
            return self.skyline_packing(observation, info)
        else:
            return self.level_packing(observation, info)
            
    def skyline_packing(self, observation, info):
        stocks = observation["stocks"]
        products = [
            (i, p) for i, p in enumerate(observation["products"]) if p["quantity"] > 0
        ]
        
        # Sort by area * quantity, then by max dimension
        products.sort(key=lambda x: (
            -(x[1]["size"][0] * x[1]["size"][1] * x[1]["quantity"]),
            -max(x[1]["size"]),
            -min(x[1]["size"])
        ))

        best_action = None
        best_score = float('-inf')

        for prod_idx, prod in products:
            w, h = prod["size"]
            orientations = [(w, h)]
            if w != h:  # Try both orientations if not square
                orientations.append((h, w))

            # First try used stocks
            used_stocks = [(idx, stock) for idx, stock in enumerate(stocks) 
                          if idx in self.used_stocks]
            unused_stocks = [(idx, stock) for idx, stock in enumerate(stocks) 
                           if idx not in self.used_stocks]
            
            # Try used stocks first, then unused
            for stock_group in [used_stocks, unused_stocks]:
                for stock_idx, stock in stock_group:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    
                    # Initialize skyline if not exists
                    if stock_idx not in self.skylines:
                        self.skylines[stock_idx] = np.zeros(stock_w, dtype=int)

                    for width, height in orientations:
                        if width > stock_w or height > stock_h:
                            continue

                        # Find valid positions along skyline
                        positions = self._find_skyline_positions(
                            stock_idx, stock, width, height
                        )

                        for pos in positions:
                            score = self._calculate_skyline_score(
                                stock, pos, (width, height), stock_idx
                            )
                            
                            if score > best_score:
                                best_score = score
                                best_action = {
                                    "stock_idx": stock_idx,
                                    "size": (width, height),
                                    "position": pos
                                }

                        if best_action and best_action["stock_idx"] == stock_idx:
                            # Update skyline and mark stock as used
                            self._update_skyline(stock_idx, best_action)
                            self.used_stocks.add(stock_idx)
                            return best_action

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def _find_skyline_positions(self, stock_idx, stock, width, height):
        stock_w, stock_h = self._get_stock_size_(stock)
        skyline = self.skylines[stock_idx]
        positions = []

        # Ensure we don't exceed array bounds
        for x in range(min(stock_w - width + 1, len(skyline) - width + 1)):
            # Safely get height for the region
            region = skyline[x:min(x + width, len(skyline))]
            if len(region) < width:
                continue
                
            max_height = max(region)
            y = max_height
            
            if y + height <= stock_h:
                if self._can_place_(stock, (x, y), (width, height)):
                    positions.append((x, y))
                    
        return positions

    def _update_skyline(self, stock_idx, action):
        x, y = action["position"]
        w, h = action["size"]
        skyline = self.skylines[stock_idx]
        
        # Ensure we stay within bounds
        x_end = min(x + w, len(skyline))
        skyline[x:x_end] = y + h

    def _calculate_skyline_score(self, stock, pos, size, stock_idx):
        x, y = pos
        w, h = size
        stock_w, stock_h = self._get_stock_size_(stock)
        skyline = self.skylines[stock_idx]

        # Waste calculation with height penalty
        x_end = min(x + w, len(skyline))
        waste_area = sum(y - skyline[i] for i in range(x, x_end))
        height_penalty = (y + h) / stock_h  # Penalize high placements
        
        # Enhanced contact scoring
        contact = 0
        # Edge bonuses with corner rewards
        if x == 0: 
            contact += h * 1.5
            if y == 0: contact += h  # Corner bonus
        if y == 0: 
            contact += w * 1.5
            if x + w == stock_w: contact += w  # Corner bonus
        if x + w == stock_w: contact += h * 1.5
        if y + h == stock_h: contact += w * 1.2

        # Adjacent piece bonuses
        adjacent_count = 0
        for i in range(x, x + w):
            if skyline[i] == y:
                adjacent_count += 1
        contact += adjacent_count * 2.0  # Stronger bonus for adjacent pieces

        # Height balance score
        nearby_range = max(w * 2, 10)  # Look at nearby region
        x_start = max(0, x - nearby_range)
        x_finish = min(len(skyline), x + w + nearby_range)
        nearby_heights = skyline[x_start:x_finish]
        height_variation = np.std(nearby_heights) if len(nearby_heights) > 0 else 0
        balance_score = 1.0 / (1.0 + height_variation)

        # Normalize scores
        waste_score = 1.0 - (waste_area / (stock_w * stock_h))
        contact_score = contact / (4 * (w + h))  # Adjusted normalization
        
        # Combined score with balance
        return (
            0.5 * waste_score + 
            0.3 * contact_score +
            0.2 * balance_score -
            0.1 * height_penalty
        )
    
    def level_packing(self, observation, info):
        stocks = observation["stocks"]
        products = [
            (i, p) for i, p in enumerate(observation["products"]) if p["quantity"] > 0
        ]
        # Sort products to prioritize larger products
        products.sort(
            key=lambda x: (
                -(x[1]["size"][0] * x[1]["size"][1] * x[1]["quantity"]),
                -max(x[1]["size"]),
            )
        )

        best_action = None
        best_score = -float('inf')

        for prod_idx, prod in products:
            w, h = prod["size"]
            orientations = [(w, h)]
            if w != h:
                orientations.append((h, w))

            for idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)

                for width, height in orientations:
                    if width > stock_w or height > stock_h:
                        continue  # Skip orientations that don't fit

                    positions = self._find_positions(stock, width, height)
                    for pos in positions:
                        # Simulate placement
                        simulated_stock = stock.copy()
                        self._place(simulated_stock, pos, (width, height), prod_idx)

                        # Calculate score
                        score = self._calculate_placement_score(simulated_stock)

                        # Check if this is the best action so far
                        if score > best_score:
                            best_score = score
                            best_action = {
                                "stock_idx": idx,
                                "size": (width, height),
                                "position": pos,
                            }

                        # Break after finding a valid placement
                        break  # Remove if you want to consider all positions

                    # Early exit if a best action is found
                    if best_action and best_action['stock_idx'] == idx:
                        break  # Exit orientation loop

                # Early exit if a best action is found
                if best_action and best_action['stock_idx'] == idx:
                    break  # Exit stock loop

            # Early exit if a best action is found
            if best_action:
                break  # Exit product loop

        if best_action:
            return best_action
        else:
            # No valid action found
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
    
    def _get_stock_size_(self, stock):
        key = id(stock)
        if key not in self.stock_size_cache:
            self.stock_size_cache[key] = (
                np.any(stock != -2, axis=1).sum(),
                np.any(stock != -2, axis=0).sum(),
            )
        return self.stock_size_cache[key]
    
    def _find_positions(self, stock, width, height):
        positions = []
        stock_w, stock_h = self._get_stock_size_(stock)
        occupancy_map = (stock != -1)
        for y in range(0, stock_h - height + 1):
            x_range = range(0, stock_w - width + 1)
            for x in x_range:
                if occupancy_map[x:x+width, y:y+height].any():
                    continue
                positions.append((x, y))
                # Break after finding the first valid position in the row
                break
        return positions
    
    def _place(self, stock, position, size, prod_idx=0):
        x, y = position
        w, h = size
        stock[x : x + w, y : y + h] = prod_idx  # Simulate placement

    def _calculate_placement_score(self, stock):
        # Calculate filled area
        if 'filled_area' not in self.__dict__:
            self.filled_area = np.sum(stock >= 0)
        else:
            self.filled_area += 1  # Adjust based on actual placement size

        stock_w, stock_h = self._get_stock_size_(stock)
        total_area = stock_w * stock_h
        utilization = self.filled_area / total_area if total_area > 0 else 0

        # Fragmentation Penalty
        fragmented = (stock == -1) & (
            (np.roll(stock, 1, axis=0) == -1) |
            (np.roll(stock, -1, axis=0) == -1) |
            (np.roll(stock, 1, axis=1) == -1) |
            (np.roll(stock, -1, axis=1) == -1)
        )
        fragmented_cells = np.sum(fragmented)
        fragmentation_penalty = fragmented_cells / total_area if total_area > 0 else 0

        # Contact Scoring using vectorized operations
        occupied = (stock >= 0).astype(int)

        # Shift in four directions: up, down, left, right
        contact_up = np.roll(occupied, 1, axis=0)
        contact_down = np.roll(occupied, -1, axis=0)
        contact_left = np.roll(occupied, 1, axis=1)
        contact_right = np.roll(occupied, -1, axis=1)

        # Prevent wrap-around by zeroing the borders
        contact_up[0, :] = 0
        contact_down[-1, :] = 0
        contact_left[:, 0] = 0
        contact_right[:, -1] = 0

        # Total contacts per cell
        total_contacts = contact_up + contact_down + contact_left + contact_right
        contact_score = np.sum(total_contacts * occupied)

        # Normalize contact score
        max_contacts = 4 * self.filled_area if self.filled_area > 0 else 1
        normalized_contact = contact_score / max_contacts if max_contacts > 0 else 0

        # Combined score with weights
        score = (
            0.5 * utilization +
            0.3 * normalized_contact -
            0.2 * fragmentation_penalty
        )

        return score