from policy import Policy
import numpy as np

class Greedy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        if info["filled_ratio"] == 0:
            stocks = np.array(observation["stocks"]) == -1
            s_size = np.vstack((np.sum(stocks[:,:,0], axis=1), np.sum(stocks[:,0], axis=1))).T

            s_rotate = s_size[:,0] > s_size[:,1]
            s_size[s_rotate] = s_size[s_rotate,::-1]
            stocks[s_rotate] = np.transpose(stocks[s_rotate], (0,2,1))

            p_size = np.array([product["size"] for product in observation["products"]])
            p_size.sort(axis=1)
            products = np.column_stack((p_size, np.array([prod["quantity"] for prod in observation["products"]])))[np.argsort(-p_size[:,1])]

            self.actions = []

            for i in np.argsort(-s_size[:,1]):
                for product in products:
                    x_range = range(s_size[i, 1] - product[1] + 1)
                    for y in range(s_size[i, 0] - product[0] + 1):
                        for x in x_range:
                            if np.all(stocks[i, y:y + product[0], x:x + product[1]]):
                                stocks[i, y:y + product[0], x:x + product[1]].fill(False)
                                product[2] -= 1
                                self.actions.append({"stock_idx": i, "size": product[[1,0]], "position": (x, y)} if s_rotate[i]
                                                    else {"stock_idx": i, "size": product[[0,1]], "position": (y, x)})
                                if product[2] == 0: break
                        if product[2] == 0: break
                products = products[products[:,2] > 0]

                for product in products:
                    x_range = range(s_size[i, 1] - product[0] + 1)
                    for y in range(s_size[i, 0] - product[1] + 1):
                        for x in x_range:
                            if np.all(stocks[i, y:y + product[1], x:x + product[0]]):
                                stocks[i, y:y + product[1], x:x + product[0]].fill(False)
                                product[2] -= 1
                                self.actions.append({"stock_idx": i, "size": product[[0,1]], "position": (x, y)} if s_rotate[i]
                                                    else {"stock_idx": i, "size": product[[1,0]], "position": (y, x)})
                                if product[2] == 0: break
                        if product[2] == 0: break
                products = products[products[:,2] > 0]
                if len(products) == 0: break

        return self.actions.pop()