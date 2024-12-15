from policy import Policy
import numpy as np

class Policy2352189_2352390_2352170_2352598_2352592(Policy):
    def __init__(self, policy_id = 1):
        self.population_size = 10   # Giảm số cá thể để giảm thời gian chạy
        self.mutation_rate = 0.8   # Giảm tỷ lệ mutation
        self.rotation_rate = 0
        self.max_attempts = 100
        self.crossover_rate = 0.5
        self.elite_size_select = 1
        self.generation_count = 1
        self.fitness_list = []
        self.actions = []
        self.products = self.stocks = []
        self.policy_id = policy_id
        
    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.GeneticAlgorithm(observation, info)
        elif self.policy_id == 2:
            return self.BestFitDecreasing(observation, info)

    def new_individual(self):
        product_counts = [0] * len(self.products)
        sheets = []
        #indices = np.arange(len(self.products))
        indices = list(range(len(self.products)))
        #original_indices = indices.copy()
        
        if np.random.random() < self.mutation_rate:
            np.random.shuffle(indices)
        else:
                    # Dùng FFD
            indices = sorted(indices, key=lambda p: self.products[p]["size"][0] * self.products[p]["size"][1], reverse=True)
        for stock in self.stocks:
            sheet = stock.copy()
            W, H = (
                np.sum(np.any(stock != -2, axis=1)),
                np.sum(np.any(stock != -2, axis=0)),
            )

            # Ma trận đánh dấu các ô trống
            empty_map = np.full((H, W), True)

            #for i, product in enumerate(self.products):
            #for i in range(len(indices)):
            for i in indices:
                product = self.products[i]
                w, h = product["size"]
                #old_index = original_indices[indices[i]]
                #w, h = self.products[old_index]["size"]

                # Xoay sản phẩm nếu cần
                if np.random.random() < self.rotation_rate:
                    w, h = h, w

                if product_counts[i] >= product["quantity"]:
                    continue  # Bỏ qua nếu đã đặt đủ sản phẩm này

                # Duyệt để đặt sản phẩm
                for x in range(W - w + 1):
                    for y in range(H - h + 1):
                        if empty_map[y : y + h, x : x + w].all():
                            # Đặt sản phẩm vào vị trí (x, y)
                            sheet[y : y + h, x : x + w] = i
                            empty_map[y : y + h, x : x + w] = False
                            product_counts[i] += 1
                            if product_counts[i] >= product["quantity"]:
                                break  # Dừng khi đạt đủ số lượng
                    if product_counts[i] >= product["quantity"]:
                        break

            sheets.append(sheet)

        fitness = self.evaluate_fitness((sheets, product_counts))
        self.fitness_list.append(fitness)
        return sheets, product_counts

    def init(self):
        # Sinh cá thể ban đầu
        self.population = []
        # Sinh từng cá thể
        for _ in range(self.population_size):
            self.population.append(self.new_individual())


    def evaluate_fitness(self, individual):
        trim_loss = []

        for sid, stock in enumerate(individual[0]):
            if np.all(stock<0) == False:
                tl = (stock == -1).sum() / (stock != -2).sum()
                trim_loss.append(tl)

        result = np.mean(trim_loss).item() if trim_loss else 1
        return 1-result

    def calculate_fitness(self):
        self.fitness_points = [self.evaluate_fitness(ind) for ind in self.population]

    def crossover(self):
        # Lấy các cá thể tốt nhất
        elite_size = self.elite_size_select
        ranked_population = sorted(
            self.population, key=lambda x: self.evaluate_fitness(x), reverse=True
        )
        elite = ranked_population[:elite_size]

        # Sinh các cá thể mới thay thế các cá thể yếu
        offspring = [self.new_individual() for _ in range(self.population_size - elite_size)]
        self.population = elite + offspring

    def evolve(self):
        for _ in range(self.generation_count):
            self.calculate_fitness()
            self.crossover()

    def GeneticAlgorithm(self, observation, info):
        if len(self.actions) > 0:
            action = self.actions[0]
            self.actions.pop(0)
            return action
        else:        
            self.stocks = observation["stocks"]
            self.products = observation["products"]
            prods_length=0
            for prods_ in self.products:
                prods_length+=prods_["quantity"]
            self.max_attempts=prods_length
        # Đếm tổng số sản phẩm ban đầu
            #total_products = sum(product["quantity"] for product in self.products)
            #self.max_attempts = total_products

        # Khởi tạo dân số và tiến hóa
            self.init()
            self.evolve()

        # Sắp xếp dân số theo fitness
            self.population = sorted(self.population, key=lambda x: self.evaluate_fitness(x), reverse=True)

        # Chuyển đổi cá thể tốt nhất sang các hành động
            sheets, product_counts = self.population[0]

        # Chuyển đổi sheets thành danh sách hành động
            for i, _ in enumerate(observation["stocks"]):
                sheet = sheets[i]
                W = np.sum(np.any(sheet != -2, axis=1))
                H = np.sum(np.any(sheet != -2, axis=0))
                for x in range(W):
                    for y in range(H):
                        pos_val = sheet[y][x]
                        if pos_val >= 0:
                            curr_prod = self.products[pos_val]
                            w, h = curr_prod["size"]
                            #w, h = size
                            sheet[y:y + h, x:x + w] = -1
                            action = {
                                "stock_idx": i,
                                "size": [w, h],
                                "position": (x, y)
                            }
                            self.actions.append(action)
            if len(self.actions) > 0:
                action = self.actions[0]
                self.actions.pop(0)
                return action

        # Nếu không còn hành động nào, trả về hành động mặc định
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    def BestFitDecreasing(self, observation, info):
        if info["filled_ratio"] == 0:
            self.stocks = sorted(observation["stocks"],key=lambda p: (w := self._get_stock_size_(p))[0] * w[1])
            self.products = sorted(observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse = True)

        for product_index, product in enumerate(self.products):
            if product["quantity"] == 0: continue
            for stock_index, stock in enumerate(self.stocks):
                if product["quantity"] == 0: break
                W, H = self._get_stock_size_(stock)
                w, h = product["size"]
                
                #print([np.count_nonzero(stock == -1) for stock in self.stocks])
                
                for y in range(H - min(w, h) + 1):
                    for x in range(W - min(w, h) + 1):
                        if stock[y, x] >= 0: x += w
                        
                        if np.all(stock[y:y+h, x:x+w] == -1):
                            stock[y:y+h, x:x+w] = product_index
                            product["quantity"] -= 1
                            self.heapifyStocks(stock_index)                
                            return {"stock_idx": stock_index, "size": [w, h], "position": (x, y)}
                            
                        elif np.all(stock[y:y+w, x:x+h] == -1):
                            stock[y:y+w, x:x+h] = product_index
                            product["quantity"] -= 1
                            self.heapifyStocks(stock_index)
                            return {"stock_idx": stock_index, "size": [h, w], "position": (x, y)}
                        
                        if product["quantity"] == 0: break
                    if product["quantity"] == 0: break         
                    
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        
    def heapifyStocks(self, index):
        l, r = 0, len(self.stocks) - 1
        m = (l + r) // 2
        while l < r:
            m = (l + r) // 2
            if np.count_nonzero(self.stocks[m] == -1) < np.count_nonzero(self.stocks[index] == -1): l = m + 1
            elif np.count_nonzero(self.stocks[m] == -1) == np.count_nonzero(self.stocks[index] == -1): r = m
            else: r = m - 1
            
        if index < m: m += 1
        self.stocks.insert(m, self.stocks.pop(index))