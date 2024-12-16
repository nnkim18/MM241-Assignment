from policy import Policy
import numpy as np



class Policy2312329_2312285_2312430_2312419_2312658(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if self.policy_id == 1:
            self.list_stocks = []
            self.list_prods = []
            pass
        elif self.policy_id == 2:
            self.T = 00
            self.square_stock = np.zeros(100100)
            self.obs = 0
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return self.get_action1(observation, info)
        elif self.policy_id == 2:
            return self.get_action2(observation, info)
    

    def get_action1(self, observation, info):
        if(info["filled_ratio"]==0):
            self.__init__()
            
            for i,stock in enumerate( observation["stocks"]):
                stock_w,stock_h = self._get_stock_size_(stock)
                self.list_stock = {"idx":i,"stock":stock,"dientich":stock_w * stock_h}
                self.list_stocks.append(self.list_stock)
            self.list_stocks.sort(key=lambda item: item["dientich"], reverse=True) 
            
            
            self.list_prods = list(observation["products"]) #tạo list_prods
            self.list_prods.sort(key=lambda x: (x["size"][0] * x["size"][1]), reverse=True) 
        
        
        stock_idx = -1
        pos_x, pos_y = None, None

        
        for prod in self.list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                
                for item in self.list_stocks: 
                

                    i = item["idx"]
                    
                    stock = item["stock"] 
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    
         
                    if (stock_w > stock_h):
                        if prod_w < prod_h:
                            prod_w,prod_h = prod_h,prod_w
                    else:
                        if prod_w > prod_h:
                            prod_w,prod_h = prod_h,prod_w
                    
                    
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = self._find_bottom_left_position(stock, prod_size)

                    if pos_x is None and pos_y is None:
                        prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
                        pos_x, pos_y = self._find_bottom_left_position(stock, prod_size)
                    
           
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break
   
            if pos_x is not None and pos_y is not None:
                break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


    def _find_bottom_left_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock) 
        prod_w, prod_h = prod_size 

        for y in range(stock_h - prod_h, -1, -1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size): 
                    return x, y  
        return None, None
    

    def SA_help(self, stock, square_stock, i, k):
        #stock_w2, stock_h2 = self._get_stock_size_(stock2)
        DE = self.square(stock, i, square_stock) - self.square(stock, k, square_stock)
        return DE
        #pass

    def set_obs(self, observation):
        self.obs = observation
        pass

    def square(self, stock, i, square_stock):
        stock_w, stock_h = self._get_stock_size_(stock[i])
        S_remain = stock_w * stock_h - square_stock[i]
        return S_remain
        #pass

    def set_T(self, observation):
        list = observation["products"]
        for prod in list:
            self.T += prod["quantity"]
        pass                  

    def get_action2(self, observation, info):
     self.set_obs(observation)
     square_stock = self.square_stock
     self.set_T(observation)
  
     list_prods = observation["products"]  # Danh sách sản phẩm
     list_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
     stocks = observation["stocks"]
     stock_idx, pos_x, pos_y = -1, None, None
     prod_size = [0, 0]


     for prod in list_prods:
        if prod["quantity"] > 0:
            prod_size = prod["size"][:]

            for i, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)

                for rotated in [False, True]:  
                    if rotated:
                        prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h :
                        if prod_h >= prod_w:
                         for x in range(stock_w - prod_w + 1):
                            
                            for y in range(stock_h - prod_h + 1):
                                
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    stock_idx = i
                             
                            if pos_x is not None:  
                                break
                        if prod_h < prod_w:
                          for y in range(stock_h - prod_h + 1):
                            
                            for x in range(stock_w - prod_w + 1):
                                
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    stock_idx = i
 
                            if pos_x is not None:  
                                break
                    
                    if pos_x is not None:
                       for k in range(i + 1, min (i + 4, len(stocks))):
                        
                        stock2 = stocks[k]
                        
                        stock_w2, stock_h2 = self._get_stock_size_(stock2)
                        if stock_w2 < prod_w or stock_h2 < prod_h:
                            continue
                        p = self.square(stocks, k, square_stock)/ (prod_h * prod_w)
                        DE = self.SA_help(stocks, square_stock, i , k)
                        
                        if prod_h >= prod_w:
                         for x1 in range(stock_w2 - prod_w + 1):
                            for y1 in range(stock_h2 - prod_h + 1):
                                if self._can_place_(stock2, (x1, y1), prod_size):
                                    if DE < 0 and -(DE)  > 2 * prod_h * prod_w :
                                        pos_x, pos_y = x1, y1
                                        stock_idx = k
                                        
                                        break
                                    if DE > 0 and np.exp(-DE/self.T) > 0.8 and p < 5:
                                        pos_x, pos_y = x1, y1
                                        stock_idx = k
                                        
                                        break
                                    break # xem xet co break hay ko
                            if stock_idx == k:
                                break
                        if prod_h < prod_w:
                         for y1 in range(stock_h2 - prod_h + 1):
                            for x1 in range(stock_w2 - prod_w + 1):
                                if self._can_place_(stock2, (x1, y1), prod_size):
                                    if DE < 0 and -(DE) > 2 * prod_h * prod_w :
                                        pos_x, pos_y = x1, y1
                                        stock_idx = k
                                        
                                        break
                                    if DE > 0 and np.exp(-DE/self.T) > 0.2 and p < 5:
                                        pos_x, pos_y = x1, y1
                                        stock_idx = k
                                        break
                                    break 
                            if stock_idx == k:
                                break
                        if stock_idx == k:
                            #m = 0
                            break
                       if stock_idx == k:
                          break
                    if pos_x is not None:  # Thoát khỏi vòng lặp xoay
                        break
                if pos_x is not None:  # Thoát khỏi vòng lặp stock
                    break
        self.T = 0.7 * self.T
        if self.T < 0:
            self.T = 1   
        if pos_x is not None:  
            break
     self.square_stock = square_stock

     return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


    # Student code here
    # You can add more functions if needed
