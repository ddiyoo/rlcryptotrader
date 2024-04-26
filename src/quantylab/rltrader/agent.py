import numpy as np
import curses
from quantylab.rltrader import utils


class Agent:
    #environment에 추가되야할것
    #mark_price
    #fundingrate 컬럼 가져오기
    #get_funding_fee_time 구매한 시점에서 funding_fee 적용 시간 구하기?
    #get_time 현재시점
    
    #진입 symbol
    #maintenance_margin -> 여기에서는 max_leverage, maintenance_margin_rate와 maintenance_amount를 가져올 수 있음

    STATE_DIM = 2 #long, short, none    -> 이게 아니였음.. get_states


    TRADING_FEE_RATE = 0.0002
    Position_LEVERAGE = 1.0
    leverage = 1

    ACTION_LONG = 0
    ACTION_SHORT = 1
    ACTION_HOLD = 2
    ACTION_CLOSE = 3

    ACTIONS = [ACTION_LONG, ACTION_SHORT, ACTION_HOLD, ACTION_CLOSE]
    NUM_ACTIONS = len(ACTIONS)

    # position_status = [long, short, hold, none]


    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price, leverage,crypto_symbol):
        #현재 coin들 가격 가져오기 위해 환경 참조
        self.crypto_symbol = crypto_symbol
        self.environment = environment

        #initial_margin 금액의 최대 최소(강화학습 confidence와 결합해 결정)
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price
        
        #Agent 클래스의 속성
        self.initial_balance = initial_balance
        self.wallet_balance = self.initial_balance #trading_fee, funding_fee 차감계좌
        self.available_balance = self.initial_balance # 새로운 position open시 사용되는 계좌, initial_balance 에서 notional_position_value을 뺀 값
        self.margin_balance = self.initial_balance # open된 position의 미실현 PNL + wallet_balance
        # 목적에 따른 balance 재정리
        # wallet_balance 미실현 PNL 빼고 보여줌 -> 음..굳이따지자면,,수수료비용얼마나 나갔는지 확인?
        # available_balance 진입한 margin 빼고 보여줌 -> 다른 포지션 오픈시 사용(isolated일 경우)
        # margin_balance 전부 보여줌 -> 이걸로 portfoilo_value 하면 됨
        self.leverage = leverage
        
        

        self.num_long = 0
        self.num_short = 0
        self.num_long_hold = 0
        self.num_short_hold = 0
        self.num_close = 0
        self.num_long_liquidation = 0
        self.num_short_liquidation = 0
        self.num_hold = 0

        self.position_size = 0

        #Agent 클래스의 상태
        # POSITION_NONE = "none"
        # POSITION_LONG = "long"
        # POSITION_SHORT = "short"
        # self.position_status = Agent.POSITION_NONE  # 초기 상태는 포지션이 없는 상태
        self.position_status = 'none'

        
        
        
        self.open_size = 0
        self.entry_price = 0
        self.break_even_price = 0 #수수료 계산해서, 손해도 수익도 안보는 가격대 #써야하나? 흠..
        self.mark_price = environment.get_mark_price() # 만들어야함 # 강제청산계산위해

        self.leverage = leverage
        
        
        self.margin_ratio = 0 # position 비율
        self.margin = 0 # initial margin에서 업데이트
        self.margin_used = 0
        
        self.unrealized_pnl = 0 # unrealized profit and loss

        #
        self.TRADING_FEE_RATE = 0.0002
                     
        
        #포트폴리오 가치
        self.portfolio_value = self.initial_balance
        
        
        

    def reset(self):
        self.num_long = 0
        self.num_short = 0
        self.num_long_hold = 0
        self.num_short_hold = 0
        self.num_close = 0
        self.num_liquidation = 0
        
        self.open_size = 0
        self.entry_price = 0
        self.break_even_price = 0
        self.mark_price = self.environment.get_mark_price()
        
        
        self.margin_ratio = 0
        self.margin = 0
        
        self.unrealized_pnl = 0
        
        
        self.wallet_balance = self.initial_balance 
        self.available_balance = self.initial_balance 
        self.margin_balance = self.initial_balance 
        self.portfolio_value = self.initial_balance

    # def set_balance(self, inital_balance):
    #     self.initial_balance = self.inital_balance

    def get_states(self):
        if self.portfolio_value == 0:
            ratio_position = 0
        else:
            ratio_position = (self.position_size * self.environment.get_price()) / self.portfolio_value
        if self.entry_price != 0:
            unrealized_pnl = (self.environment.get_price() - self.entry_price) / self.entry_price
        else :
            unrealized_pnl = 0
        return (ratio_position, unrealized_pnl)

    
    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.
        
        pred = pred_policy
        if pred is None:
            pred = pred_value
        
        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1
        
            # if pred_policy is not None:
            #     if np.max(pred_policy) - np.min(pred_policy) < 0.05:
            #         epsilon = 1
        
        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)
        
        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])
        
        return action, confidence, exploration    

        
    #구매 unit
    def calculate_actual_tradable_amount(self, trading_price,crypto_symbol):  # trading_price는 USDT
        min_unit = self.environment.get_min_unit(crypto_symbol)  # csv 테이블에서 정보 가져옴 (0.001, 1 등)
        entry_price = self.environment.get_price()  # 현재 가격 가져오기
        
        # 최소 단위 수량 계산
        min_quantity = trading_price / (entry_price * min_unit)    
        # 최소 단위 수량에 맞춰 내림
        position_size = np.floor(min_quantity) * min_unit    
        # 실제 구매 가능한 포지션 금액 계산
        notional_position_value = entry_price * position_size

        # print(f"min_unit: {min_unit}, entry_price: {entry_price}, position_size: {position_size}  ") 
        if notional_position_value == 0: #금액으로 구매 불가일 경우
            return False
        return notional_position_value, entry_price, position_size
        
        
    
        
        

    #강화학습 confidence 값에 따라서 거래금액 을 정함
    def get_trading_amount_based_on_confidence(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_price
        added_trading_price = max(min(
            int(confidence * (self.max_trading_price - self.min_trading_price)),
            self.max_trading_price-self.min_trading_price), 0)
        trading_price = self.min_trading_price + added_trading_price
        # print(confidence,trading_price)

        return trading_price


    

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.
    
        pred = pred_policy
        if pred is None:
            pred = pred_value
    
        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1
    
            # if pred_policy is not None:
            #     if np.max(pred_policy) - np.min(pred_policy) < 0.05:
            #         epsilon = 1
    
        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)
    
        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        # print(action)
        return action, confidence, exploration

    #기본 설정: 단일 심볼, 단일배율
    #여러가지 버전을 생각해야할듯
    #isloated, cross
    #포지션 물타기 여부  
    #같은 심볼 양방향 포지션 거래 가능 버전(근데, 배율 없는 버전이라 의미없음)->지갑 알고리즘 다시확인필요

    #단일심볼, 단일배율, isolated, 물타기 불가
    def validate_action(self, action):
        # print(self.position_status)
        if action == Agent.ACTION_LONG:
            if self.position_status == 'long' or self.position_status == 'short':
                return False

        elif action == Agent.ACTION_SHORT:
            if self.position_status == 'long' or self.position_status == 'short':
                return False

        elif action == Agent.ACTION_HOLD:
            if self.position_status == 'none':
                return False

        elif action == Agent.ACTION_CLOSE:
            if self.position_status == 'none':
                return False

        return True

    def act(self, action, confidence):        
        # 환경에서 현재 가격 얻기
        mark_price = self.environment.get_mark_price()
        curr_price = self.environment.get_price()
        curr_time = self.environment.get_time()
        next_funding_fee_time = self.environment.get_next_funding_time(curr_time)

        if not self.validate_action(action):
            action = Agent.ACTION_HOLD
            #position hold 시 unrealized PNL을 reward로 return
            if self.position_status == 'long':
                unrealized_pnl = (curr_price - self.entry_price) * self.position_size
            elif self.position_status == 'short':
                unrealized_pnl = (self.entry_price - curr_price) * self.position_size
            else:
                unrealized_pnl = 0

            self.num_hold += 1
            mark_price = self.environment.get_mark_price()
            curr_price = self.environment.get_price()
            curr_time = self.environment.get_time()
            
            #만약 next_funding_fee_time을 넘었다면
            if next_funding_fee_time is not None: 
            
                if next_funding_fee_time is not None and curr_time > next_funding_fee_time:
                    funding_rate, fundingfee_price = self.environment.get_funding_rate()
                    fundingfee_notional_position_value = fundingfee_price * self.position_size
                    funding_fee = fundingfee_notional_position_value * funding_rate
                
                    if self.position_status == 'long':
                        self.wallet_balance -= funding_fee
                        self.margin_used -= funding_fee
                        self.margin_balance -= funding_fee
                        self.num_long_hold += 1 
                        
                        self.portfolio_value = self.margin_balance + unrealized_pnl
                        #print(f"Position :{self.position_status} holding, position_size: {self.position_size}, entry_price : {self.entry_price} ")

                        if the_maintenance_margin > self.margin_used:
                            self.num_long_liquidation += 1
                            self.wallet_balance = self.available_balance
                            self.margin_balance = self.available_balance
                            self.margin_used = 0
                            self.position_status = 'none'
                            print("Long position, Liquidation!")
                    elif self.position_status == 'short':
                        self.wallet_balance += funding_fee
                        self.margin_used += funding_fee
                        self.margin_balance += funding_fee
                        self.num_short_hold += 1 
                        

                        self.portfolio_value = self.margin_balance + unrealized_pnl
                        #print(f"Position :{self.position_status} holding, position_size: {self.position_size}, entry_price : {self.entry_price} ")
                        
                        if the_maintenance_margin > self.margin_used:
                            self.num_short_liquidation += 1 
                            self.wallet_balance = self.available_balance
                            self.margin_balance = self.available_balance
                            self.margin_used = 0
                            self.position_status = 'none'
                            #print("Short position, Liquidation!")
            else:
                

                #포트폴리오 가치 갱신            
                self.portfolio_value = self.margin_balance
                #print(f"Position :{self.position_status} holding, position_size: {self.position_size}, entry_price : {self.entry_price} ")
            
            
            return unrealized_pnl               
    
        # Long position open
        if action == Agent.ACTION_LONG:
            self.num_long += 1 
            
            # position 
            self.trading_price = self.get_trading_amount_based_on_confidence(confidence)
                            
            # 명목포지션가치(trading_fee, funding_fee 적용 기준)
            self.notional_position_value, self.entry_price, self.position_size = self.calculate_actual_tradable_amount(self.trading_price, self.crypto_symbol)
            
            

            # 초기 마진금
            self.trading_fee = (self.notional_position_value / self.leverage) * (self.TRADING_FEE_RATE)
            
            initial_margin = self.notional_position_value - self.trading_fee
            self.margin_used = initial_margin

            maintenance_margin_rate = self.environment.get_maintenance_info(self.notional_position_value)[1]
            maintenance_amount = self.environment.get_maintenance_info(self.notional_position_value)[2]
            
            the_maintenance_margin = (self.notional_position_value * maintenance_margin_rate) - maintenance_margin_rate
            # 마진 유지 증거금
            
            #initial_balance 에서 주문 넣은 금액 : notional_position_value
            #initial_balance 에서 주문 넣은 금액 수수료 차감 : initial_margin
            self.wallet_balance -= self.trading_fee  # Unrealized PNL 발생전까지 margin_balance랑 같음
            self.available_balance -= self.notional_position_value # 새로운 position open시 사용되는 계좌, initial_balance 에서 notional_position_value을 뺀 값
            self.margin_balance -= self.trading_fee # Unrealized PNL 발생전

            #chart_data 보고 포지션 오픈시점에서 다음 펀딩비 계산시점 계산
            next_funding_fee_time = self.environment.get_funding_fee_time()

            self.position_status = 'long'
            #print(f"Position :{self.position_status} open, position_size: {self.position_size}, entry_price : {self.entry_price} ")
            return 0
            # return -self.trading_fee
    
        # Short position open
        elif action == Agent.ACTION_SHORT:
            self.num_short += 1 
            
            #position
            self.trading_price = self.get_trading_amount_based_on_confidence(confidence)
                            
            self.notional_position_value, self.entry_price, self.position_size = self.calculate_actual_tradable_amount(self.trading_price, self.crypto_symbol)
            
            
    
            
            self.trading_fee = (self.notional_position_value / self.leverage) * (self.TRADING_FEE_RATE)

            #isolated는 증거금에서 까이니 사실상 위의 open_position_size 만큼 사는게 아니게됨 계산용..
            initial_margin = self.notional_position_value - self.trading_fee
            self.margin_used = initial_margin
            
            maintenance_info = self.environment.get_maintenance_info(self.notional_position_value)
            the_maintenance_margin = (self.notional_position_value * maintenance_info[1]) - maintenance_info[2]
            

            # the_maintenance_margin = (notional_position_value * self.environment.get_maintenance_info()[1]) - self.environment.get_maintenance_info()[2]

            # margin = initial_margin

            self.wallet_balance -= self.trading_fee  # Unrealized PNL 발생전까지 margin_balance랑 같음, isolated 일 경우 margin 이랑 같음
            self.available_balance -= self.notional_position_value # 새로운 position open시 사용되는 계좌, initial_balance 에서 notional_position_value을 뺀 값
            self.margin_balance -= self.trading_fee # Unrealized PNL 발생전

            #chart_data 보고 포지션 오픈시점에서 다음 펀딩비 계산시점 계산
            next_funding_fee_time = self.environment.get_funding_fee_time()

            self.position_status = 'short'
            #print(f"Position: {self.position_status} open, position_size: {self.position_size}, entry_price : {self.entry_price} ")
            return 0
            # return -self.trading_fee
        
            # 관망
        if action == Agent.ACTION_HOLD:
            self.num_hold += 1
            mark_price = self.environment.get_mark_price()
            curr_price = self.environment.get_price()
            curr_time = self.environment.get_time()

            #position hold 시 unrealized PNL을 reward로 return
            if self.position_status == 'long':
                unrealized_pnl = (curr_price - self.entry_price) * self.position_size
                 
            elif self.position_status == 'short':
                unrealized_pnl = (self.entry_price - curr_price) * self.position_size
                
            else:
                unrealized_pnl = 0
            
            
            
            #만약 next_funding_fee_time을 넘었다면
            if next_funding_fee_time is not None: 
            
                if next_funding_fee_time is not None and curr_time > next_funding_fee_time:
                    funding_rate, fundingfee_price = self.environment.get_funding_rate()
                    fundingfee_notional_position_value = fundingfee_price * self.position_size
                    funding_fee = fundingfee_notional_position_value * funding_rate
                
                    if self.position_status == 'long':
                        self.wallet_balance -= funding_fee
                        self.margin_used -= funding_fee
                        self.margin_balance -= funding_fee
                        self.num_long_hold += 1 
                        
                        self.portfolio_value = self.margin_balance + unrealized_pnl
                        #print(f"Position :{self.position_status} holding, position_size: {self.position_size}, entry_price : {self.entry_price} ")

                        if the_maintenance_margin > self.margin_used:
                            self.num_long_liquidation += 1
                            self.wallet_balance = self.available_balance
                            self.margin_balance = self.available_balance
                            self.margin_used = 0
                            self.position_status = 'none'
                            #print("Long position, Liquidation!")
                    elif self.position_status == 'short':
                        self.wallet_balance += funding_fee
                        self.margin_used += funding_fee
                        self.margin_balance += funding_fee
                        self.num_short_hold += 1 
                        

                        self.portfolio_value = self.margin_balance + unrealized_pnl
                        #print(f"Position :{self.position_status} holding, position_size: {self.position_size}, entry_price : {self.entry_price} ")
                        
                        if the_maintenance_margin > self.margin_used:
                            self.num_short_liquidation += 1 
                            self.wallet_balance = self.available_balance
                            self.margin_balance = self.available_balance
                            self.margin_used = 0
                            self.position_status = 'none'
                            #print("Short position, Liquidation!")
            else:
                

                #포트폴리오 가치 갱신            
                self.portfolio_value = self.margin_balance + unrealized_pnl
                #print(f"Position :{self.position_status} holding, position_size: {self.position_size}, entry_price : {self.entry_price} ")
            return unrealized_pnl

        elif action == Agent.ACTION_CLOSE:
            self.num_close += 1

            close_price = self.environment.get_price()
            curr_time = self.environment.get_time()

            

            # 펀딩 수수료 적용 여부 확인
            if next_funding_fee_time is not None and curr_time > next_funding_fee_time:
                funding_rate, fundingfee_price = self.environment.get_funding_rate()
                fundingfee_notional_position_value = fundingfee_price * self.position_size
                funding_fee = fundingfee_notional_position_value * funding_rate

                if self.position_status == 'long':
                    self.wallet_balance -= funding_fee
                    self.margin_used -= funding_fee
                    self.margin_balance -= funding_fee
                elif self.position_status == 'short':
                    self.wallet_balance += funding_fee
                    self.margin_used += funding_fee
                    self.margin_balance += funding_fee

           
            # 이미 저장된 포지션 수량을 사용하여 청산 가치를 계산합니다.
            close_notional_position_value = close_price * self.position_size
            #tradingfee 계산
            self.trading_fee = (close_notional_position_value / self.leverage) * self.TRADING_FEE_RATE
            

            # 포지션 청산으로 인한 수익 또는 손실을 계산합니다.
            if self.position_status == 'long':
                realized_pnl = (close_price - self.entry_price) * self.position_size - self.trading_fee
            elif self.position_status == 'short':
                realized_pnl = (self.entry_price - close_price) * self.position_size - self.trading_fee


            # realized_pnl = (close_price - self.entry_price) * self.position_size
            # profit_or_loss = close_notional_position_value - (self.entry_price * self.position_size)
            
            # 지갑 및 마진 잔액을 업데이트합니다.
            self.margin_balance += realized_pnl

            self.wallet_balance = self.margin_balance        # Unrealized PNL 갱신             
            self.available_balance = self.margin_balance
                        
            # 청산 이후의 포트폴리오 가치를 갱신합니다.
            self.portfolio_value = self.margin_balance
            
            #print(f"{self.position_status} Position closed. available_balance:{self.available_balance}, initial_balance: {self.initial_balance}, margin_balance: {self.margin_balance}, Profit/Loss: {self.margin_balance-self.initial_balance}.")
            #print(f"long: {self.num_long}, long hold: {self.num_long_hold}, short: {self.num_short}, short hold: {self.num_short_hold}")
            
            #position close시 수익률 보상으로 return
            if self.entry_price > 0:
                return_rate = (close_price - self.entry_price) / self.entry_price
            else:
                return_rate = 0
            
            self.position_status = 'none'
            

            self.position_size = 0
            self.entry_price = 0
            self.margin_used = 0
            self.notional_position_value = 0  # 포지션 가치 초기화

            # print(f"ror:{return_rate}")
            return return_rate
            #symbol: {}, size: {}, entry_price: {}, unrealizedPNL: {},           
            # 포지션 청산 후 포지션 관련 변수 초기화
            
             