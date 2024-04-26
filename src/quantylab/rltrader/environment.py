import pandas as pd
import os
from pandas import to_datetime
from quantylab.rltrader import settings



class Environment:
    

    # environment에 추가되야할것
    # mark_price
    # fundingrate 컬럼 가져오기
    # get_funding_fee_time 구매한 시점에서 funding_fee 적용 시간 구하기?
    # get_time 현재시점

    # 진입 symbol
    # maintenance_margin -> 여기에서는 max_leverage, maintenance_margin_rate와 maintenance_amount를 가져올 수 있음

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.chart_data['time'] = to_datetime(self.chart_data['time'])
        self.observation = None
        self.idx = -1
       
        

    def reset(self):
        self.observation = None
        self.idx = -1
        

    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation['close']
        return None

    def get_mark_price(self):
        # 우선은 이렇게
        if self.observation is not None:
            return self.observation['close']
        return None

    def get_funding_rate(self, time):
        if self.observation is not None:
            funding_rate = self.chart_data[self.chart_data['time'] == time]['fundingrate'].values
            if len(funding_rate) > 0:
                fundingfee_price = self.chart_data[self.chart_data['time'] == time]['close'].values
                return funding_rate[0], fundingfee_price[0]
        return None

    def get_next_funding_time(self, curr_time):
        # curr_time이 문자열 타입이라면, datetime 타입으로 변환
        curr_time = pd.to_datetime(curr_time)

        future_data = self.chart_data[
            (self.chart_data['time'] > curr_time) & (self.chart_data['fundingrate'].notnull())]
        
        if len(future_data) > 0:
            next_funding_time = future_data['time'].iloc[0]
            return next_funding_time
        return None

    #임시
    def get_min_unit(self, crypto_symbol):
        data_path = os.path.join(settings.BASE_DIR, 'data', 'crypto_v1', 'min_trade_amount_value.csv')

        df = pd.read_csv(data_path)
        
        # notional_position_value 값이 속하는 행 찾기
        min_amount = df.loc[df['symbol'] == crypto_symbol, 'min_trade_amount'].item() 
        
        return min_amount


    # def get_time(self):
    #     if self.observation is not None:
    #         return self.observation.iloc[self.idx]
    #     return None

    def get_time(self):
        # self.observation이 Series 객체인지 확인
        if isinstance(self.observation, pd.Series):
            # Series 객체라면, 직접 'time' 값을 반환합니다.
            return self.observation['time']
        elif self.observation is not None and 'time' in self.observation.columns:
            # DataFrame 객체라면, 현재 idx에 해당하는 'time' 값을 반환합니다.
            return self.observation['time'].iloc[self.idx]
        return None

    def get_maintenance_info(self, notional_position_value):
        data_path = os.path.join(settings.BASE_DIR, 'data', 'crypto_v1', 'leverage_n_margin.csv')
        df = pd.read_csv(data_path)
        # 데이터프레임 df의 관련 컬럼을 숫자 타입으로 변환
        df['over_notional_value'] = pd.to_numeric(df['over_notional_value'], errors='coerce')
        df['notional_value_and_under'] = pd.to_numeric(df['notional_value_and_under'], errors='coerce')

        # 여기에서 notional_position_value는 이미 숫자 타입이라고 가정
        row = df[(df['over_notional_value'] < notional_position_value) & (df['notional_value_and_under'] >= notional_position_value)]
        if row.empty:
            raise ValueError(f"No matching row found for notional_position_value: {notional_position_value}")
        
        # 결과 추출
        max_leverage = row['max_leverage'].values[0]
        maintenance_margin_rate = row['maintenance_margin_rate'].values[0]
        maintenance_amount = row['maintenance_amount'].values[0]
        
        return max_leverage, maintenance_margin_rate, maintenance_amount
        


    # def get_maintenance_info(self, notional_position_value):
    #     data_path = os.path.join(settings.BASE_DIR, 'data', 'crypto_v1', 'leverage_n_margin.csv')
        
    #     # CSV 파일 읽어오기
    #     df = pd.read_csv(data_path)
        
    #     # notional_position_value 값이 속하는 행 찾기
    #     row = df[(df['over_notional_value'] < notional_position_value) & (df['notional_value_and_under'] >= notional_position_value)]
        
    #     # 해당 행이 없는 경우 (notional_position_value가 CSV 파일의 범위를 벗어난 경우)
    #     if row.empty:
    #         raise ValueError(f"No matching row found for notional_position_value: {notional_position_value}")
        
    #     # 결과 추출
    #     max_leverage = row['max_leverage'].values[0]
    #     maintenance_margin_rate = row['maintenance_margin_rate'].values[0]
    #     maintenance_amount = row['maintenance_amount'].values[0]
        
    #     return max_leverage, maintenance_margin_rate, maintenance_amount

    



    def get_funding_fee_time(self):
            if self.observation is not None and 'funding_rate' in self.chart_data.columns:
                # 현재 관찰중인 행의 시간 가져오기
                current_time = self.observation['time']

                # 현재 행 이후로 funding_rate가 있는 첫 번째 행 찾기
                subsequent_rows_with_funding = self.chart_data.loc[self.idx:].dropna(subset=['fundingrate'])

                if not subsequent_rows_with_funding.empty:
                    # 첫 번째 funding_rate 행의 시간 가져오기
                    first_funding_time = subsequent_rows_with_funding.iloc[0]['time']

                    # 시간 차이 계산 (Pandas Timestamp로 변환)
                    time_difference = pd.Timestamp(first_funding_time) - pd.Timestamp(current_time)

                    return time_difference.total_seconds() / 3600  # 시간 단위로 반환
            return None

    def get_next_funding_time(self,curr_time):
            # 현재 idx 기준으로 다음 funding_rate가 있는 행 찾기
            # next_funding_idx = self.chart_data.loc[self.idx+1:, 'fundingrate'].first_valid_index()
            # if next_funding_idx is not None:
            #     return self.chart_data.loc[next_funding_idx, 'time']
            # return None


            current_idx = self.chart_data[self.chart_data['time'] == curr_time].index[0]
            next_funding_idx = self.chart_data.loc[current_idx+1:, 'fundingrate'].first_valid_index()
            if next_funding_idx is not None:
                return self.chart_data.loc[next_funding_idx, 'time']
            return None