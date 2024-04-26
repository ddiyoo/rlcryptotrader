import os
import pandas as pd
import numpy as np

from quantylab.rltrader import settings


# COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']
COLUMNS_CHART_DATA = ['time', 'open', 'high', 'low', 'close', 'volume','fundingrate']

#mp_ratio(괴리율) 계산
COLUMNS_TRAINING_DATA_V1 = [
    'BANDb', 'BANDWidth', 'CCI', 'CO', 'DMI', 'EOM', 'MACD_OSC', 'MI', 'Momentum', 'NCO', 'PO', 'ROC', 'RSI', 'SMI',
    'Fast Stochastic', 'TRIX', 'VO', 'VROC', 'Williams', 'ZScore','open_marketbasis' , 'high_marketbasis' , 'low_marketbasis' , 'close_marketbasis'
]





def load_data(crypto_symbol,  ver='v2'):
    if ver in ['v1', 'v1.1', 'v2']:
        return load_data_v1_v2(crypto_symbol,  ver)
    elif ver in ['crypto_v1']:
        return load_data_crypto_v1(crypto_symbol, ver)
    elif ver in ['v3', 'v4']:
        return load_data_v3_v4(crypto_symbol,  ver)
    elif ver in ['v4.1', 'v4.2']:
        crypto_filename = ''
        market_filename = ''
        data_dir = os.path.join(settings.BASE_DIR, 'data', 'v4.1')
        for filename in os.listdir(data_dir):
            if crypto_symbol in filename:
                crypto_filename = filename
            elif 'market' in filename:
                market_filename = filename
        
        chart_data, training_data = load_data_v4_1(
            os.path.join(data_dir, crypto_filename),
            os.path.join(data_dir, market_filename)
            
        )
        if ver == 'v4.1':
            return chart_data, training_data
        
        tips_filename = ''
        taylor_us_filename = ''
        data_dir = os.path.join(settings.BASE_DIR, 'data', 'v4.2')
        for filename in os.listdir(data_dir):
            if filename.startswith('tips'):
                tips_filename = filename
            if filename.startswith('taylor_us'):
                taylor_us_filename = filename
        return load_data_v4_2(
            pd.concat([chart_data, training_data], axis=1),
            os.path.join(data_dir, tips_filename),
            os.path.join(data_dir, taylor_us_filename)
        )


def load_data_v1_v2(crypto_symbol,  ver):
    header = None if ver == 'v1' else 0
    df = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, f'{crypto_symbol}.csv'),
        thousands=',', header=header, converters={'date': lambda x: str(x)})

    if ver == 'v1':
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # 데이터 전처리
    df = preprocess(df)
    
    # 기간 필터링
    # df['date'] = df['date'].str.replace('-', '')
    # df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    # df = df.fillna(method='ffill').reset_index(drop=True)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = None
    if ver == 'v1':
        training_data = df[COLUMNS_TRAINING_DATA_V1]
    elif ver == 'v1.1':
        training_data = df[COLUMNS_TRAINING_DATA_V1_1]
    elif ver == 'v2':
        df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = df[COLUMNS_TRAINING_DATA_V2]
        training_data = training_data.apply(np.tanh)
    else:
        raise Exception('Invalid version.')
    
    return chart_data, training_data.values

def load_data_crypto_v1(crypto_symbol, ver):
    data_path = os.path.join(settings.BASE_DIR, 'data', ver, f'{crypto_symbol}_2024_1m.csv')
    
    # 데이터 파일 존재 여부 확인
    if not os.path.exists(data_path):
        raise ValueError(f"Data file not found at: {data_path}")
    
    # 데이터 파일 읽어오기
    df = pd.read_csv(data_path, thousands=',', header=0, converters={'time': lambda x: str(x)})
    
    df = df.sort_values(by='time').reset_index(drop=True)
    
    chart_data = df[COLUMNS_CHART_DATA]
    
    # 학습 데이터 분리 (DataFrame으로 반환)
    training_data = df[COLUMNS_TRAINING_DATA_V1]
    
    return chart_data, training_data

def generate_sliding_windows(df, window_hours=8):
    # 'time' 컬럼을 datetime으로 변환 (이미 되어 있다면 생략)
    df['time'] = pd.to_datetime(df['time'])

    # 데이터 초기화
    training_data = []
    validation_data = []

    # 데이터셋 전체에 대해 슬라이딩 윈도우 적용
    total_hours = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / 3600
    num_windows = int(total_hours / window_hours) - 1  # 전체 윈도우 수 계산

    for window in range(num_windows):
        # 현재 윈도우의 시작 시간
        window_start_time = df['time'].iloc[0] + pd.Timedelta(hours=window * window_hours)
        
        # 학습 데이터 윈도우 설정
        train_end_time = window_start_time + pd.Timedelta(hours=window_hours)
        train_df = df[(df['time'] >= window_start_time) & (df['time'] < train_end_time)]
        
        # 검증(또는 테스트) 데이터 윈도우 설정
        validation_end_time = train_end_time + pd.Timedelta(hours=window_hours)
        validation_df = df[(df['time'] >= train_end_time) & (df['time'] < validation_end_time)]
        
        training_data.append(train_df)
        validation_data.append(validation_df)

    return training_data, validation_data



def load_data_v3_v4(crypto_symbol,  ver):
    columns = None
    if ver == 'v3':
        columns = COLUMNS_TRAINING_DATA_V3
    elif ver == 'v4':
        columns = COLUMNS_TRAINING_DATA_V4

    # 시장 데이터
    df_marketfeatures = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, 'marketfeatures.csv'), 
        thousands=',', header=0, converters={'date': lambda x: str(x)})
    
    # 종목 데이터
    df_cryptofeatures = None
    for filename in os.listdir(os.path.join(settings.BASE_DIR, 'data', ver)):
        if filename.startswith(crypto_symbol):
            df_cryptofeatures = pd.read_csv(
                os.path.join(settings.BASE_DIR, 'data', ver, filename), 
                thousands=',', header=0, converters={'date': lambda x: str(x)})
            break

    # 시장 데이터와 종목 데이터 합치기
    df = pd.merge(df_cryptofeatures, df_marketfeatures, on='date', how='left', suffixes=('', '_dup'))
    df = df.drop(df.filter(regex='_dup$').columns.tolist(), axis=1)

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # NaN 처리
    df = df.ffill().bfill().reset_index(drop=True)
    df = df.fillna(0)

    # 기간 필터링
    # df['date'] = df['date'].str.replace('-', '')
    # df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    # df = df.reset_index(drop=True)

    # 데이터 조정
    if ver == 'v3':
        df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = df[columns].values

    # 스케일링
    if ver == 'v4':
        from sklearn.preprocessing import RobustScaler
        from joblib import dump, load
        scaler_path = os.path.join(settings.BASE_DIR, 'scalers', f'scaler_{ver}.joblib')
        scaler = None
        if not os.path.exists(scaler_path):
            scaler = RobustScaler()
            scaler.fit(training_data)
            dump(scaler, scaler_path)
        else:
            scaler = load(scaler_path)
        training_data = scaler.transform(training_data)

    return chart_data, training_data


def load_data_v4_1(stock_data_path, market_data_path, date_from, date_to):
    df_stock = None
    if stock_data_path.endswith('.csv'):
        df_stock = pd.read_csv(stock_data_path, dtype={'date': str})
    elif stock_data_path.endswith('.json'):
        import json
        with open(stock_data_path) as f:
            df_stock = pd.DataFrame(**json.load(f))
    df_market = None
    if market_data_path.endswith('.csv'):
        df_market = pd.read_csv(market_data_path, dtype={'date': str})
    elif market_data_path.endswith('.json'):
        import json
        with open(market_data_path) as f:
            df_market = pd.DataFrame(**json.load(f))
    df = pd.merge(df_stock, df_market, on='date', how='left')
    # df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df[COLUMNS_CHART_DATA + COLUMNS_TRAINING_DATA_V4_1]
    df = df.fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
    df = df.fillna(0)
    return df[COLUMNS_CHART_DATA], df[COLUMNS_TRAINING_DATA_V4_1].values


def load_data_v4_2(df_v4_1, stock_data_path, market_data_path):
    df_tips = None
    if stock_data_path.endswith('.csv'):
        df_tips = pd.read_csv(stock_data_path, dtype={'date': str})
    elif stock_data_path.endswith('.json'):
        import json
        with open(stock_data_path) as f:
            df_tips = pd.DataFrame(**json.load(f))
    df_taylor_us = None
    if market_data_path.endswith('.csv'):
        df_taylor_us = pd.read_csv(market_data_path, dtype={'date': str})
    elif market_data_path.endswith('.json'):
        import json
        with open(market_data_path) as f:
            df_taylor_us = pd.DataFrame(**json.load(f))
    df = pd.merge(df_v4_1, df_tips.rename(columns={'value': 'tips'}), on='date', how='left')
    df = pd.merge(df, df_taylor_us.rename(columns={'taylor': 'taylor_us'}), on='date', how='left')
    df[['tips', 'taylor_us']] = df[['tips', 'taylor_us']] / 100
    COLUMNS_TRAINING_DATA = COLUMNS_TRAINING_DATA_V4_1 + ['tips', 'taylor_us']
    df = df[COLUMNS_CHART_DATA + COLUMNS_TRAINING_DATA]
    df = df.fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
    df = df.fillna(0)
    return df[COLUMNS_CHART_DATA], df[COLUMNS_TRAINING_DATA].values

def generate_sliding_windows(df, window_hours=8):
    # 'time' 컬럼을 datetime으로 변환 (이미 되어 있다면 생략)
    df['time'] = pd.to_datetime(df['time'])

    # 데이터 초기화
    training_data = []
    validation_data = []

    # 데이터셋 전체에 대해 슬라이딩 윈도우 적용
    total_hours = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / 3600
    num_windows = int(total_hours / window_hours) - 1  # 전체 윈도우 수 계산

    for window in range(num_windows):
        # 현재 윈도우의 시작 시간
        window_start_time = df['time'].iloc[0] + pd.Timedelta(hours=window * window_hours)
        
        # 학습 데이터 윈도우 설정
        train_end_time = window_start_time + pd.Timedelta(hours=window_hours)
        train_df = df[(df['time'] >= window_start_time) & (df['time'] < train_end_time)]
        
        # 검증(또는 테스트) 데이터 윈도우 설정
        validation_end_time = train_end_time + pd.Timedelta(hours=window_hours)
        validation_df = df[(df['time'] >= train_end_time) & (df['time'] < validation_end_time)]
        
        training_data.append(train_df)
        validation_data.append(validation_df)

    return training_data, validation_data