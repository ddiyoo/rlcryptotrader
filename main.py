import os
import sys
import logging
import argparse
import json
import pandas as pd

from quantylab.rltrader import settings
from quantylab.rltrader import utils
from quantylab.rltrader import data_manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4', 'v4.1', 'v4.2','crypto_v1'], default='crypto_v1')
    parser.add_argument('--name', default=utils.get_time_str())
    parser.add_argument('--crypto_symbol', nargs='+')
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'ppo', 'monkey'], default='a2c')
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='lstm')
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default='pytorch')
    # parser.add_argument('--start_date', default='20240101')
    # parser.add_argument('--end_date', default='20240110')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--initial_balance', type=int, default=100000)
    args = parser.parse_args()

    # 학습기 파라미터 설정
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}'
    learning = args.mode in ['train', 'update']
    reuse_models = args.mode in ['test', 'update', 'predict']
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.h5'
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.h5'
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = 100 if args.mode in ['train', 'update'] else 1
    num_steps = 60 if args.net in ['lstm', 'cnn'] else 1 # 1m :60, 5m: 60, 10m: 6

    # Backend 설정
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 생성
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)

    # 모델 경로 준비
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    value_network_path = os.path.join(settings.BASE_DIR, 'models', f'{value_network_name}.h5')
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', f'{policy_network_name}.h5')

    # 로그 기록 설정
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)
    
    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from quantylab.rltrader.learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner, PPOLearner

    common_params = {}
    list_crypto_symbol = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []

    for crypto_symbol in args.crypto_symbol:
        # 차트 데이터, 학습 데이터 준비

        



        # chart_data, training_data = data_manager.load_data(
        #     crypto_symbol, args.start_date, args.end_date, ver=args.ver)

        # assert len(chart_data) >= num_steps
        
        # # 최소/최대 단일 매매 금액 설정
        # min_trading_price = 10000
        # max_trading_price = 100000

        # #임시
        # leverage = 1

        # # 공통 파라미터 설정
        # common_params = {'rl_method': args.rl_method, 
        #     'net': args.net, 'num_steps': num_steps, 'lr': args.lr,
        #     'initial_balance': args.initial_balance, 'num_epoches': num_epoches, 
        #     'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,
        #     'output_path': output_path, 'reuse_models': reuse_models,'leverage':leverage}

        # # 강화학습 시작
        # learner = None
        # if args.rl_method != 'a3c':
        #     common_params.update({'crypto_symbol': crypto_symbol,
        #         'chart_data': chart_data, 
        #         'training_data': training_data,
        #         'min_trading_price': min_trading_price, 
        #         'max_trading_price': max_trading_price})
        #     if args.rl_method == 'dqn':
        #         learner = DQNLearner(**{**common_params, 
        #             'value_network_path': value_network_path})
        #     elif args.rl_method == 'pg':
        #         learner = PolicyGradientLearner(**{**common_params, 
        #             'policy_network_path': policy_network_path})
        #     elif args.rl_method == 'ac':
        #         learner = ActorCriticLearner(**{**common_params, 
        #             'value_network_path': value_network_path, 
        #             'policy_network_path': policy_network_path})
        #     elif args.rl_method == 'a2c':
        #         learner = A2CLearner(**{**common_params, 
        #             'value_network_path': value_network_path, 
        #             'policy_network_path': policy_network_path})
        #     elif args.rl_method == 'ppo':
        #         learner = PPOLearner(**{**common_params, 
        #             'value_network_path': value_network_path, 
        #             'policy_network_path': policy_network_path})
        #     elif args.rl_method == 'monkey':
        #         common_params['net'] = args.rl_method
        #         common_params['num_epoches'] = 10
        #         common_params['start_epsilon'] = 1
        #         learning = False
        #         learner = ReinforcementLearner(**common_params)
        # else:
        #     list_crypto_symbol.append(crypto_symbol)
        #     list_chart_data.append(chart_data)
        #     list_training_data.append(training_data)
        #     list_min_trading_price.append(min_trading_price)
        #     list_max_trading_price.append(max_trading_price)

        # if args.rl_method == 'a3c':
        #     learner = A3CLearner(**{
        #         **common_params, 
        #         'list_crypto_symbol': list_crypto_symbol, 
        #         'list_chart_data': list_chart_data, 
        #         'list_training_data': list_training_data,
        #         'list_min_trading_price': list_min_trading_price, 
        #         'list_max_trading_price': list_max_trading_price,
        #         'value_network_path': value_network_path, 
        #         'policy_network_path': policy_network_path})
        
        # assert learner is not None

        # if args.mode in ['train', 'test', 'update']:
        #     learner.run(learning=learning)
        #     if args.mode in ['train', 'update']:
        #         learner.save_models()
        # elif args.mode == 'predict':
        #     learner.predict()

        chart_data, training_data = data_manager.load_data(crypto_symbol, ver=args.ver)

        # 학습 데이터의 시작 시간과 종료 시간 설정
        # 학습 데이터의 시작 시간과 종료 시간 설정
        start_time = pd.to_datetime(chart_data.iloc[0]['time'])
        end_time = pd.to_datetime(chart_data.iloc[-1]['time'])

        # 8시간 간격으로 학습 및 모델 저장
        current_time = start_time
        while current_time <= end_time:
            # 현재 시간부터 8시간 후까지의 데이터 선택
            window_start_time_str = current_time.strftime('%Y%m%d%H%M%S')
            window_end_time = current_time + pd.Timedelta(hours=8)
            window_end_time_str = window_end_time.strftime('%Y%m%d%H%M%S')

            chart_data['time'] = pd.to_datetime(chart_data['time'])  # 'time' 컬럼을 datetime으로 변환
            curr_chart_data = chart_data[(chart_data['time'] >= current_time) & (chart_data['time'] < window_end_time)]
            curr_training_data = training_data.loc[curr_chart_data.index]

            # 최소/최대 단일 매매 금액 설정
            min_trading_price = 10000
            max_trading_price = 100000

            #임시
            leverage = 1

            # 공통 파라미터 설정
            common_params = {'rl_method': args.rl_method, 
                'net': args.net, 'num_steps': num_steps, 'lr': args.lr,
                'initial_balance': args.initial_balance, 'num_epoches': num_epoches, 
                'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,
                'output_path': output_path, 'reuse_models': reuse_models,'leverage':leverage}

            # 강화학습 시작
            learner = None

            if args.rl_method != 'a3c':
                common_params.update({'crypto_symbol': crypto_symbol,
                    'chart_data': chart_data, 
                    'training_data': training_data,
                    'min_trading_price': min_trading_price, 
                    'max_trading_price': max_trading_price})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                    'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ppo':
                learner = PPOLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                common_params['net'] = args.rl_method
                common_params['num_epoches'] = 10
                common_params['start_epsilon'] = 1
                learning = False
                learner = ReinforcementLearner(**common_params)

            else:
                list_crypto_symbol.append(crypto_symbol)
                list_chart_data.append(chart_data)
                list_training_data.append(training_data)
                list_min_trading_price.append(min_trading_price)
                list_max_trading_price.append(max_trading_price)

            if args.rl_method == 'a3c':
                learner = A3CLearner(**{
                    **common_params, 
                    'list_crypto_symbol': list_crypto_symbol, 
                    'list_chart_data': list_chart_data, 
                    'list_training_data': list_training_data,
                    'list_min_trading_price': list_min_trading_price, 
                    'list_max_trading_price': list_max_trading_price,
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            
            assert learner is not None

            # 학습 실행
            learner.run(learning=True)

            # 모델 저장
            model_save_name = f"{window_start_time_str}_{window_end_time_str}"
            learner.save_models(model_name=model_save_name)

            # 다음 윈도우로 이동
            current_time = window_end_time