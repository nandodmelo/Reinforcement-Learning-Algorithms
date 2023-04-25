import Environment
from binance.client import Client
from DataGenerator import Data_Generator
import numpy as np

import time


api_key = "zSO4fSqoc2GigjvOnDqy0GHbRqX10DLeDKer5kDG8g73qx2eqpuVpnQUYV5Ns8UY"
api_secret = "5yc8vMJa1lewkj6l2y3Q20TqKbxzs46ZWCChZ1UB4XL7eMIf8yULkDLylVvgDiC9"

client=Client(api_key,api_secret)


timestamp = client._get_earliest_valid_timestamp('XRPUSDT', '1d')



a = Data_Generator(['ETHUSDT','BTCUSDT','BNBUSDT','XRPUSDT'], '1d', client, timestamp)
df = a.GetStocks(indicators=True)
xa = a.formatData(df, False)

#print('SHAPEZER1', x)
x_norma = np.copy(xa)
#print('CCC',x_norma, x_norma.shape)
x_norm = a.normalize(np.copy(x_norma))



lookback_window_size = 50

train_df = xa
train_df_norm = x_norm

env  = Environment.PortEnv(train_df, cash =10000, assets = ['ETHUSDT','BTCUSDT','BNBUSDT','XRPUSDT'])




def Random_games(env, train_episodes = 50, training_batch_size=500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)
        #print('episodio',episode)
        while True:
            #env.render()

            action = np.random.uniform(-1,1,4)

            state, reward, done, _ = env.step(action)

            if env.start_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/(episode+1))



start = time.time()
Random_games(env, train_episodes = 10000, training_batch_size=500)

end = time.time()

print('Tempo gasto', end-start)
#307.3961067199707