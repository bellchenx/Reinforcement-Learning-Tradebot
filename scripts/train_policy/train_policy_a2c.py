from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv

from config import LOG_INTERVAL, N_BATCHES, N_STEPS, TRAIN_DATA_RANGE
from environments.simulated_trade_environment import \
    SimulatedTradingEnvironment
from tradebot.a2c_tradebot_multiscale_ohlc import A2CPolicy

env = DummyVecEnv([lambda: SimulatedTradingEnvironment(debug=True)])

model = A2C(A2CPolicy, env, verbose=1)
model.learn(total_timesteps=N_STEPS, log_interval=LOG_INTERVAL)
model.save("a2c_tradebot_multiscale_ohlc")
