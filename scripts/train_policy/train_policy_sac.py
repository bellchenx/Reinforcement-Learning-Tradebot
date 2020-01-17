#TODO Incomplete

from stable_baselines import SAC
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac.policies import MlpPolicy

from config import LOG_INTERVAL, N_BATCHES, N_STEPS, TRAIN_DATA_RANGE
from environments.simulated_trade_environment import \
    SimulatedTradingEnvironment
from tradebot.sac_tradebot_multiscale_ohlc import SoftActorCriticsPolicy

env = SimulatedTradingEnvironment(debug=True)

model = SAC(SoftActorCriticsPolicy, env, verbose=1, policy_kwargs={})
model.learn(total_timesteps=N_STEPS, log_interval=LOG_INTERVAL)
model.save("sac_tradebot_multiscale_ohlc")
