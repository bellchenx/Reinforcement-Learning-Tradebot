from stable_baselines import A2C

from environments.simulated_trade_environment import \
    SimulatedTradingEnvironment
from tradebot.sac_tradebot_multiscale_ohlc import SoftActorCriticsPolicy
from config import BACKTEST_RANGE

env = SimulatedTradingEnvironment(backtest=True)
                                  
model = A2C.load("a2c_tradebot_multiscale_ohlc")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
