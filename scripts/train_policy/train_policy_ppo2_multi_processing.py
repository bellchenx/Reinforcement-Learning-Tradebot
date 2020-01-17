from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv

from config import TRAIN_DATA_RANGE
from environments.simulated_trade_environment import \
    SimulatedTradingEnvironment
from tradebot.ppo_tradebot_multiscale_ohlc import PPO2Policy


def make_env(rank, seed=0):
    def _init():
        env = SimulatedTradingEnvironment(backtest=False)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def main():
    num_cpu = 1
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    model = PPO2(PPO2Policy, env, verbose=0, tensorboard_log='log', learning_rate=1e-4)
    model.learn(total_timesteps=500000, log_interval=10)
    model.save("ppo2_tradebot_multiscale_ohlc_2")

if __name__ == '__main__':
    main()
