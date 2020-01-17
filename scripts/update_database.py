from config import INTERVALS, TRADE_PAIRS
from data.get_trade_data import GetTradeData

pairs = TRADE_PAIRS
timezone = 'America/New_York'

for pair in pairs:
    downloader = GetTradeData(folder='data', pair=pair)
    downloader.download_trade_data(0)
    downloader.agg_ohlc(0, [1])
