from utilities.functions import str2time

# Database Configuration
INSTRUMENTS = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH']
TRADE_PAIRS = ['XXBTZUSD', 'XETHZUSD', 'XXRPZUSD', 'XLTCZUSD', 'BCHUSD']
INTERVALS = [1, 5, 60]
ENTRIES_PER_FILE = 1024
RESAMPLE_THREAD = 3

# Environment Configuration
TRADE_INTERVAL = 1 * 60
WINDOW = 16
TRANSACTION_FEE = 0.0005
MAX_STEP_PERCENT = 0.5
LEVERAGE = 5
train_data_range_str = ('2017-9-1 8:00', '2019-8-31 8:00')
backtest_range_str = ('2019-9-1 9:00','2019-11-30 9:00')

# Training Configuration
N_STEPS = 50000
N_BATCHES = 5
LOG_INTERVAL = 10

# Other
TOTAL_NUM_OF_INSTRUMENTS = len(INSTRUMENTS)
INTERVALS_REV = INTERVALS
INTERVALS_REV.reverse()
TRAIN_DATA_RANGE = (str2time(train_data_range_str[0], integer=True), str2time(train_data_range_str[1], integer=True))
BACKTEST_RANGE = (str2time(backtest_range_str[0], integer=True), str2time(backtest_range_str[1], integer=True))

