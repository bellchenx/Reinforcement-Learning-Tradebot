from config import INSTRUMENTS, TRADE_PAIRS

class Portfolio(object):
    """ Cryptocurrency portfolio object """

    def __init__(self,
                 num_of_instrumenet: int = len(INSTRUMENTS),
                 transaction_fee: float = 0.0005):
        self.n = num_of_instrumenet  # first n instrument
        self.instruments = INSTRUMENTS[0:self.n]  # instrument list
        self.pairs = TRADE_PAIRS[0:self.n]  # trade pair list

        self._fee = transaction_fee
        self._balance = [0.0 for i in range(
            self.n + 1)]  # initial balance list
        self._balance[self.n] = 1.  # initial USD factor
        self._net_worth_factor = 1.  # net worth factor

    def netWorth(self, initial_deposit: int = 1.):
        # print(self._net_worth_factor)
        return float(initial_deposit * self._net_worth_factor)

    def updatePorfolio(self, balance: list):
        if (self.n+1) != len(balance):
            AttributeError('Wrong balance array length.')
        self._balance = balance
    
    def applyFee(self, portion):
        self._net_worth_factor = self._net_worth_factor*(1.-self._fee * portion) 

    def updateNetWorth(self, factor: float):
        self._net_worth_factor = factor * self._net_worth_factor
        
