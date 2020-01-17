import argparse
import os
import time
from pathlib import Path

import krakenex
import pandas as pd
import pytz
from pykrakenapi import KrakenAPI


class GetTradeData(object):

    def __init__(self, pair='XXBTZUSD', timezone='America/New_York', folder='data'):

        # initiate api
        api = krakenex.API()
        self.k = KrakenAPI(api, tier=0, retry=.1)

        # set pair
        self.pair = pair
        self.tz = pytz.timezone(timezone)

        # set and create folder
        self.folder = folder
        os.makedirs(os.path.join(self.folder, pair), exist_ok=True)

    def download_trade_data(self, since):

        folder = os.path.join(self.folder, self.pair)

        # update or new download
        if since is 0:
            fs = os.listdir(folder)
            if len(fs) > 0:
                fs.sort()
                last = int(fs[-1].split('.')[0])
            else:
                last = 0
        else:
            last = since

        # get data
        while True:
            try:
                time.sleep(0.75)
                fname = os.path.join(
                    folder, '{}.pickle'.format(str(last).zfill(19)))
                trades, last = self.k.get_recent_trades(pair=self.pair,
                                                        since=last)

                # set timezone
                index = trades.index.tz_localize(pytz.utc).tz_convert(self.tz)
                trades.index = index

                # store
                print('Data Downloader: storing', fname)
                trades.to_pickle(fname)

            except AttributeError:  # can't tz_localize on empty (last) trades
                print('Data Downloader: download/update finished!')
                break

    def agg_ohlc(self, since, interval=[0]):

        folder = os.path.join(self.folder, self.pair)
        if interval == [0]:
            from config import INTERVALS
            interval = INTERVALS

        # fetch files and convert to dataframe
        fs = os.listdir(folder)
        fs.sort(reverse=True)
        if since > 0:
            fs = [f for f in fs if int(f.split('.')[0]) >= since*1e9]

        trades = []
        for f in fs:
            print(f)
            if f[0] == '.':
                continue
            trades.append(pd.read_pickle(os.path.join(folder, f)))
        trades = pd.concat(trades, axis=0)
        trades.loc[:, 'cost'] = trades.price * trades.volume

        # store on disc
        # fname = self.folder + self.pair + '_trades.pickle'
        # print('\nData Downloader: storing', fname)
        # trades.to_pickle(fname)

        for i in interval:
            # resample
            gtrades = trades.resample('{}min'.format(i))
            if i == 1:
                fname = os.path.join(self.folder, 'OHLC', self.pair +
                                    '.pickle')
            else:
                fname = os.path.join(self.folder, 'OHLC', self.pair +
                                    '_{}.pickle'.format(i))

            # ohlc, volume
            ohlc = gtrades.price.ohlc()
            ohlc.loc[:, 'volume'] = gtrades.volume.sum()
            ohlc.volume.fillna(0, inplace=True)
            closes = ohlc.close.fillna(method='pad')
            ohlc = ohlc.apply(lambda x: x.fillna(closes))

            # vwap
            ohlc.loc[:, 'vwap'] = gtrades.cost.sum() / ohlc.volume
            ohlc.vwap.fillna(ohlc.close, inplace=True)

            # count
            ohlc.loc[:, 'count'] = gtrades.size()

            # store on disc

            print('\nData Downloader: storing', fname)
            ohlc.to_pickle(fname)


def main(folder, pair, since, timezone, interval):

    dl = GetTradeData(folder, pair, timezone)

    if interval == 0:
        dl.download_trade_data(since)
    else:
        dl.agg_ohlc(since, interval)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--folder',
        help='which (parent) folder to store data in',
        type=str,
        default='')

    parser.add_argument(
        '--pair',
        help=('asset pair to get trade data for. '
              'see KrakenAPI(api).get_tradable_asset_pairs().index.values'),
        type=str,
        default='XXBTZUSD')

    parser.add_argument(
        '--since',
        help=("download/aggregate trade data since given unixtime (exclusive)."
              " If 0 (default) and this script was called before, only an"
              " update to the most recent data is retrieved. If 0 and this"
              " function was not called before, retrieve from earliest time"
              " possible. When aggregating (interval>0), aggregate from"
              " ``since`` onwards (unixtime)."),
        type=str,
        default=0)

    parser.add_argument(
        '--timezone',
        help=("convert the timezone of timestamps to ``timezone``, which must "
              "be a string that pytz.timezone() accepts (see "
              "pytz.all_timezones)"),
        type=str,
        default='America/New_York')

    parser.add_argument(
        '--interval',
        help=("sample downloaded trade data to ohlc format with the given time"
              "interval (minutes). If 0 (default), only download/update trade "
              "data."),
        type=int,
        default=0)

    args = parser.parse_args()
    main(args.folder, args.pair, args.since, args.timezone, args.interval)
