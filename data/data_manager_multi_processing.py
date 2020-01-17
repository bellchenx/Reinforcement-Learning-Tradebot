import datetime
import os
import pickle
import multiprocessing
import threading
import time
import copy

import numpy as np
import pandas as pd
import pytz

from config import (BACKTEST_RANGE, ENTRIES_PER_FILE, INTERVALS, INTERVALS_REV,
                    TRADE_INTERVAL, TRAIN_DATA_RANGE, WINDOW, TRADE_PAIRS, RESAMPLE_THREAD)
from data.get_trade_data import GetTradeData
from utilities.functions import time2str


def ohlcSum(df):
    total_volumn = df['volume'].sum()

    if total_volumn != 0:
        return {
            'open': df['open'][0],
            'high': df['high'].max(),
            'low': df['low'].min(),
            'close': df['close'][-1],
            'volume': total_volumn,
            'count': df['count'].sum(),
            'vwap': (df['vwap'] * df['volume']).sum() / total_volumn
        }
    else:
        return {
            'open': df['open'][0],
            'high': df['high'].max(),
            'low': df['low'].min(),
            'close': df['close'][-1],
            'volume': total_volumn,
            'count': df['count'].sum(),
            'vwap': df['vwap'].mean()
        }


class ResampleThread(multiprocessing.Process):
    def __init__(self, dictionary, ohlc_memory):
        super(ResampleThread, self).__init__()
        # self.manager = multiprocessing.Manager()
        # self.dict = self.manager.dict()
        self.dict = dictionary
        self.queue = multiprocessing.Queue()
        self.status = 'idle'
        self.result = None
        self.ohlc_memory = ohlc_memory

    def addTask(self, timestamp):
        self.status = 'running'
        self.queue.put(timestamp)

    def task(self, timestamp):
        print('Data Manager: Resampling data at ' + time2str(timestamp))
        timezone = pytz.timezone('US/Eastern')
        data_frame = []
        for pair in TRADE_PAIRS:
            key = pair + '_1'
            ohlc = self.ohlc_memory[key]
            single_instr = []
            for interval in INTERVALS_REV:
                end_time = int(timestamp - timestamp % 60)
                start_time = end_time - (interval*WINDOW) * 60
                start_time = pd.Timestamp(start_time*1e9, tzinfo=timezone)
                end_time = pd.Timestamp(end_time*1e9, tzinfo=timezone)
                ohlc = ohlc.loc[start_time: end_time]
                if len(ohlc) == 0:
                    raise AttributeError('Data Manager: Wrong input time')
                if interval != 1:
                    ohlc_resample = ohlc.groupby(pd.Grouper(freq='%dmin' %
                                                            interval, label='right')).agg(ohlcSum)
                else:
                    ohlc_resample = ohlc
                ohlc_resample = ohlc_resample[len(
                    ohlc_resample)-WINDOW-1:len(ohlc_resample)-1]
                ohlc_resample = np.array(ohlc_resample.values.tolist())
                single_instr.append(ohlc_resample)

            data_frame.append((np.array(single_instr)))
        data_frame = np.array(data_frame)
        self.dict[str(timestamp)] = data_frame

    def getResult(self):
        result = dict()
        result.update(self.dict)
        self.status = 'idle'
        return result

    def run(self):
        time.sleep(0.1)
        started = False
        while(True):
            if not self.queue.empty():
                started = True
                self.task(self.queue.get())
            else:
                if started:
                    return
                self.status = 'idle'
                time.sleep(0.01)


class DataManager(threading.Thread):
    def __init__(self,
                 dir: str = 'data',
                 training: bool = True,
                 resample: bool = False,
                 threading: bool = True):
        super(DataManager, self).__init__()

        if not os.path.isdir(dir):
            raise NotADirectoryError('No data directory.')
        self.dir = dir
        self.ready = False

        self.threading = threading
        self.training = training
        if self.training:
            self.folder = 'BASE'
        else:
            self.folder = 'BASE_BACKTEST'
            
        self.resample = resample

        self.ohlc_memory = {}
        self.temp_memory = None
        self.temp_memory_name = None
        self.base_list = None
        self.min_time = None
        self.max_time = None

    @staticmethod
    def _normalize(ohlc):
        for i in range(ohlc.shape[0]):
            for j in range(7):
                dataframe = ohlc[i, :, :, j]
                max_frame = np.max(dataframe, axis=1)
                min_frame = np.min(dataframe, axis=1)
                avg_frame = np.average(dataframe, axis=1)
                
                max_frame = np.transpose(np.tile(max_frame, (dataframe.shape[1], 1)), [1, 0])
                min_frame = np.transpose(np.tile(min_frame, (dataframe.shape[1], 1)), [1, 0])
                avg_frame = np.transpose(np.tile(avg_frame, (dataframe.shape[1], 1)), [1, 0])

                range_frame = (max_frame - min_frame)
                median = np.median(avg_frame[avg_frame > 0], axis=0)
                range_frame[range_frame == 0] = median
                ohlc[i, :, :, j] = (dataframe - min_frame) / range_frame
        return ohlc

    def getDataframe(self, timestamp: int, normalize: bool):
        # timer = time.time()

        if not self.threading:
            raise EnvironmentError(
                'Data Manager: getDataframe needs threading')

        if timestamp < self.min_time or timestamp > self.max_time:
            raise AttributeError(
                'Data Manager: Wrong timestamp %d' % timestamp)

        df = None
        file_index = (
            timestamp - self.min_time)//ENTRIES_PER_FILE//TRADE_INTERVAL
        if file_index == self.temp_memory_name:
            df = copy.deepcopy(self.temp_memory)
        else:
            df = pickle.load(
                open(os.path.join(self.dir, self.folder, self.base_list[file_index]), 'rb'))
            self.temp_memory = copy.deepcopy(df)
            self.temp_memory_name = file_index

        if normalize:
            df = self._normalize(df[str(timestamp)])
        else:
            df = df[str(timestamp)]

        # print('Get dataframe in %.3f sec' % (time.time()-timer))

        return df

    def getTradesInRange(self,
                         time_range: tuple,
                         pair: str = 'XXBTZUSD',
                         price_array: bool = False):
        # Debug
        # timer = time.time()

        start_time, end_time = time_range
        dir_name = os.path.join(self.dir, pair)
        if not os.path.isdir(dir_name):
            raise NameError('Data Manager: No such pair name.')

        # Definition
        name_list = os.listdir(dir_name)
        name_list.sort()
        if name_list[0][0] == '.':
            del name_list[0]

        # Find end file
        idx = len(name_list) - 1
        name = name_list[idx]
        file_end_time = int(name[0:10]) + int(name[10:14]) * 0.0001
        while(end_time < file_end_time):
            idx -= 1
            name = name_list[idx]
            file_end_time = int(name[0:10]) + int(name[10:14]) * 0.0001
            if idx < 0:
                raise AttributeError('Data Manager: Range too large. No data.')
        end_file_idx = idx

        # Find start file
        idx = end_file_idx
        name = name_list[idx]
        file_end_time = int(name[0:10]) + int(name[10:14]) * 0.0001
        while(start_time < file_end_time):
            idx -= 1
            name = name_list[idx]
            file_end_time = int(name[0:10]) + int(name[10:14]) * 0.0001
            if idx < 0:
                raise AttributeError('Data Manager: Range too large. No data.')
        start_file_idx = idx

        # Read data
        name = os.path.join(dir_name, name_list[start_file_idx])
        data = pickle.load(open(name, 'rb'))
        for i in range(start_file_idx+1, end_file_idx+1):
            name = os.path.join(dir_name, name_list[i])
            data = pd.concat([pickle.load(open(name, 'rb')), data])

        data = data.loc[(data['time'] > start_time)
                        & (data['time'] < end_time)]

        # Debug
        # print('Get trades in %.3f sec' % (time.time()-timer))

        if price_array:
            return np.array(data['price'].values.tolist())
        else:
            return data

    def getOHCL(self,
                timestamp: float,
                pair: str = 'XXBTZUSD',
                window: int = WINDOW,
                numpy_array: bool = True):
        # Debug
        # timer = time.time()

        name = pair + '.pickle'
        sample_list = INTERVALS
        sample_list.sort()
        file_path = os.path.join(self.dir, 'OHLC', name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError('OHLC data file missing.')

        if not self.threading:
            ohlc = pd.read_pickle(file_path)
        elif len(self.ohlc_memory) != 0:
            ohlc = self.ohlc_memory[pair + '_1']
        else:
            raise EnvironmentError('No File Obtained by Thread')

        # Time convertion
        tz = pytz.timezone('US/Eastern')
        end_time = int(timestamp - timestamp % 60)
        start_time = end_time - (max(sample_list)*window) * 60
        start_time_str = pd.Timestamp(start_time*1e9, tzinfo=tz)
        end_time_str = pd.Timestamp(end_time*1e9, tzinfo=tz)

        # Aggregate entries
        total = []
        for interval in sample_list:
            ohlc = ohlc.loc[start_time_str: end_time_str]
            if interval != 1:
                ohlc = ohlc.groupby(pd.Grouper(freq='%dmin' %
                                               interval, label='right')).agg(ohlcSum)
            temp = ohlc[len(ohlc)-window-1:len(ohlc)-1]
            if numpy_array:
                temp = np.array(temp.values.tolist())
            total.append(temp)

        # Debug
        # print('Get ohlc in %.3f sec' % (time.time()-timer))

        if numpy_array:
            total = np.array(total)
            return total
        else:
            return total

    def _resample(self, a, b, s, n_of_threads, step=TRADE_INTERVAL, timezone=pytz.timezone('US/Eastern')):

        # if ENTRIES_PER_FILE % n_of_threads != 0:
        #     raise AttributeError('Data Manager: Number of threads are not proportional to file entries.')

        ohlc_base = {}
        a, b, s = int(a), int(b), int(s)
        count = 0
        max_step = int((b-a)/s)

        for timestamp in range(a, b, s*ENTRIES_PER_FILE):
            file_name = str(timestamp) + '.pickle'
            if os.path.isfile(os.path.join(self.dir, self.folder, file_name)):
                continue

            manager = multiprocessing.Manager()
            thread_pool = []
            
            for i in range(n_of_threads):
                thread_pool.append(ResampleThread(
                    manager.dict(), self.ohlc_memory))
                thread_pool[i].start()

            for i in range(0, ENTRIES_PER_FILE, n_of_threads):
                if count >= max_step:
                    break
                for j in range(n_of_threads):
                    if count >= max_step:
                        break
                    thread_pool[j].addTask(timestamp + (i+j)*s)
                    count += 1

            for i in range(n_of_threads):
                thread_pool[i].join()
                ohlc_base.update(thread_pool[i].getResult())

            pickle.dump(ohlc_base, open(os.path.join(
                self.dir, self.folder, str(timestamp)+'.pickle'), 'wb'))
            ohlc_base = {}

    def preprocess(self):
        print('Data Manager: Preparing OHLC data')
        self.ohlc_memory = {}
        ohlc_dir = os.path.join(self.dir, 'OHLC')
        ohlc_list = os.listdir(ohlc_dir)
        for pickle_name in ohlc_list:
            if len(pickle_name) < 10 or pickle_name[-10:-1] != 'USD.pickl' or pickle_name[0] == '.':
                continue
            pickle_path = os.path.join(ohlc_dir, pickle_name)
            data = pd.read_pickle(pickle_path)
            self.ohlc_memory[pickle_name[0:-7] + '_1'] = data

        if self.resample:
            print('Data Manager: Resampling OHLC data')
            if self.training:
                print('Data Manager: Resampling training set')
                self._resample(
                    TRAIN_DATA_RANGE[0], TRAIN_DATA_RANGE[1] + TRADE_INTERVAL, TRADE_INTERVAL, RESAMPLE_THREAD)
            else:
                print('Data Manager: Resampling backtest set')
                self._resample(
                    BACKTEST_RANGE[0], BACKTEST_RANGE[1] + TRADE_INTERVAL, TRADE_INTERVAL, RESAMPLE_THREAD)

        self.base_list = os.listdir(os.path.join(self.dir, self.folder))
        self.base_list.sort()
        if len(self.base_list) != 0:
            if self.base_list[0][0] == '.':
                del self.base_list[0]

            self.min_time = int(self.base_list[0][:-7])
            temp_data = pickle.load(
                open(os.path.join(self.dir, self.folder, self.base_list[-1]), 'rb'))
            self.max_time = int(
                self.base_list[-1][:-7]) + len(temp_data) * TRADE_INTERVAL - TRADE_INTERVAL
            print('Data Manager: From', time2str(
                self.min_time), 'to', time2str(self.max_time))

    def waitToStart(self):
        while(not self.ready):
            time.sleep(0.1)
        print('Data Manager: Ready')

    def terminate(self):
        self._running = False

    def run(self):
        self.preprocess()
        self.ready = True
        while(True):
            time.sleep(0.1)


if __name__ == '__main__':
    dm = DataManager(resample=True)
    dm.start()
