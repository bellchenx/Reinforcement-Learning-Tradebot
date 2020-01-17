import time
import datetime
import calendar

import pandas as pd
import pytz


def str2time(string, integer=True):
    t = calendar.timegm(time.strptime(string, '%Y-%m-%d %H:%M')) + 14400
    if integer:
        t = int(t)
    return t


def time2str(unix_time, pd_timestamp=False):
    tz = pytz.timezone('US/Eastern')
    if pd_timestamp:
        return pd.Timestamp(unix_time*1e9, tzinfo=tz)
    else:
        return pd.Timestamp(unix_time*1e9, tzinfo=tz).strftime('%Y-%m-%d %H:%M')

if __name__ == '__main__':
    timer = time.time()
    print(time2str(str2time('2017-9-1 8:00')), time.time()-timer)