from data.data_manager_multi_processing import DataManager

if __name__ == '__main__':
    dm = DataManager(resample=True, training=False)
    dm.start()