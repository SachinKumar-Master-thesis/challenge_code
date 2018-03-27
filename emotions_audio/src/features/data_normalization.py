import numpy as np
from os import path
from utils.common_utils import *
from glob import glob

class DataNormalization:
    def __init__(self, pd_params):
        self.d_paths = pd_params['path']
        self.v_mean = None
        self.v_var = None
        self.count = 0

    def __update_values(self, pma_data):
        lv_rows = pma_data.shape[0]
        lv_start_index = 0
        if self.v_mean is None:
            self.v_mean = pma_data[0,:]
            self.v_var = np.zeros(pma_data.shape[1])
            self.count+=1
            lv_start_index = 1

        for i in xrange(lv_start_index, lv_rows):
            self.count += 1
            lv_old_mean = self.v_mean
            self.v_mean = self.v_mean - ((pma_data[i,:] - self.v_mean)/self.count)
            self.v_var = self.v_var + ((pma_data[i,:] - self.v_mean)*(pma_data[i,:] - lv_old_mean))

    def __get_feature_files(self):
        return glob(path.join(self.d_paths['features'], '*.pickle'))

    def __get_data(self, pv_path, pv_key):
        ld_features = load_pickle(pv_path)
        return ld_features['features'][pv_key]

    def __get_save_model_address(self):
        self.v_normalizer_save_address = path.join(self.d_paths['normalizer'], 'normalizer_model.pickle')

    def fit(self, pv_key):
        ll_files = self.__get_feature_files()
        for f in ll_files:
            lma_features = self.__get_data(pv_path=f, pv_key=pv_key)
            self.__update_values(pma_data=lma_features)

    def run(self, pma_data):
        lv_std = np.sqrt(self.v_var/(self.count-1))
        return (pma_data - self.v_mean)/lv_std
