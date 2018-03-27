from os import path,mkdir
from glob import glob
import numpy as np
from utils.common_utils import *
import pandas as pd
import abc

class BaseModel(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, pd_params):
        self.d_paths = pd_params['path']
        self.d_classifier = pd_params['classifier']
        self.v_method = self.d_classifier['method']
        self.l_use_meta_col = ['video', 'utterance', 'arousal', 'valence']
        self.f_meta =None

    def get_meta_data(self, pv_key='Train'):
        lv_address = path.join(self.d_paths['meta'], 'omg_{0}Videos.csv'.format(pv_key))
        lf_meta = pd.read_csv(lv_address, usecols=self.l_use_meta_col)
        return lf_meta

    def update_label(self, pv_address, pv_count, pma_labels):
        lv_file_name = path.splitext(path.basename(pv_address))[0]
        lv_video = '_'.join(lv_file_name.split('_')[0:2])
        lv_utternce = '{0}.mp4'.format('_'.join(lv_file_name.split('_')[2:]))
        lf_label = self.f_meta.loc[(self.f_meta['video']==lv_video) & ( self.f_meta['utterance']==lv_utternce), :]
        if not lf_label.empty:
            pma_labels[pv_count,0] = lf_label['arousal'].values[0]
            pma_labels[pv_count,1] = lf_label['valence'].values[0]
            return pma_labels, True

        pma_labels = np.delete(pma_labels, pv_count, axis=0)
        return  pma_labels, False

    @abc.abstractmethod
    def prepare_data2train(self):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def instantiate_model(self):
        pass



