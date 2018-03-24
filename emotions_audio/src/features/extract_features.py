from emotions_audio.src.features.spectral_features import Mel
from emotions_audio.src.features.feature_base import FeaturesBase
from os import path
from glob import glob
from utils.common_utils import *

class ExtractFeatures(object):
    def __init__(self, pd_config):
        self.d_config = pd_config
        self.v_data_path = self.d_config['path']['data']
        self.v_feature_path = self.d_config['path']['features']
        self.files=None

    def get_load_address(self, pv_filename):
        return path.join(self.v_data_path, pv_filename)

    def get_save_address(self, pv_filename):
        lv_name = path.splitext(path.basename(pv_filename))[0]
        lv_base_folder = path.basename(path.dirname(pv_filename))
        return path.join(self.v_feature_path, lv_base_folder + '_' +lv_name+'.pickle')

    def get_files(self):
        self.files = glob(path.join(self.v_data_path,'*','*.wav'))

    def run(self):
        self.get_files()
        for f in self.files:
            ld_features = {}
            lcl_extractor = FeaturesBase(pd_config=self.d_config)
            lma_data = lcl_extractor.read_data(pv_address=f)

            # get power and mel-spectrogram
            lcl_mel  = Mel(pd_config=self.d_config)
            lcl_mel.get_mfcc(pma_data=lma_data)

            lv_save_address = self.get_save_address(f)
            ld_features = {'features':{'mel': lcl_mel.melspectogram.T, 'stft':lcl_mel.power_spectogram.T,
                            'mfcc': lcl_mel.mfcc.T},'config': self.d_config}
            save_pickle(pd_data=ld_features, pv_address=lv_save_address)


