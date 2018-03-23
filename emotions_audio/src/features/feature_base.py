from librosa import load
from os import path

class FeaturesBase(object):
    def __init__(self, pd_config):
        self.d_config = pd_config
        self.v_data_path = self.d_config['path']['data']
        self.v_feature_path = self.d_config['path']['features']
        self.sr = float(pd_config['features']['sr'])
        self.__update_lengths(self.d_config['feature'])

    def read_data(self, pv_address):
        lma_data, _ = load(pv_address, sr=self.sr, mono=True)
        return lma_data

    def __update_lengths(self, pd_features):
        self.win_length = self.sr * pd_features['win_length_seconds']
        self.hop_length = self.sr * pd_features['hop_length_seconds']

    def get_load_address(self, pv_filename):
        return path.join()

    @abstractmethod
    def run(self):
        pass

class SpectralBase(FeaturesBase):
    def __int__(self, pd_config):
        FeaturesBase.__init__(self, pd_config=pd_config)
        self.__update_basic(pd_features=pd_config['features'])




