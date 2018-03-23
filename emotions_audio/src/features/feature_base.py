import librosa as lib
from librosa.display import specshow
from os import path
import numpy as np
from scipy.signal import hanning

class FeaturesBase(object):
    def __init__(self, pd_config):
        self.d_config = pd_config['features']
        self.sr = float(self.d_config['sr'])
        self.__update_lengths()
        self.l_feature_names = None
        self.ma_features = None

    def read_data(self, pv_address):
        lma_data, _ = lib.load(pv_address, sr=self.sr, mono=True)
        return lma_data

    def __update_lengths(self):
        self.v_win_length = int(self.sr * self.d_config['win_length_seconds'])
        self.v_hop_length = int(self.sr * self.d_config['hop_length_seconds'])

    def update_feature_matrix(self, pma_data, feature_name):
        if self.ma_features == None:
            self.ma_features = pma_data
        else:
            self.ma_features = np.vstack((self.ma_features, pma_data))


class SpectralBase(FeaturesBase):
    def __int__(self, pd_config):
        FeaturesBase.__init__(self,pd_config=pd_config)
        self.n_fft = self.d_config['mfcc']['n_fft']
        self.window = hanning(self.v_win_length)
        self.stft = None
        self.power_spectogram = None

    def get_stft(self, pma_data):
        self.stft = lib.stft(y=pma_data, win_length=self.v_win_length, hop_length=self.v_hop_length, window=self.window,
                        n_fft=self.n_fft)
        self.power_spectogram = np.abs(self.stft**2)

    def plot_spectrum(self, pma_data, pv_yaxis='linear' ):
        specshow(data=pma_data, sr=self.sr, hop_length=self.v_hop_length, x_axis='time', y_axis='linear')






