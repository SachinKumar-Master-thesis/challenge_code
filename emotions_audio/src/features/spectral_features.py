from emotions_audio.src.features.feature_base import SpectralBase, FeaturesBase
import librosa as lib
import numpy as np

class Mel(SpectralBase):
    def __init__(self, pd_config):
        SpectralBase.__init__(self, pd_config)
        self.d_mel = self.d_config['mfcc']
        self.melspectogram = None
        self.mfcc = None
        self.d_mfcc_config = self.d_config['mfcc']

    def __CMN(self):
        """
        Cepstral mean normalization
        :return: None
        """
        self.melspectogram = self.melspectogram - np.tile(np.mean(self.melspectogram, axis=1), (self.melspectogram.shape[1], 1)).T

    def get_melspectogram(self, pma_data):
        self.get_stft(pma_data)

        lma_filter = lib.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.d_mel['n_mels'],
                                     fmax= self.d_mel['fmax'], fmin=self.d_mel['fmin'])
        self.melspectogram = lib.amplitude_to_db(np.dot(lma_filter, self.power_spectogram))
        self.__CMN()

    def get_mfcc(self, pma_data):
        self.get_melspectogram(pma_data)
        self.mfcc = lib.feature.mfcc(S=self.melspectogram, n_mfcc=self.d_mel['n_mfcc'])
