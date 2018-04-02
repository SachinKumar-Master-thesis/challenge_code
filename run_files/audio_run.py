from utils.audio_config import AudioConfig
from emotions_audio.src.features.extract_features import ExtractFeatures
from emotions_audio.src.features.data_normalization import DataNormalization
from emotions_audio.src.training.cnn_model import *
from emotions_audio.src.training.svm_model import *

lv_yaml_addr = '../resources/audio_config.yaml'

lcl_audio_config = AudioConfig(pv_parameter_addr=lv_yaml_addr)
lcl_features = ExtractFeatures(lcl_audio_config.d_parameters)
#lcl_features.run()

# Normalization
lcl_normalizer = DataNormalization(lcl_audio_config.d_parameters)
#lcl_normalizer.fit('stft')

# Train
lcl_train = ModelSVM(lcl_audio_config.d_parameters)
lcl_train.fit()
print 1
