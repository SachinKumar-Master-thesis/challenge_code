from utils.audio_config import AudioConfig
from emotions_audio.src.features.extract_features import ExtractFeatures
from emotions_audio.src.features.data_normalization import DataNormalization

lv_yaml_addr = '../resources/audio_config.yaml'

lcl_audio_config = AudioConfig(pv_parameter_addr=lv_yaml_addr)
lcl_features = ExtractFeatures(lcl_audio_config.d_parameters)
#lcl_features.run()

# Normalization
lcl_normalizer = DataNormalization(lcl_audio_config.d_parameters)
lcl_normalizer.run('mel')
print 1


