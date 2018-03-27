from os import path, mkdir
import yaml
import pickle
from glob import glob


def check_file(pv_address):
    if not path.exists(pv_address):
        mkdir(pv_address)


def read_yaml(pv_address):
    with open(pv_address, 'rb') as handle:
        ld_data = yaml.load(handle)
    return ld_data


def save_pickle(pd_data, pv_address):
    with open(pv_address, 'wb') as handle:
        pickle.dump(pd_data, handle)


def load_pickle( pv_address):
    with open(pv_address, 'rb') as handle:
        ld_data = pickle.load(handle)
        return ld_data


def get_feature_files(pv_address):
    return glob(path.join(pv_address,'*.pickle'))


def get_features(pv_address, pv_key):
    pd_data = load_pickle(pv_address=pv_address)
    return pd_data['features'][pv_key]



