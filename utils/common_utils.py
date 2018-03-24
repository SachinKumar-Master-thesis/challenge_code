from os import path, mkdir
import yaml
import pickle


def check_file(pv_address):
    if not path.exists(pv_address):
        mkdir(pv_address)


def read_yaml(pv_address):
    with open(pv_address, 'rb') as handle:
        ld_data = yaml.load(handle)
    return  ld_data

def save_pickle(pd_data, pv_address):
    with open(pv_address, 'wb') as handle:
        pickle.dump(pd_data, handle)

def load_pickle( pv_address):
    with open(pv_address, 'rb') as handle:
        ld_data = pickle.load(handle)
        return  ld_data