from os import path, mkdir
import yaml


def check_file(pv_address):
    if not path.exists(pv_address):
        mkdir(pv_address)


def read_yaml(pv_address):
    with open(pv_address, 'rb') as handle:
        ld_data = yaml.load(handle)
    return  ld_data