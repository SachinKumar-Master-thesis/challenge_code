from os import path
from utils.common_utils import *
import yaml

class AudioConfig:
    def __init__(self, pv_parameter_addr):
        self.d_parameters = read_yaml(pv_parameter_addr)
        self.d_parameters['path'] = self.__process_paths(self.d_parameters['path'])


    def __process_paths(self, pd_paths):

        # base address
        check_file(pd_paths['base'])

        #create and update feature folder address
        pd_paths['features'] = path.join(pd_paths['base'], pd_paths['features'])
        check_file((pd_paths['features']))

        # create and update feature folder address
        pd_paths['normalizer'] = path.join(pd_paths['base'], pd_paths['normalizer'])
        check_file((pd_paths['normalizer']))

        return pd_paths
