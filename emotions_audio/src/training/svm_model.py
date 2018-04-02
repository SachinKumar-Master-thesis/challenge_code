from emotions_audio.src.training.model_base_class import BaseModel
import numpy as np
from sklearn.svm import LinearSVR
from utils.common_utils import *

class ModelSVM(BaseModel):

    def __init__(self, pd_params):
        BaseModel.__init__(self, pd_params=pd_params)
        self.d_model_params = pd_params['classifier_parameters'][self.v_method]
        self.C_model = None

    def prepare_data2train(self):
        # Read meta data
        self.f_meta = self.get_meta_data(pv_key='Train')

        # get all the training data present in the extracted folder
        ll_feature_files = get_feature_files(self.d_paths['features'])
        lma_features = None
        lma_labels = None

        lv_count = 0
        for i, f in enumerate(ll_feature_files):
            if(i==200):
                break

            lma_data = get_features(f, 'sound_net')
            lma_data = np.squeeze(lma_data)

            lma_labels_temp, lv_true = self.update_label(pv_address=f, pv_count=lv_count, pma_labels=lma_labels)

            if len(lma_data) == 0 or not lv_true:
                continue

            if lma_features is None:
                lma_features = lma_data
            else:
                lma_features = np.vstack((lma_features, lma_data))

            if lma_labels is None:
                lma_labels = lma_labels_temp
            else:
                lma_labels = np.vstack((lma_labels, lma_labels_temp))

            lv_count+=1

        return lma_features, lma_labels

    def instantiate_model(self):
        self.C_model = LinearSVR(verbose=1)

    def fit(self):
        lma_data, lma_labels = self.prepare_data2train()
        self.instantiate_model()
        self.C_model.fit(lma_data,lma_labels[:,0])
        print 1
