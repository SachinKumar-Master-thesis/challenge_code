from emotions_audio.src.training.model_base_class import BaseModel
import keras as ker
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, Dense
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
from utils.common_utils import *

class ModelCNN(BaseModel):

    def __init__(self, pd_params):
        BaseModel.__init__(self, pd_params=pd_params)
        self.d_model_params = pd_params['classifier_parameters'][self.v_method]
        self.C_model = Sequential()
        self.v_max_time= 12.0
        self.__get_ip_shape(pd_params=pd_params['features'])


    def __get_ip_shape(self, pd_params,):
        self.v_rows = int(self.v_max_time/pd_params['hop_length_seconds'])
        self.v_columns = int(pd_params['mfcc']['n_fft']/2)+1

    def prepare_data2train(self):
        # Read meta data
        self.f_meta = self.get_meta_data(pv_key='Train')

        # get all the training data present in the extracted folder
        ll_feature_files = get_feature_files(self.d_paths['features'])
        lma_features = np.zeros((len(ll_feature_files),self.v_rows, self.v_columns))
        lma_labels = np.zeros((len(ll_feature_files), 2))

        for i, f in enumerate(ll_feature_files):
            if(i==200):
                break
            lma_labels, lv_true = self.update_label(pv_address=f, pv_count=i, pma_labels=lma_labels)
            lma_data = get_features(f, 'stft')
            if lv_true:
                if lma_data.shape[0] > self.v_rows:
                    lma_features[i,:,:] = lma_data[:self.v_rows,:]
                else:
                    lma_features[i,:lma_data.shape[0],:] = lma_data[:,:]
            else:
                lma_features = np.delete(lma_features, i, axis=0)

        return lma_features, lma_labels

    def instantiate_model(self):
        self.C_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(None, self.v_columns,1)))
        self.C_model.add(Conv2D(64, (3, 3), activation='relu'))
        self.C_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.C_model.add(Dropout(0.25))
        self.C_model.add(GlobalMaxPooling2D())
        self.C_model.add(Dense(2))
        self.C_model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])


    def fit(self):
        lma_data, lma_labels = self.prepare_data2train()
        self.instantiate_model()
        lma_data = np.expand_dims(lma_data, axis=3)
        self.C_model.fit(x=lma_data, y=lma_labels, verbose=1, validation_data=(lma_data, lma_labels))
        print 1
