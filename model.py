# -*- coding: utf-8 -*-



import keras
from keras.layers import Dense
from keras.models import Sequential

#from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from keras import backend as K
K.set_image_dim_ordering('th')


class Models:
    
    def __init__(self, nb_frames, grid_size_0, grid_size_1, nb_actions):
        self.nb_frames = nb_frames
        self.grid_size_0 = grid_size_0
        self.grid_size_1 = grid_size_1
        self.nb_actions = nb_actions
    
    def get_model(self):
        model = keras.models.Sequential()

        #model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(self.nb_frames, self.grid_size_0, self.grid_size_1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.nb_actions))
        model.compile(RMSprop(), 'MSE')
        return model
