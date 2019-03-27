# -*- coding: utf-8 -*-
__author__ = 'maoss2'
import os
import numpy as np
from glob import glob
import pickle


def load_file(directory):
    os.chdir(directory)
    for fichier in glob('*_NN*'):
        print('{}'.format(fichier))
        d = pickle.load(open(fichier, 'rb'))
        accuracy_values = []
        validation_loss_mean = []
        for cle, valeur in d.items():
            if len(valeur) == 2:
                accuracy_values.append(d[cle][1][0])
                validation_loss_mean.append(d[cle][1][1])
            else:
                accuracy_values.append(d[cle][2][0])
                validation_loss_mean.append(d[cle][2][1])
        print('accuracy loss', accuracy_values)
        print('max accuracy', np.max(accuracy_values))
        # print('validation loss', validation_loss_mean)
