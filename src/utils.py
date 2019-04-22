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


# def anaylses_results(fichier):
#     d = pickle.load(open(fichier, 'rb'))
#     for cle in d.keys():
#         print('{} Test Risk {}'.format(d[cle][0], d[cle][2][0]))

def anaylses_results(fichier):
    d = pickle.load(open(fichier, 'rb'))
    valeurs = list(d.values())
    test_risk = [el[2][0] for el in valeurs]
    hps_combi = [el[0] for el in valeurs]
    best_risk_value = test_risk[np.argmax(test_risk)]
    best_hps_selected = hps_combi[np.argmax(test_risk)]
    print('best hps is {}  and the best test risk is {}'.format(best_hps_selected, best_risk_value))


def main_analyses(directory):
    os.chdir(directory)
    for fichier in glob('*.pck'):
        print('{}'.format(fichier))
        anaylses_results(fichier=fichier)


if __name__== '__main__':
    main_analyses(directory='./')


# def weights_set_to_zeros(model):
#     for name, param in model.named_parameters():
