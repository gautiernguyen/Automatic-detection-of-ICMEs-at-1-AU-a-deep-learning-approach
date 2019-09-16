import pandas as pds
import datetime
import numpy as np
import time
import event as evt
import seaborn as sns
import matplotlib.pyplot as plt


def get_weights(serie, bins):
    a, b = np.histogram(serie, bins=bins)
    weights = 1/(a[1:]/np.sum(a[1:]))
    weights = np.insert(weights, 0,1)
    weights_Serie = pds.Series(index = serie.index, data=1)
    for i in range(1, bins):
        weights_Serie[(serie>b[i]) & (serie<b[i+1])] = weights[i]
    return weights_Serie


def scatterFeature(dataset, listsOfEvents, featuresList, legend,
                   method='mean'):
    '''
    Scatter the different confusion parameters in a plane made of two defined
    features among those existing in the dataset

    ARGS :
    dataset : Pandas dataframe
    listsOfEvents : list of Events
    List of features : list of 2 features among those in the dataset
    method : how to compute the feature for a given event
    '''
    funcs = {'mean': np.mean, 'max': np.max, 'min': np.min, 'std': np.std}
    fig = plt.figure()
    for elt in ListsOfEvents:
        absc = np.zeros(len(elt))
        ordo = np.zeros(len(elt))
        for i, event in enumerate(elt):
            absc[i] = np.mean(dataset[event.begin:event.end][featuresList[0]])
            ordo[i] = np.mean(dataset[event.begin:event.end][featuresList[1]])
        fig.scatter(absc, ordo)
    fig.xlabel(featuresList[0])
    fig.ylabel(featuresList[1])
    plt.legend(legend)
    return fig
