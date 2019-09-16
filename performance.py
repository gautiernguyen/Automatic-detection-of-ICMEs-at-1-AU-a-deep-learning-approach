import pandas as pds
import datetime
import numpy as np
import time
import event as evt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import recall_score, precision_score, auc
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import postProcess


def evaluate(predicted_list, test_list, thres=0.51, durationCreepies=2.5):
    '''
    for each cloud of validation_list, gives the list of clouds in the
    predicted_list that overlap the cloud among the threshold
    '''
    TP = []
    FN = []
    FP = []
    detected = []
    for event in test_list:
        corresponding = evt.find(event, predicted_list, thres, 'best')
        if corresponding is None:
            FN.append(event)
        else:
            TP.append(corresponding)
            detected.append(event)
    FP = [x for x in predicted_list if max(evt.overlapWithList(x, test_list, percent=True)) == 0]
    seum = [x for x in FP if x.duration < datetime.timedelta(hours=durationCreepies)]
    for event in seum:
        FP.remove(event)
        predicted_list.remove(event)

    return TP, FN, FP, detected


def errorBegin(predicted_list, validation_list):
    '''
    for two validation lists, compute the error on beginning
    between the events of the two lists
    '''
    return [abs(predicted_list[i].begin-validation_list[i].begin).total_seconds()/validation_list[i].duration.total_seconds() for i in range(0, len(predicted_list))]


def errorEnd(predicted_list, validation_list):
    '''
    for two validation lists, compute the error on ending
    between the events of the two lists
    '''
    return [abs(predicted_list[i].end-validation_list[i].end).total_seconds()/validation_list[i].duration.total_seconds() for i in range(0, len(predicted_list))]


def reduced_precision(precisions, recalls):
    model = RandomForestRegressor()
    model.fit(precisions.reshape(-1, 1), recalls.reshape(-1, 1))

    y = model.predict(np.arange(min(precisions), max(precisions), 0.01).reshape(-1,1))
    score = auc(np.arange(min(precisions), max(precisions), 0.01), y)/(1-x[0])

    return score, y, np.arange(min(precisions),max(precisions), 0.01)
