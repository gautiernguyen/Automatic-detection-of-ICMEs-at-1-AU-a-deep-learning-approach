import pandas as pds
import datetime
import numpy as np
import time
import make_data
import event as evt

'''
Utilities used for transforming a dataset in sliding window in a smart way,
less memory costing than what was done for the sliding windows
'''


def windowed(X, window):
    '''
    Using stride tricks to create a windowed view on the original
    data.
    '''
    shape = int((X.shape[0] - window) + 1), window, X.shape[1]
    strides = (X.strides[0],) + X.strides
    X_windowed = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    return X_windowed


def label(data, width, evtList, method='similarity', thres=0.9):
    '''
    For a given size of window ( expressed in hours),
    return an array with the label according a given method
    The different method considered up to now are as follows :
    - by window : thres of the window is in an event
    - yby event : in the window there is thres of an event
    - by similarity : the window is similar at thres to an event
    - first : the first point of the window is in an event
    - middle : the center point of the window is in an event
    '''
    if method == 'similarity':
        return lblSimilarity(data, width, evtList, thres)
    if method == 'window':
        return lblWindow(data, width, evtList, thres)
    if method == 'event':
        return lblEvent(data, width, evtList, thres)
    if method == 'first':
        return lblFirst(data, evtList, thres)
    if method == 'middle':
        return lblMiddle(data, width, evtList, thres)


def lblEvent(data, width, evtList, thres):
    y = pds.Series(index=data.index, data=0)
    for i, date in enumerate(data.index):
        window = evt.Event(date, date+datetime.timedelta(hours=width))
        if True in [evt.overlap(x, window) > thres*x.duration for x in evtList if (window.begin < x.end) and (window.end > x.begin)]:
            y[i] = 1
    return y


def lblSimilarity(data, width, evtList, thres):
    y = pds.Series(index=data.index, data=0)
    for i, date in enumerate(data.index):
        window = evt.Event(date, date+datetime.timedelta(hours=width))
        if True in [evt.similarity(x, window) > thres for x in evtList if (window.begin < x.end) and (window.end > x.begin)]:
            y[i] = 1
    return y


def lblWindow(data, width, evtList, thres):
    y = pds.Series(index=data.index, data=0)
    for i, date in enumerate(data.index):
        window = evt.Event(date, date+datetime.timedelta(hours=width))
        if True in [evt.overlap(x, window) > thres*window.duration for x in evtList if (window.begin < x.end) and (window.end > x.begin)]:
            y[i] = 1
    return y


def lblFirst(data, evtList, thres):
    y = pds.Series(index=data.index, data=0)
    for i, date in enumerate(data.index):
        if len([x for x in evtList if (x.begin < date) and (x.end > date)]) > 0:
            y[i] = 1
    return y


def lblMiddle(data, width, evtList, thres):
    y = pds.Series(index=data.index, data=0)
    for i, date in enumerate(data.index):
        point = date + datetime.timedelta(hours=width)/2
        if len([x for x in evtList if (x.begin < point) and (x.end > point)]) > 0:
            y[i] = 1
    return y


def random_undersampling_generator(X_train, y_train,
                                   batch_size=128, factor=1.5):
    """
    Data generator with random undersampling.
    Parameters
    ----------
    X_train, y_train
    batch_size
        Number of samples in each batch
    factor
        The number of samples taken from the majority class
        relative to the number of samples in the minority class

    """
    selected = y_train > 0
    minority_idxs = np.nonzero(selected)[0]
    majority_idxs = np.nonzero(~selected)[0]
    n_minority = selected.sum()

    while True:
        majority_undersampled = np.random.choice(majority_idxs, size=int(n_minority * factor))
        indices = np.concatenate((minority_idxs, majority_undersampled))
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            yield X_train[indices[i:i+batch_size]], y_train[indices[i:i+batch_size]]


def get_epoch_steps_random_undersampling(X_train, y_train,
                                         batch_size=128, factor=1.5):
    """
    In the keras generator interface, you need to pass the number of steps
    in each epoch (so keras knows how many times it needs to yield from the
    generator for a single epoch).
    This repeats some of the code from above.
    """
    selected = y_train > 0
    minority_idxs = np.nonzero(selected)[0]
    majority_idxs = np.nonzero(~selected)[0]
    majority_undersampled = np.random.choice(majority_idxs, size=int(selected.sum() * factor))
    indices = np.concatenate((minority_idxs, majority_undersampled))
    return int(len(indices) / batch_size)


def random_undersampling(X_train, y_train, factor=1.5, random_state=0):
    """
    A non-generator version of the random undersampling for the
    validation data passed to keras.
    """
    rng = np.random.RandomState(random_state)
    # majority class is 0, select everything of all other classes
    # selected = y_train.argmax(axis=1) > 0 # for multi-class case
    selected = y_train > 0
    minority_idxs = np.nonzero(selected)[0]
    majority_idxs = np.nonzero(~selected)[0]
    majority_undersampled = rng.choice(majority_idxs, size=int(selected.sum() * factor))
    indices = np.concatenate((minority_idxs, majority_undersampled))
    indices.sort()

    return X_train[indices], y_train[indices]


def wdwMean(inidialData, finalData, width=30, delta=10,
            weights=np.ones(int(30*60/10)),
            column=1):
    '''
    for a given series, compute the punctual mean of the
    windows passing by a point and add it as a column of finalData
    Weights must be a np array of size (width*60/delta)
    '''
    seum = inidialData.copy()
    for i in range(1, int(width*60/delta), 1):
        seum['1+'+str(i)+'0Min'] = seum[column].shift(i)
    seum['mean'] = (seum*weights).sum(axis=1)/np.sum(weights)
    finalData['pred'+str(width)] = seum['mean'].dropna()
    return finalData


def get_similarity(data, width, evtList):
    '''
    For a given df data, a window size width (expressed as an int of hours) and
    an event list, return a serie of the similarity each windows of size width
    of the similarity the windows have with events of evtList
    '''
    y = np.zeros(len(data))
    for i, date in enumerate(data.index):
        window = evt.Event(date-datetime.timedelta(hours=int(width)/2),
                           date+datetime.timedelta(hours=int(width)/2))
        seum = [evt.similarity(x, window)for x in evtList if (window.begin < x.end) and (window.end > x.begin)]
        if len(seum) > 0:
            y[i] = max(seum)
    return pds.Series(index=data.index, data=y)
