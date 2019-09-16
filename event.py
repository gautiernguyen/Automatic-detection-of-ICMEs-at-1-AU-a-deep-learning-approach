import pandas as pds
import datetime
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Event:

    def __init__(self, begin, end, param=None):
        self.begin = begin
        self.end = end
        self.proba = None
        self.duration = self.end-self.begin

    def __eq__(self, other):
        '''
        return True if other overlaps self during 65/100 of the time
        '''
        return overlap(self, other) > 0.65*self.duration

    def __str__(self):
        return "{} ---> {}".format(self.begin, self.end)

    def get_Proba(self, y):
        '''
        Give the mean probability of the event following the list
        of event predicted probability y
        '''
        self.proba = y[self.begin:self.end].mean()

    def get_data(self, df):
        self.param = df

    def plot(self, data,  predictedProba=None,
             predictedY=None, expectedY=None, delta=20,
             singleFeatures=['B', 'Beta', 'V', 'Np', 'Vth'],
             groupFeatures=['Bx', 'By', 'Bz'],
             thres=0.75,
             expectedSimi=None,
             predictedSimi=None,
             figsize=(25, 35),
             cmap='viridis'):
        return _plot(data, self.begin, self.end,
                     predictedProba=predictedProba,
                     predictedY=predictedY,
                     expectedY=expectedY, delta=delta,
                     singleFeatures=singleFeatures,
                     groupFeatures=groupFeatures,
                     expectedSimi=expectedSimi,
                     predictedSimi=predictedSimi,
                     figsize=figsize,
                     thres=thres,
                     cmap=cmap)

    def getValue(self, df, feature):
        '''
        for a given df, return the mean of a given feature during the events
        '''
        return df[feature][self.begin:self.end].mean()


def overlap(event1, event2):
    '''return the time overlap between two events as a timedelta'''
    delta1 = min(event1.end, event2.end)
    delta2 = max(event1.begin, event2.begin)
    return max(delta1-delta2,
               datetime.timedelta(0))


def similarity(event1, event2):
    if event1 is None:
        return 0
    inter = overlap(event1, event2)
    return inter/(event1.duration+event2.duration-inter)


def jaccard(predicted_list, validation_list):
    '''
    compute jaccard indice of two event lists
    '''
    inter = [[overlap(predicted, expected).total_seconds() for predicted in predicted_list] for expected in validation_list]
    return np.sum(inter)/(np.sum([x.duration.total_seconds() for x in validation_list])+np.sum([x.duration.total_seconds() for x in predicted_list])-np.sum(inter))


def overlapWithList(ref_event, event_list, percent=False):
    '''
    return the list of the overlaps between an event and the elements of
    an event list
    Have the possibility to have it as the percentage of fthe considered event
    in the list
    '''
    if percent:
        return [overlap(ref_event, elt)/elt.duration for elt in event_list]
    else:
        return [overlap(ref_event, elt) for elt in event_list]


def isInList(ref_event, event_list, thres):
    '''
    returns True if ref_event is overlapped thres percent of its duration by
    at least one elt in event_list
    '''
    return max(overlapWithList(ref_event,
                               event_list)) > thres*ref_event.duration


def choseEventFromList(ref_event, event_list, choice='first'):
    '''
    return an event from even_list according to the choice adopted
    first return the first of the lists
    last return the last of the lists
    best return the one with max overlap
    merge return the combination of all of them
    '''
    if choice == 'first':
        return event_list[0]
    if choice == 'last':
        return event_list[-1]
    if choice == 'best':
        return event_list[np.argmax(overlapWithList(ref_event, event_list))]
    if choice == 'merge':
        return evt.merge(event_list[0], event_list[-1])


def find(ref_event, event_list, thres, choice='first'):
    '''
    Return the event in event_list that overlap ref_event for a given threshold
    if it exists
    Choice give the preference of returned :
    first return the first of the lists
    Best return the one with max overlap
    merge return the combination of all of them
    '''
    if isInList(ref_event, event_list, thres):
        return(choseEventFromList(ref_event, event_list, choice))
    else:
        return None


def merge(event1, event2):
    return Event(event1.begin, event2.end)


def merge_proba(event1, event2):
    evt = merge(event1, event2)
    evt.proba = (event1.duration.total_seconds()*event1.proba+event2.duration.total_seconds()*event2.proba)/(event1.duration.total_seconds()+event2.duration.total_seconds())
    return evt


def listToCSV(eventList, filename):
    '''
    For a given event list, save it into a csv file ( filename as str)
    The csv file will then give for each event of the listbegin, end and proba
    '''
    edf = pds.DataFrame(data={'begin': [x.begin for x in eventList],
                              'end': [x.end for x in eventList],
                              'proba': [x.proba for x in eventList]})
    edf.to_csv(filename)
    return edf


def read_csv(filename, get_proba=False,
             index_col=0, header=None, dateFormat="%Y/%m/%d %H:%M",
             sep=','):
    '''
    Consider a  list of events as csv file ( with at least begin and end)
    and return a list of events
    index_col and header allow the correct reading of the current fp lists
    '''
    df = pds.read_csv(filename, index_col=index_col, header=header, sep=sep)
    df['begin'] = pds.to_datetime(df['begin'], format=dateFormat)
    df['end'] = pds.to_datetime(df['end'], format=dateFormat)
    evtList = [Event(df['begin'][i], df['end'][i])
               for i in range(0, len(df))]
    if get_proba is True:
        for i, elt in enumerate(evtList):
            elt.proba = df['proba'][i]
    return evtList


def get_similarity(index, width, evtList):
    '''
    For a given list of event and a given window size (in hours) and
    a datetime index, return the associated serie of similarities
    '''
    y = np.zeros(len(index))
    for i, date in enumerate(index):
        window = Event(date-datetime.timedelta(hours=int(width)/2),
                       date+datetime.timedelta(hours=int(width)/2))
        seum = [similarity(x, window)for x in evtList if (window.begin < x.end) and (window.end > x.begin)]
        if len(seum) > 0:
            y[i] = max(seum)
    return pds.Series(index=data.index, data=y)


def _plot(data, startTime, endTime, predictedProba=None,
          predictedY=None, expectedY=None, delta=20,
          singleFeatures=['B', 'Beta', 'V', 'Np', 'Vth'],
          groupFeatures=['Bx', 'By', 'Bz'],
          expectedSimi=None,
          predictedSimi=None,
          thres=0.75,
          figsize=(25, 35),
          cmap='viridis'):
    '''
    Consider a dataset, plot the main features during the considered interval
      delta hours around the event begin and end
    Supposes dataset has the mentioned features
    features
    Possibility to plot the preditcions and the proba by setting predictedY,
     predictedProba or expectedY to the vector got from classification
    '''
    data0 = data[startTime-datetime.timedelta(hours=delta):
                 endTime+datetime.timedelta(hours=delta)]
    label = []
    if expectedY is not None:
        label.append(expectedY[startTime-datetime.timedelta(hours=delta):endTime+datetime.timedelta(hours=delta)])
    if predictedY is not None:
        label.append(predictedY[startTime-datetime.timedelta(hours=delta):endTime+datetime.timedelta(hours=delta)])
    if predictedProba is not None:
        label.append(predictedProba[startTime-datetime.timedelta(hours=delta):endTime+datetime.timedelta(hours=delta)])
    rays = []
    if expectedSimi is not None:
        rays.append(expectedSimi)
    if predictedSimi is not None:
        rays.append(predictedSimi)
    numplots = len(singleFeatures) + min(len(groupFeatures), 1) + min(len(label), 1)+len(rays)

    fig, axarr = plt.subplots(nrows=numplots, ncols=1, figsize=figsize,
                              sharex=True)

    for i, param in enumerate([x for x in singleFeatures]):
        axarr[i].plot(data0.index, data0[param])
        if param == 'Beta':
            axarr[i].set_ylim(-0.05, 1.7)
        if param == 'Np':
            axarr[i].set_ylim(0, 50)
    if len(groupFeatures) > 0:
        axarr[len(singleFeatures)].plot(data0.index, data0[groupFeatures])
    axarr[len(singleFeatures)].legend(groupFeatures)
    for i in range(0, len(label)):
        axarr[-len(rays)-1].plot(label[i].index, label[i])
    for i in range(0, len(rays)):
        plotRays(rays[i], axarr[-i-1], startTime, endTime,
                 datetime.timedelta(hours=delta), thres=0, cmap=cmap)
    for ax in axarr:
        ax.axvline(startTime, color='k')
        ax.axvline(endTime, color='k')
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=int(delta/10)))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
        ax.xaxis.grid(True, which="minor")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('\n\n %d/%m/%y' +
                                                          ' \n %H'':'+'\n %M'))
        ax.legend(loc='best')
    axarr[-len(rays)-1].axhline(thres, ls='dashed')
    return fig, axarr


def _plotRays(data, ax, startTime, endTime, delta=datetime.timedelta(hours=0),
              thres=0.5, cmap='viridis', vmin=None, vmax=None, norm=None):
    '''
    For a df or a series of probability, plot the rays of plotPrediction
    '''
    t_data = data[startTime-delta:endTime+delta].index
    data_tmp = data[startTime-delta:endTime+delta].values.T
    data_tmp[data_tmp < thres] = 0

    xx, yy = np.meshgrid(t_data, np.arange(0, len(data.columns)))
    im1 = ax.pcolormesh(xx, yy, data_tmp, vmin=vmin, vmax=vmax, cmap=cmap)
