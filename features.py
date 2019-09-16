import spacepy.pycdf as pycdf
import scipy.constants as constants
import pandas as pds
import datetime
import numpy as np
import time


'''
These functions compute extrafeatures from data loaded from space missions
Current computed features are :
Beta
AlfvenMach
AlphaProtonRatio
Theta
Phi
'''


def computeBeta(data):
    '''
    compute the evolution of the Beta for data
    data is a Pandas dataframe
    The function assume data already has ['Np','B','Vth'] features
    '''
    try:
        data['Beta'] = 1e6 * data['Vth']*data['Vth']*constants.m_p*data['Np']*1e6*constants.mu_0/(1e-18*data['B']*data['B'])
    except KeyError:
        print('Error computing Beta,B,Vth or Np'
              ' might not be loaded in dataframe')
    return data


def computePm(data):
    '''
    compute the evolution of the Magnetic pressure for data
    data is a Pandas dataframe
    The function assume data already has 'B' features
    '''
    try:
        data['Pm'] = 1e-18*data['B']*data['B']/(2*constants.mu_0)
    except KeyError:
        print('Error computing Beta,B,Vth or Np'
              ' might not be loaded in dataframe')
    return data


def computeRmsBob(data):
    '''
    compute the evolution of the rmsbob instantaneous for data
    data is a Pandas dataframe
    The function assume data already has ['B_rms] features
    '''
    try:
        data['RmsBob'] =np.sqrt(data['Bx_rms']**2+data['By_rms']**2+data['Bz_rms']**2)/data['B']
    except KeyError:
        print('Error computing rmsbob,B or rms of components'
              ' might not be loaded in dataframe')
    return data


def computePdyn(data):
    '''
    compute the evolution of the Beta for data
    data is a Pandas dataframe
    the function assume data already has ['Np','V'] features
    '''
    try:
        data['Pdyn'] = 1e12*constants.m_p*data['Np']*data['V']**2
    except KeyError:
        print('Error computing Pdyn, V or Np might not be loaded '
              'in dataframe')


def computeAlfvenMach(data):
        '''
        compute the evolution of the Alfven Mach for data
        data is a Pandas dataframe
        The function assume data already has ['Np','B','V'] features
        '''
        try:
            data['Malfven'] = 1e3*data['V']*np.sqrt(1e6*data['Np']*constants.m_p*constants.mu_0)/((1e-9)*data['B'])
        except KeyError:
            print('Error computing the Alfven Mach, Np,V or B'
                  '  might not be loaded in dataframe')
        return data


def computeAlphaProtonRatio(data):
        '''
        compute the evolution of the AlphaProton Ratio for data
        data is a Pandas dataframe
        The function assume data already has ['Np_nl','Na_nl'] features
        '''
        try:
            data['C'] = data['Na_nl']/data['Np_nl']
        except KeyError:
            print('Error computing C, Na_nl or Np_nl'
                  '  might not be loaded in dataframe')
        return data


def computeRotation(data):
        '''
        compute the evolution of the AlphaProton Ratio for data
        data is a Pandas dataframe
        The function assume data already has ['Bx','By','Bz'] features
        '''
        try:
            data['Theta'] = np.arctan2(np.sqrt(data['Bx']**2 + data['By']**2), data['Bz']) * 180 / np.pi %360
            data['Phi'] = (np.arctan2(data['By'], data['Bx']) + np.pi) * 180 / np.pi %360
        except KeyError:
            print('Error computing rotation parameters,'
                  ' components of B might not be loaded')
        return data


def computeMagnetosonicMach(data):
    '''
    Compute the evolution of the MagnetosonicMach for a given DataFrame
    data is a Pandas DataFrameThe function assume data already has ['B,V,Np
    Vth] features
    '''
    try:
        va = (1e-9)*data['B']/np.sqrt(constants.mu_0*data['Np']*1e6)
        cs = np.sqrt((1.6e-19)*5/3*data['Vth']/constants.m_p)
        data['Mms'] = data['V']/np.sqrt(va**2 + cs**2)
    except KeyError:
        print('Error computing Magnetosonic Mach,'
              'B,Np,Vth or V might not be loaded')
    return data


def computeAngles(data):
    '''
    Compute the evolution of the MagnetosonicMach for a given DataFrame
    data is a Pandas DataFrameThe function assume data already has ['B,Bx,By,
    Bz] features
    '''
    data['theta'] = np.arctan2(data['Bz'],
                               np.sqrt(data['Bx']**2+data['By']**2))
    data['phi'] = np.arctan2(data['By'], data['Bz'])
    return data


def computeRollingMean(data, timeWindow, feature, center=False):
    '''
    for a given dataframe, compute the mean over a defined period of time (
    timeWindow) of a defined feature*
    ARGS :
    data : dataframe
    feature : feature in the dataframe we wish to compute the rolling mean from
                (string format)
    timeWindow : string that defines the length of the timeWindow (see pds doc)
    center : boolean to indicate if the point of the dataframe considered is
    center or end of the window
    '''
    name = feature+timeWindow+'mean'
    data[name] = data[feature].rolling(timeWindow, center=center).mean()
    return data


def computeRollingStd(data, timeWindow, feature, center=False):
    '''
    for a given dataframe, compute the standard dev over
    a defined period of time (timeWindow) of a defined feature*
    ARGS :
    data : dataframe
    feature : feature in the dataframe we wish to compute the rolling mean from
                (string format)
    timeWindow : string that defines the length of the timeWindow (see pds doc)
    center : boolean to indicate if the point of the dataframe considered is
    center or end of the window
    '''
    name = feature+timeWindow+'std'
    data[name] = data[feature].rolling(timeWindow, center=center,
                                       min_periods=1).std()
    return data


def computeRollingMax(data, timeWindow, feature, center=False):
    '''
    for a given dataframe, compute the maximum over a defined period of time (
    timeWindow) of a defined feature*
    ARGS :
    data : dataframe
    feature : feature in the dataframe we wish to compute the rolling mean from
                (string format)
    timeWindow : string that defines the length of the timeWindow (see pds doc)
    center : boolean to indicate if the point of the dataframe considered is
    center or end of the window
    '''
    name = feature+timeWindow+'max'
    data[name] = data[feature].rolling(timeWindow, center=center).max()
    return data


def computeRollingMin(data, timeWindow, feature, center=False):
    '''
    for a given dataframe, compute the minimum over a defined period of time (
    timeWindow) of a defined feature*
    ARGS :
    data : dataframe
    feature : feature in the dataframe we wish to compute the rolling mean from
                (string format)
    timeWindow : string that defines the length of the timeWindow (see pds doc)
    center : boolean to indicate if the point of the dataframe considered is
    center or end of the window
    '''
    name = feature+timeWindow+'min'
    data[name] = data[feature].rolling(timeWindow, center=center).max()
    return data


def computeRollingAmplitude(data, timeWindow, feature, center=False):
    '''
    for a given dataframe, compute the amplitude over a defined period of time(
    timeWindow) of a defined feature*
    ARGS :
    data : dataframe
    feature : feature in the dataframe we wish to compute the rolling mean from
                (string format)
    timeWindow : string that defines the length of the timeWindow (see pds doc)
    center : boolean to indicate if the point of the dataframe considered is
    center or end of the window
    '''
    name = feature+timeWindow+'Amplitude'
    data[name] = data[feature].rolling(timeWindow, center=center).max() - data[feature].rolling(timeWindow, center=center).min()
    return data


def computeRollingRMSBoB(data, timeWindow, center=False):
        '''
        for a given dataframe, compute the Fluctuation level of B
        over a defined period of time( timeWindow) the function assumes B is
        loaded with its components
        ARGS :
        data : dataframe
        feature : feature in the dataframe we wish to compute
                  the rolling mean from(string format)
        timeWindow : string that defines the length
                     of the timeWindow (see pds doc)
        center : boolean to indicate if the point of the
                 dataframe considered is center or end of the window
        '''
        try:
            components = ['Bx', 'By', 'Bz']
            rmsbob = 0
            for j in components:
                amean = data[j] - data[j].rolling(timeWindow, center=center).mean()
                amean = amean**2
                a = amean.rolling(timeWindow, center=center).mean()
                rmsbob += a
            data['rmsbob'+timeWindow] = np.sqrt(rmsbob)/data['B']
        except KeyError:
            print('Error computing the RmsBob, components or Amplitude of '
                  'B might not be loaded')
        return data


def computePrevious(data, feature, timeOffset='0.5H', holeCriteria='120S'):
    '''
    For a given df data and a given feature ( str format) add as features the
    previous steps during the specified timeOffset
    HoleCriteria indicate the expected df frequency, default value is '120S'
    '''
    numOfTimesteps = np.int(pds.to_timedelta(timeOffset) /
                            pds.to_timedelta(holeCriteria))
    for i in range(1, numOfTimesteps+1):
        name = feature+'-'+str(i)
        data[name] = np.NAN
        data[name][i:] = data[feature][0:-i]
        data[name][0:i] = np.zeros(i)
        timediff = data.index[i:]-data.index[0:-i]
        for j in range(0, i):
            timediff = timediff.insert(j, datetime.timedelta(0))
        data.loc[data.index.isin(data.index[timediff >
                 i*pds.to_timedelta(holeCriteria)]), name] = np.NAN
        data[name] = data[name].fillna(method='bfill')
    return data


def computeNext(data, feature, timeOffset='0.5H', holeCriteria='120S'):
    '''
    For a given df data and a given feature ( str format) add as features the
    next steps during the specified timeOffset
    HoleCriteria indicate the expected df frequency, default value is '120S'
    '''
    numOfTimesteps = np.int(pds.to_timedelta(timeOffset) /
                            pds.to_timedelta(holeCriteria))
    for i in range(1, numOfTimesteps+1):
        name = feature+'+'+str(i)
        data[name] = np.NAN
        data[name][0:-i] = data[feature][i:]
        data[name][-i:] = 0
        timediff = data.index[i:]-data.index[0:-i]
        for j in range(0, i):
            timediff = timediff.insert(-j, datetime.timedelta(0))
        data.loc[data.index.isin(data.index[timediff >
                 i*pds.to_timedelta(holeCriteria)]), name] = np.NAN
        data[name] = data[name].fillna(method='ffill')
    return data
