import pandas as pds
import datetime
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import time
import glob
import event


'''
The functions here are to be used to load a dataset from defined missions,
sync them, write them into a parquet file or eventually read a parquet file
works Currently for wind shall be implemented for ace
'''


def mkfilelist(startTime, endTime, instr, directory):
    '''
    #Creates a list with the name of the files between startTime and endTime
    Args :
    startTime as a datetime
    endTime as a datetime
    instr as a string in the value ['mfi_h0','swe_k0','swe_h1','3dp_plsp']
        additional values shall be implemented to add more space missions
    directory as string : stirng that indicate the location of specified data
    '''
    dateRange = [d for d in pds.date_range(startTime, endTime)]
    [inst, method] = instr.split("_")
    filenames = []
    for day in dateRange:
        try:
            filename = "%s%s/%d/wi_%s_%s_%d%02d%02d_v" % (directory, instr,
                                                          day.year,
                                                          method, inst,
                                                          day.year, day.month,
                                                          day.day)
            filename = glob.glob(filename+"*.cdf")[-1]
            filenames.append(filename)
        except IndexError:
            print("no file found for " + instr +
                  " at date "+day.strftime('%Y-%m-%d'))
    return filenames


def setSameTimeScale(dfList, method='interpolate', ref=0, dropna='any'):
    '''
    Set a list of Pandas dataframes and set them on the same timescale
    Data:  List of dataframes
    numinstr : number of considered instruments
    method : wished adjustment method : None or 'interpolate' for interpolation
             'shift' for datashifting
    '''

    if method == 'interpolate':
        merged_df = pds.concat(dfList).sort_index()
        for df in dfList:
            if df is not dfList[ref]:
                merged_df[df.columns] = \
                    merged_df[df.columns].interpolate(method='time')
        merged_df = merged_df[merged_df.index.isin(dfList[ref].index)].dropna(how=dropna)

    elif method == 'shift':
        merged_df = dfList[ref]
        for df in dfList:
            if df is not dfList[ref]:
                merged_df = \
                    merged_df.join(dfList[i].reindex(dfList[ref].index, method='nearest'))
    else:
        raise ValueError('Error : wrong argument specified for the method'
                         'correct value are "interpolate" or "shift"')
    return merged_df


def loadData(startTime, endTime, instruments, directory, ref):
    '''
    Create dataframes for each instrument
    startTime and endTime expressed as
    datetime.datetime(year, month, day, hour, min, sec)
    instruments as a string array [mfi_h0, swe_k0, swe_h1,3dp_plsp]
    features as a string array ['C','Beta','Theta','Phi','Malfven','rolling',
    'rollingMean', 'rollingStd' 'all']
    '''

    data = []
    fields = {'mfi_h0': wind.loadMFIh0, 'swe_k0': wind.loadSWEk0,
              'swe_h1': wind.loadSWEh1, '3dp_plsp': wind.load3DP}
    for instr in instruments:
        files = mkfilelist(startTime, endTime, instr, directory)
        datafr = pds.DataFrame()
        print('loading data')
        for file in files:
            data_tmp = fields[instr](file)
            datafr = datafr.append(data_tmp)
        data.append(datafr)
    data = setSameTimeScale(data, 'interpolate', ref)

    data = data[data.index > startTime]
    data = data[data.index < endTime]

    return data


def makeParquet(data, out_path):
    '''
    Convert a Pandas dataframe to a parquet file and save it
    data : Pandas DataFrame
    out_path : exit filename (string format)
    '''
    arrow_table = pa.Table.from_pandas(data)
    pq.write_table(arrow_table, out_path, use_dictionary=False)


def loadParquet(filename):
    '''
    load a parquet file and return it as a Pandas DataFrame,
    function might be modified to add features specific to space mission ?
    '''
    return pq.read_table(filename).to_pandas()


def labelByPoints(data, directory, label=1):
        '''
        Label a given Pandas dataframe with the help of the ICME list
        Directory is the path to a given event list, assume the list has the
        two column names 'begin' and 'end'
        '''
        evtDates = pds.read_csv(directory)
        evtDates['begin'] = pds.to_datetime(evtDates['begin'],
                                            format="%Y/%m/%d %H:%M")
        evtDates['end'] = pds.to_datetime(evtDates['end'],
                                          format="%Y/%m/%d %H:%M")
        evtDates = evtDates.reset_index()
        data['label'] = 0
        for i in range(0, len(evtDates)):
            data['label'][(evtDates['begin'][i] < data.index) &
                          (data.index < evtDates['end'][i])] = label
        return data


def removeList(data, directory):
    '''
    Remove a given list of time interval from a df data
    assums the list is in csv format and has two columns 'begin' and 'end'
    '''
    intervalList = pds.read_csv(directory)
    intervalList['begin'] = pds.to_datetime(intervalList['begin'],
                                            format="%Y/%m/%d %H:%M")
    intervalList['end'] = pds.to_datetime(intervalList['end'],
                                          format="%Y/%m/%d %H:%M")
    for i in range(o, len(intervalList)):
        data = data.drop(data[intervalList['begin'][i]:
                         data[intervalList['end'][i]]])
    return data
