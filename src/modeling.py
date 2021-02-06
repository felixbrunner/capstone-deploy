### IMPORTS ###

import pandas as pd
import numpy as np
import datetime as dt

### FUNCTIONS ###

coordinate_dict = \
{'lat': {'avon-and-somerset': 51.33301534853675,
  'bedfordshire': 51.99520005035238,
  'btp': 52.04498213727202,
  'cambridgeshire': 52.401295645642094,
  'cheshire': 53.27243561712015,
  'city-of-london': 51.515283227020134,
  'cleveland': 54.581267715107955,
  'cumbria': 54.53797709350655,
  'derbyshire': 53.01032026513161,
  'devon-and-cornwall': 50.53555175443321,
  'dorset': 50.71701186632197,
  'durham': 54.68330272246215,
  'dyfed-powys': 52.06194421428571,
  'essex': 51.72007429847026,
  'gloucestershire': 51.84977152248181,
  'greater-manchester': 53.47859898644197,
  'gwent': 51.60761234256556,
  'hampshire': 50.93652569163378,
  'hertfordshire': 51.76037049985289,
  'humberside': 53.48481868431375,
  'kent': 51.374468082246494,
  'lancashire': 53.78500562755108,
  'leicestershire': 52.65751631679406,
  'lincolnshire': 53.10683554523824,
  'merseyside': 53.436241618732225,
  'metropolitan': 52.515362957924715,
  'norfolk': 52.64006103976567,
  'north-wales': 53.17569351116731,
  'north-yorkshire': 54.041700087409595,
  'northamptonshire': 52.29232862938288,
  'northumbria': 54.98786282962144,
  'nottinghamshire': 52.98302869082842,
  'south-yorkshire': 52.515362957924715,
  'staffordshire': 52.87808759739836,
  'suffolk': 52.139193867362344,
  'surrey': 51.32796300224013,
  'sussex': 50.90980622282589,
  'thames-valley': 51.69650944703028,
  'warwickshire': 52.36734886374945,
  'west-mercia': 52.413887931124385,
  'west-yorkshire': 53.77299550808837,
  'wiltshire': 51.37899916477772},
 'long': {'avon-and-somerset': -2.6959433559473505,
  'bedfordshire': -0.4249400266868079,
  'btp': -0.7715286391456624,
  'cambridgeshire': -0.04725813646789005,
  'cheshire': -2.6387643686903965,
  'city-of-london': -0.0896220290722526,
  'cleveland': -1.2457966712230222,
  'cumbria': -3.136673164935065,
  'derbyshire': -1.4773875202376467,
  'devon-and-cornwall': -4.066873729177866,
  'dorset': -2.0917077684638166,
  'durham': -1.594638016918649,
  'dyfed-powys': -4.084971542857141,
  'essex': 0.538986324416861,
  'gloucestershire': -2.188883543046356,
  'greater-manchester': -2.248505129745514,
  'gwent': -3.0290781326530665,
  'hampshire': -1.1898053439685032,
  'hertfordshire': -0.2575871883632099,
  'humberside': -0.4107479908496726,
  'kent': 0.5604151309456336,
  'lancashire': -2.7154059084062276,
  'leicestershire': -1.1627258218159886,
  'lincolnshire': -0.35191183908730256,
  'merseyside': -2.9378837346396676,
  'metropolitan': -1.3420896304429029,
  'norfolk': 1.1136251113436533,
  'north-wales': -3.5303938708385822,
  'north-yorkshire': -1.1725005511775388,
  'northamptonshire': -0.8297320040567943,
  'northumbria': -1.5853070855440516,
  'nottinghamshire': -1.1530782153432024,
  'south-yorkshire': -1.3420896304429029,
  'staffordshire': -2.0221622171637286,
  'suffolk': 1.0365897135618494,
  'surrey': -0.4413027473143175,
  'sussex': -0.16365491495299755,
  'thames-valley': -0.9660928864365886,
  'warwickshire': -1.4958054560654521,
  'west-mercia': -2.3846032813072715,
  'west-yorkshire': -1.6477957410954118,
  'wiltshire': -1.9166443269398472}}

def fill_coordinates_with_station_means(X):
    '''Fills missing coordinates with mean coordinates of respective station.'''
    X_ = X.copy()[['lat', 'long']]
    X_.lat = X_.lat.fillna(X.station.map(coordinate_dict['lat'])).values
    X_.long = X_.long.fillna(X.station.map(coordinate_dict['long'])).values
    return X_

def grid_to_category(X):
    '''Combines longitude and latitude categories into a single feature.'''
    X_ = np.array([','.join(row) for row in X.astype(int).astype(str)]).reshape(-1, 1)
    return X_

def extract_datetime_features(X):
    '''Extracts features from Date column:
    - hour
    - day of the week
    - days after sample start
    - square root of days since sample start
    '''
    X_ = pd.DataFrame(index=X.index)
    X_['hour'] = X['date'].dt.hour
    X_['weekday'] = X['date'].dt.weekday
    X_['daycount'] = (X['date'].dt.date - dt.date(2017, 12, 1)).dt.days
    X_['sqrt_daycount'] = X_.daycount**0.5
    return X_