### IMPORTS ###
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

### FUNCTIONS ###

def authorise_search(pipeline, X):
    '''Authorises a search whenever there is a probability
    greater than 10% that the search will be successful.
    '''
    authorise = pipeline.predict_proba(X)[:,1] > 0.1
    return authorise

def min_max_range(data):
    '''Retruns the range between the maximum and minimum values.'''
    mmr = data.max()-data.min()
    return mmr

def calculate_subgroup_precisions(y_pred, y_true, X,
                                  grouping=['station', 'ethnicity', 'gender']):
    '''Returns a dataframe with precision scores within subgroups.'''
    
    # merge data & drop gender 'other'
    df = pd.DataFrame(data={'station': X.station.values,
                            'ethnicity': X.ethnicity_officer.values,
                            'gender': X.sex.values,
                            'pred': y_pred,
                            'true': y_true})
    df = df[df.gender != 'Other']
    
    def truncated_precision(y_true, y_pred, min_values=30):
        '''Returns the precision score if input data has more than 'min_values' rows.
        Otherwise returns nan.
        '''
        if len(y_true) < min_values:
            precision = np.nan
        elif sum(y_pred) == 0:
            precision = np.nan
        else:
            precision = precision_score(y_true, y_pred)
        return precision
    
    # calculate scores
    df = df.groupby(grouping).apply(lambda x: truncated_precision(x.true, x.pred, min_values=30))
    return df

def within_station_discrepancy(y_pred, y_true, X):
    '''Returns a series with the maximum discrepancies within each police station.'''
    subgroup_precisions = calculate_subgroup_precisions(y_pred, y_true, X,
                                  grouping=['station', 'ethnicity', 'gender'])\
                                .unstack(['gender','ethnicity'])\
                                .T\
                                .apply(min_max_range)
    return subgroup_precisions

def across_station_discrepancy(y_pred, y_true, X):
    '''Returns the maximum discrepancy between stations.'''
    station_precisions = calculate_subgroup_precisions(y_pred, y_true, X,
                                  grouping=['station'])\
                                .to_frame()\
                                .apply(min_max_range)[0]
    return station_precisions

def across_subgroup_discrepancy(y_pred, y_true, X):
    '''Returns the maximum discrepancy between ['ethnicity', 'gender'] subgroups.'''
    station_precisions = calculate_subgroup_precisions(y_pred, y_true, X,
                                  grouping=['ethnicity', 'gender'])\
                                .to_frame()\
                                .apply(min_max_range)[0]
    return station_precisions
    