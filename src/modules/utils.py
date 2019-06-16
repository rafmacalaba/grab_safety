import pandas as pd
import numpy as np

# I'm using 16GB so use tricks for better memory handling.
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def interaction_features(df, fea1, fea2, prefix):
    """
    Basic interaction feature combination using selected features from dataset
    
    Returns:
        DataFrame
    """
    df['interaction_mul_{}'.format(prefix)] = df[fea1] * df[fea2]
    df['interaction_div_{}'.format(prefix)] = df[fea1] / df[fea2]

    return df

def data_aggregation_in_minutes(data, minute=1):
    """
    Data aggregation of seconds converting to minutes
    
    Input and Return:
        DataFrame
    """
     # aggregate the data using the seconds column maybe there's signal if the time interval is standardized.
    data.sort_values(['bookingid', 'second'], ascending=True)
    data['minute'] = np.round(data['second'] / (minute * 60))
    agg_data = data.groupby(['bookingid', 'minute'])\
        .agg([np.mean, min, max, np.std, sum])
    
    agg_data.columns = [f'_minute{minute}_'.join(col).strip() for col in agg_data.columns.values]    
    return agg_data.reset_index()