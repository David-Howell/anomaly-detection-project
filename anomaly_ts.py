import numpy as np
import pandas as pd
import env
import matplotlib.pyplot as plt




def acquire_logs(user=env.username, password=env.password, host=env.host):
    '''    
    Uses the env.py information and pandas.read_sql to get the:
                
                date, path, user, cohort, ip
                
                FROM logs
    
    This is returned as a DataFrame
    
    Parameters :
    ------------
    
    None are needed, as the user, password, and host are predefined and pulled from env.py
    '''
#     Set the url using env
    url = f'mysql+pymysql://{env.username}:{env.password}@{env.host}/curriculum_logs'
#     Set the query
    query = '''
    SELECT date,
           path as endpoint,
           user_id,
           cohort_id,
           ip as source_ip
    FROM logs;
    '''
#     use pandas to read SQL into a DataFrame
    df = pd.read_sql(query, url)
#     return the DataFrame
    return df



def one_user_df_prep(df, user):
    '''
    This prepares a DataFrame with one given user's information.
    It resamples the data by day, and as such will show us gaps in the user's site access
    
    Parameters :
    ------------
    
    df : The dataframe you want to use
    
    user : the user within the dataset that you want to look at
    
    Returns    :
    ------------
    
    The prepared DataFrame for the given User
    
    '''
#     saves a copy of the DataFrame where the user_id is the specified user
    df = df[df.user_id == user].copy()
#     set the date column to datetime
    df.date = pd.to_datetime(df.date)
#     sets the index as the date
    df = df.set_index(df.date)
#     counts the pages accessed by the user on each day
#        Because of the resample, this will also show 0 on days that the curriculum wasn't accessed
    pages_one_user = df['endpoint'].resample('d').count()
#     Returns the DataFrame of the single user for the entire time period
    return pages_one_user




def compute_pct_b(pages_one_user, span, weight, user):
    '''
    
    Parameters :
    ------------
    pages_one_user  : This is the single user dataframe showing the counts of page visits
    
    span            : The span of time in units based on the dataset
    
    weight          : The weight multiplier for the 𝞂 
    
    user            : The User to compute the %b for 
    
    Returns    :
    ------------
    
    DataFrame with the midband, standard deviation, upper bound, lower bound, %b, and user
    
    '''
#     the midband is mean from the .ewm()
    midband = pages_one_user.ewm(span=span).mean()
#     the stdev is the standard deviation from the .ewm()
    stdev = pages_one_user.ewm(span=span).std()
#     upper bound is found by adding the weighted std to the midband
    ub = midband + stdev*weight
#     lower bound is found by subtracting the weighted std from the midband
    lb = midband - stdev*weight
    
    bb = pd.concat([ub, lb], axis=1)
    
    my_df = pd.concat([pages_one_user, midband, bb], axis=1)
    my_df.columns = ['pages_one_user', 'midband', 'ub', 'lb']
    
    my_df['pct_b'] = (my_df['pages_one_user'] - my_df['lb'])/(my_df['ub'] - my_df['lb'])
    my_df['user_id'] = user
    return my_df




def plot_bands(my_df, user):
    '''
    
    
    Parameters :
    ------------
    
    
    Returns    :
    ------------
    
    '''
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(my_df.index, my_df.pages_one_user, label='Number of Pages, User: '+str(user))
    ax.plot(my_df.index, my_df.midband, label = 'EMA/midband')
    ax.plot(my_df.index, my_df.ub, label = 'Upper Band')
    ax.plot(my_df.index, my_df.lb, label = 'Lower Band')
    ax.legend(loc='best')
    ax.set_ylabel('Number of Pages')
    plt.show()

    
    
    
def find_anomalies(df, user, span, weight, plot=False):
    '''
    
    
    Parameters :
    ------------
    
    
    Returns    :
    ------------
    
    '''
    
    pages_one_user = one_user_df_prep(df, user)
    
    my_df = compute_pct_b(pages_one_user, span, weight, user)
    
    if plot:
        plot_bands(my_df, user)
    
    return my_df[my_df.pct_b>1]