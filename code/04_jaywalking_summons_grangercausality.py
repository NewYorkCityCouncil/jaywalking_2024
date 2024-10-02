#!/usr/bin/env python
# coding: utf-8

# # Jaywalking Summons Granger Causality Analysis
# 
# This Jupyter Notebook performs a Granger causality analysis to investigate the relationship between jaywalking summonses and jaywalking collisions. The analysis aims to determine whether the number of jaywalking summonses can predict the number of jaywalking collisions and vice versa.
# 
# ## Overview
# 
# The notebook is divided into several sections:
# 
# 1. **Data Preparation**: 
#    - Load the dataset containing quarterly data on jaywalking collisions and summonses.
#    - Filter out outliers based on z-scores to ensure the analysis is not skewed by extreme values.
# 
# 2. **Data Visualization**:
#    - Plot the original and transformed data to visualize trends and relationships.
#    - Use Matplotlib to create line plots for the pedestrian share of total collisions and jaywalking summonses over time.
#    - Rotate x-tick labels for better readability and set the maximum number of x-ticks.
# 
# 3. **Granger Causality Test**:
#    - Perform the Granger causality test to determine if one time series can predict another.
#    - In order to perform GCT the data is transformed to be stationary using detrending by subtracting the moving average from the original data
#    - Interpret the results to understand the causal relationship between jaywalking summonses and collisions.
# 
# ## How It Works
# 
# - **Data Loading**: The dataset is loaded into a DataFrame, and relevant columns are extracted for analysis.
# - **Outlier Removal**: Outliers are filtered out by keeping data points within three standard deviations of the mean.
# - **Plotting**: The data is plotted to visualize trends. The plots include the pedestrian share of total collisions and jaywalking summonses over different quarters.
# - **Granger Causality Test**: The test is conducted to check if past values of one variable can predict the current values of another variable.
# 
# ## Usage
# 
# Run each cell sequentially to perform the analysis. Ensure that the required libraries (e.g., Matplotlib, NumPy) are installed in your environment.
# 
# ## Dependencies
# 
# - Matplotlib
# - NumPy
# - Pandas
# 

# In[1]:


import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import wkt
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller, kpss

# In[2]:


def test_stationarity(timeseries):
    # ADF Test
    print("Results of Dickey-Fuller Test:")
    adf_result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print('\t%s: %.3f' % (key, value))
    if adf_result[1] <= 0.05:
        print("Series is stationary according to ADF test")
    else:
        print("Series is not stationary according to ADF test")

    # KPSS Test
    print("\nResults of KPSS Test:")
    kpss_result = kpss(timeseries, regression='c')
    print('KPSS Statistic: %f' % kpss_result[0])
    print('p-value: %f' % kpss_result[1])
    print('Critical Values:')
    for key, value in kpss_result[3].items():
        print('\t%s: %.3f' % (key, value))
    if kpss_result[1] >= 0.05:
        print("Series is stationary according to KPSS test")
    else:
        print("Series is not stationary according to KPSS test")

# In[3]:


# jaywalking criminal summonses by quarter

jaywalking_crim_summonses_quarterly = pd.read_csv('../data/output/jaywalking_crim_summonses_quarterly.csv')

# In[4]:


# uploading crash data (has data on persons involved in collisions)

collisions_person_dataset = pd.read_csv('https://data.cityofnewyork.us/resource/f55k-p6yu.csv?$limit=9999999')

# In[5]:


collisions_person_dataset['crash_date'] = pd.to_datetime(collisions_person_dataset['crash_date'])

# creating quarter column 
collisions_person_dataset['quarter'] = collisions_person_dataset['crash_date'].dt.quarter.astype(str) 
collisions_person_dataset['quarter'] = 'Q' + collisions_person_dataset['quarter']
collisions_person_dataset['year'] = collisions_person_dataset['crash_date'].dt.year.astype(str)
collisions_person_dataset['quarter'] = collisions_person_dataset['year'] + ' ' + collisions_person_dataset['quarter']

# In[6]:


# will be used to normalize

collisions_per_quarter = collisions_person_dataset.groupby('quarter').count()[['unique_id']].rename(columns={'unique_id':'total_collisions'})

# In[7]:


# creating pedestrian collisions dataset

pedestrian_collisions = collisions_person_dataset[collisions_person_dataset['person_type'] == 'Pedestrian']
pedestrian_collisions['pedestrians_killed'] = np.where(pedestrian_collisions['person_injury'] == 'Killed', 1, 0)
pedestrian_collisions['pedestrians_injured'] = np.where(pedestrian_collisions['person_injury'] == 'Injured', 1, 0)
pedestrian_collisions['pedestrian_ksi'] = np.where(pedestrian_collisions['person_injury'].isin(['Injured', 'Killed']), 1, 0)

# In[8]:


# will be used to normalize

ped_collisions_per_quarter = pedestrian_collisions.groupby('quarter').count()[['unique_id']].rename(columns={'unique_id':'pedestrian_collisions'})

# In[9]:


# important to note that 32% of pedestrian collisions have 'ped_action' listed as null
# this means that subsetting the dataset based on 'ped_action' entries that indicate jaywalking will likely result in false negatives
# will run regression on both all pedestrian collisions AND all jaywalking collisions in case the results differ

round((100*pedestrian_collisions['ped_action'].value_counts(dropna=False) / len(pedestrian_collisions['ped_action'])),2)

# In[10]:


# with that in mind...
# narrowing down to just crashes involving jaywalkers (based on available data)
    # 1) crossing without a signal AND not in a crosswalk
    # 2) crossing in a crosswalk without a signal
    # 3) crossing without a signal

jaywalking_collisions = pedestrian_collisions[pedestrian_collisions['ped_action'].isin(['Crossing, No Signal, or Crosswalk', 'Crossing, No Signal, Marked Crosswalk', 'Crossing Against Signal'])]

# In[11]:


# investigating where these types of crashes occur

jaywalking_collisions[jaywalking_collisions['ped_action'] == 'Crossing, No Signal, or Crosswalk'][['ped_location']].value_counts(dropna=False)

# In[12]:


# investigating where these types of crashes occur

jaywalking_collisions[jaywalking_collisions['ped_action'] == 'Crossing, No Signal, Marked Crosswalk'][['ped_location']].value_counts(dropna=False)

# In[13]:


# investigating where these types of crashes occur

jaywalking_collisions[jaywalking_collisions['ped_action'] == 'Crossing Against Signal'][['ped_location']].value_counts(dropna=False)

# In[14]:


# grouping by quater
# pre-2016 Q2 seems badly affected by NaN issue
# should just look at 2016 and beyond for jaywalking dataset

jaywalking_collisions_per_quarter = jaywalking_collisions.groupby('quarter').agg({'unique_id': 'count', 'pedestrians_injured': 'sum', 'pedestrians_killed': 'sum', 'pedestrian_ksi': 'sum'}).rename(columns={'unique_id':'jaywalking_collisions'})
jaywalking_collisions_per_quarter = jaywalking_collisions_per_quarter.merge(ped_collisions_per_quarter, on='quarter', how='left')
jaywalking_collisions_per_quarter = jaywalking_collisions_per_quarter.merge(collisions_per_quarter, on='quarter', how='left')
jaywalking_collisions_per_quarter['jay_share_collisions'] = round((100*jaywalking_collisions_per_quarter['jaywalking_collisions'] / jaywalking_collisions_per_quarter['total_collisions']),2)
jaywalking_collisions_per_quarter['ped_share_collisions'] = round((100*jaywalking_collisions_per_quarter['pedestrian_collisions'] / jaywalking_collisions_per_quarter['total_collisions']),2)
jaywalking_collisions_per_quarter = jaywalking_collisions_per_quarter.merge(jaywalking_crim_summonses_quarterly, on='quarter', how='left')


# In[15]:


# eliminating pre-2016 quarters (plus not including 2024 Q3 because it's not over)

jaywalking_collisions_per_quarter = jaywalking_collisions_per_quarter.loc[5:38]

# In[16]:


# creating df without outliers

# identify outliers using Z-score
jaywalking_collisions_per_quarter['zscore_jaywalking_collisions'] = zscore(jaywalking_collisions_per_quarter['jaywalking_collisions'])
jaywalking_collisions_per_quarter['zscore_pedestrian_collisions'] = zscore(jaywalking_collisions_per_quarter['pedestrian_collisions'])
jaywalking_collisions_per_quarter['zscore_total_collisions'] = zscore(jaywalking_collisions_per_quarter['total_collisions'])
jaywalking_collisions_per_quarter['zscore_pedestrian_ksi'] = zscore(jaywalking_collisions_per_quarter['pedestrian_ksi'])
jaywalking_collisions_per_quarter['zscore_jay_share_collisions'] = zscore(jaywalking_collisions_per_quarter['jay_share_collisions'])
jaywalking_collisions_per_quarter['zscore_ped_share_collisions'] = zscore(jaywalking_collisions_per_quarter['ped_share_collisions'])
jaywalking_collisions_per_quarter['zscore_total_summonses'] = zscore(jaywalking_collisions_per_quarter['total_summonses'])


# In[17]:


# jaywalking collisions vs jaywalking summonses

# filter out outliers (keeping those within 3 standard deviations)
df_no_outliers = jaywalking_collisions_per_quarter[(np.abs(jaywalking_collisions_per_quarter['zscore_total_summonses']) < 3) & (np.abs(jaywalking_collisions_per_quarter['zscore_jaywalking_collisions']) < 3)]

# check if there are outliers
len(df_no_outliers) == len(jaywalking_collisions_per_quarter)

# In[18]:


# since there are no outliers, proceed

# plot lines 

quarter = jaywalking_collisions_per_quarter['quarter']
jaywalking_collisions = jaywalking_collisions_per_quarter['jaywalking_collisions']
jaywalking_summonses = jaywalking_collisions_per_quarter['total_summonses']

plt.plot(quarter, jaywalking_collisions, label = "Jaywalking Collisions") 
plt.plot(quarter, jaywalking_summonses, label = "Jaywalking Summonses") 

# Set maximum number of x-ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))

# rotate x-tick labels for better readability
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.legend() 
plt.show()

# In[19]:


# scatter plot

# add axis labels
plt.xlabel('Jaywalking Collisions (Per Quarter)')
plt.ylabel('Jaywalking Summonses (Per Quarter)')

plt.scatter(jaywalking_collisions, jaywalking_summonses)
plt.show()

# In[20]:


# pedestrian collisions vs jaywalking summonses

# filter out outliers (keeping those within 3 standard deviations)
df_no_outliers = jaywalking_collisions_per_quarter[(np.abs(jaywalking_collisions_per_quarter['zscore_total_summonses']) < 3) & (np.abs(jaywalking_collisions_per_quarter['zscore_pedestrian_collisions']) < 3)]

# check if there are outliers
len(df_no_outliers) == len(jaywalking_collisions_per_quarter)

# In[21]:


# since there are no outliers, proceed

# plot lines 

pedestrian_collisions = jaywalking_collisions_per_quarter['pedestrian_collisions']

plt.plot(quarter, pedestrian_collisions, label = "Pedestrian Collisions") 
plt.plot(quarter, jaywalking_summonses, label = "Jaywalking Summonses") 

# set maximum number of x-ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))

# rotate x-tick labels for better readability
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.legend() 
plt.show()

# In[22]:


# scatter plot

# add axis labels
plt.xlabel('Pedestrian Collisions (Per Quarter)')
plt.ylabel('Jaywalking Summonses (Per Quarter)')

plt.scatter(pedestrian_collisions, jaywalking_summonses)
plt.show()

# In[23]:


# jaywalking share of total collisions vs jaywalking summonses

# filter out outliers (keeping those within 3 standard deviations)
df_no_outliers = jaywalking_collisions_per_quarter[(np.abs(jaywalking_collisions_per_quarter['zscore_total_summonses']) < 3) & (np.abs(jaywalking_collisions_per_quarter['zscore_jay_share_collisions']) < 3)]

# check if there are outliers
len(df_no_outliers) == len(jaywalking_collisions_per_quarter)

# In[24]:


# since there are no outliers, proceed

# plot lines 

jay_share_collisions = jaywalking_collisions_per_quarter['jay_share_collisions']

plt.plot(quarter, jay_share_collisions, label = "Jaywalking Share of Total Collisions") 
plt.plot(quarter, jaywalking_summonses, label = "Jaywalking Summonses") 

# set maximum number of x-ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))

# rotate x-tick labels for better readability
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.legend() 
plt.show()

# In[25]:


jaywalking_collisions_per_quarter

# In[26]:


# scatter plot

# add axis labels
plt.xlabel('Jaywalking Share of Total Collisions (Per Quarter)')
plt.ylabel('Jaywalking Summonses (Per Quarter)')

plt.scatter(jay_share_collisions, jaywalking_summonses)
plt.show()

# In[27]:


test_stationarity(jaywalking_collisions_per_quarter['total_summonses'])

# In[28]:


test_stationarity(jaywalking_collisions_per_quarter['ped_share_collisions'])

# In[29]:


# Calculate moving average
x_ma = jaywalking_collisions_per_quarter['total_summonses'].rolling(window=2).mean()

# Detrend the data by subtracting the moving average from the original data
x_detrended = jaywalking_collisions_per_quarter['total_summonses'] - x_ma

# In[30]:


test_stationarity(x_detrended.dropna())

# In[31]:


# Calculate moving average
y_ma = jaywalking_collisions_per_quarter['ped_share_collisions'].rolling(window=2).mean()

# Detrend the data by subtracting the moving average from the original data
y_detrended = jaywalking_collisions_per_quarter['ped_share_collisions'] - y_ma

# In[32]:


test_stationarity(y_detrended.dropna())

# In[33]:


jaywalking_collisions_per_quarter

# In[34]:


jaywalking_collisions_per_quarter['y_detrended'] = y_detrended
jaywalking_collisions_per_quarter['x_detrended'] = x_detrended

# In[35]:


jaywalking_collisions_per_quarter 

# In[36]:


df = pd.DataFrame({'ped_share_collisions': y_detrended.dropna(),'total_summonses': x_detrended.dropna()})

# In[37]:


#Plotting original data

quarter = jaywalking_collisions_per_quarter['quarter'][1:]
jaywalking_summonses = jaywalking_collisions_per_quarter['total_summonses'][1:]
ped_share_collisions = jaywalking_collisions_per_quarter['ped_share_collisions'][1:]

plt.plot(quarter, ped_share_collisions, label = "Pedestrian Share of Total Collisions") 
plt.plot(quarter, jaywalking_summonses, label = "Jaywalking Summonses") 

# set maximum number of x-ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))

# rotate x-tick labels for better readability
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.legend() 
plt.show()

#Plotting transformed data

quarter = jaywalking_collisions_per_quarter['quarter'][1:]
jaywalking_summonses = df['total_summonses']
ped_share_collisions = df['ped_share_collisions']

plt.plot(quarter, ped_share_collisions, label = "Pedestrian Share of Total Collisions") 
plt.plot(quarter, jaywalking_summonses, label = "Jaywalking Summonses") 

# set maximum number of x-ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))

# rotate x-tick labels for better readability
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.legend() 
plt.show()

# In[38]:


# Result without 2016 Q2 and 2024 Q2

#Plotting original data

quarter = jaywalking_collisions_per_quarter['quarter'][2:]
jaywalking_summonses = jaywalking_collisions_per_quarter['total_summonses'][2:]
ped_share_collisions = jaywalking_collisions_per_quarter['ped_share_collisions'][2:]

plt.plot(quarter, ped_share_collisions, label = "Pedestrian Share of Total Collisions") 
plt.plot(quarter, jaywalking_summonses, label = "Jaywalking Summonses") 

# set maximum number of x-ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))

# rotate x-tick labels for better readability
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.legend() 
plt.show()

#Plotting transformed data

quarter = jaywalking_collisions_per_quarter['quarter'][2:-1]
jaywalking_summonses = df['total_summonses'][1:-1]
ped_share_collisions = df['ped_share_collisions'][1:-1]

plt.plot(quarter, ped_share_collisions, label = "Pedestrian Share of Total Collisions") 
plt.plot(quarter, jaywalking_summonses, label = "Jaywalking Summonses") 

# set maximum number of x-ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))

# rotate x-tick labels for better readability
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.legend() 
plt.show()

# The plots look relatively similar to one another (the same peaks and valleys, except for 2016 Q2 and 2024 Q2) therefore if the Granger Causality Tests returns to be true for the transformed data we can say that Jaywalking Summonses Granger-causes Pedestrian Share of Total Collisions.

# In[39]:


# Full results
test_results = grangercausalitytests(df, maxlag=1, verbose=True)

# In[40]:


test_results[1][1][1].summary()

# In[41]:


type(test_results[1][1][1])

# In[42]:


y_pred = test_results[1][1][1].predict()
y_actual = df['ped_share_collisions'][1:]
len(y_pred) == len(y_actual)

# In[43]:


import matplotlib.pyplot as plt

plt.plot(y_actual.index, y_actual, label='Actual')
plt.plot(y_actual.index, y_pred, label='Predicted')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Predicted vs Actual')
plt.legend()

plt.show()

# In[49]:


"""
Mean Absolute Percentage Error (MAPE) formula:
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_{\text{actual}, i} - y_{\text{pred}, i}}{y_{\text{actual}, i}} \right| \times 100
"""

# In[45]:


mean_error = np.mean(np.abs((y_actual - y_pred) / (y_actual+1e-6))) * 100
print(f"The MAPE when including zeros is {mean_error}")

# In[46]:


df1 = pd.DataFrame({'actual': y_actual, 'predicted': y_pred})
df1 = df1[df1['actual'] != 0]
print(f"The MAPE when not including zeros is {np.mean(np.abs((df1['actual'] - df1['predicted']) / (y_actual))) * 100}")
