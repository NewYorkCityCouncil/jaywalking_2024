#!/usr/bin/env python
# coding: utf-8

# The `jaywalking_comparison_analysis.ipynb` notebook, located in the code/ directory, performs a comparative analysis of jaywalking incidents across different datasets. The analysis includes data from Virginia and Denver, as indicated by the variables `virginia_df` and `denver_gdf`.
# 
# ### Key Components:
# 1. **Data Loading and Preparation**:
#    - The notebook reads various datasets, including CSV and GeoJSON files, to create dataframes for analysis.
#    - Example datasets include `CrashData_test_6478750435646127290.csv` and `Traffic_Accidents_(Offenses).geojson`.
#    - Further transformations are performed to capture information on intervention (decriminalization or legalization of jaywalking) and incident counts.
#     - T (Month)	- counting up from 0 for the earliest entry then continuing 1,2,3,etc
#     - D (Intervention)	- 0 if the intervention has not taken place yet and 1 if it has
#     - P (Intervention month) - number of months since intervention
# 
# 2. **Data Analysis**:
#    - The analysis includes aggregating incident counts by month and year, as shown in the excerpt:
#      ```plaintext
#      month  year  total_incidents  total_pedestrian_incidents
#      1      2013  1855             68
#      2      2013  1571             42
#      ...
#      3      2024  1431             37
#      ```
#    - The notebook calculates the percentage of pedestrian incidents relative to total incidents.
# 
# 3. **Modeling and Statistical Analysis**:
#    - The notebook creates two models for each dataset: one which contains T, D, and P and is trained on the full dataset, and another (the counterfactual) which contains only T and is trained on the dataset up to the intervention.
#    - The results are then graphed to show the relationship between the intervention and pedestrian incidents, and the counterfactuals show the expected number of incidents if the intervention had not taken place.
#      
# 
# ### Observations:
# #### Denver:
# - Pedestrian-Vehicle Crashes / Total Crashes each month
#     - Small (less than 1%) statistically significant increase in pedestrian involved crashes each month and the amount of months since jaywalking was decriminalized 
# -  Non-Intersection Pedestrian-Vehicle Crashes / Total Crashes each month
#      - No statistically significant relationship between decriminalization and non-intersection pedestrian crashes  
# 
# #### Virginia Beach:
# - Pedestrian-Vehicle Crashes / Total Crashes each month
#      - Small (less than 1%) statistically significant increase in pedestrian involved crashes each month and the amount of months since jaywalking was legalized, but also small (less than 1%) statistically significant decrease in pedestrian involved crashes each month and whether jaywalking was legal that month
# - Non-Intersection Pedestrian-Vehicle Crashes / Total Crashes each month
#      - No statistically significant relationship between decriminalization and non-intersection pedestrian crashes  
# 
# ### Recommendations:
# - Further transform the data to improve results and distribution of residuals.
# - Rerun the analysis in R and compare the results to ensure consistency and accuracy.
# 
# 
# 

# # Import Libraries

# In[26]:


import requests
import pandas as pd
import json
import geopandas as gpd
import re
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# # Denver

# ## Read in data

# In[106]:


# https://drive.google.com/file/d/1F5--R6Gz2B1w7HNlR55RzErPHojYvIWG/view?usp=drive_link
denver_gdf = gpd.read_file("../data/input/Traffic_Accidents_(Offenses).geojson")

# In[8]:


denver_gdf

# ## Data Cleaning

# In[107]:


denver_gdf['occurrence_date'] = pd.to_datetime(denver_gdf['first_occurrence_date'])

# In[108]:


# Create a new column for month and year
denver_gdf['month'] = denver_gdf['occurrence_date'].dt.month
denver_gdf['year'] = denver_gdf['occurrence_date'].dt.year

# Group by month and year, and count the occurrences
summary_table = denver_gdf.groupby(['month', 'year']).count()['object_id']

# Display the summary table
summary_table = summary_table.reset_index().rename(columns={'object_id':'total_incidents'})

# In[109]:


# Filter to only pedestrian events
denver_gdf['pedestrian_related'] = (denver_gdf['pedestrian_ind'] > 0) | (denver_gdf['HARMFUL_EVENT_SEQ_1'].str.contains(r'PEDESTRIAN', flags=re.IGNORECASE)) | (denver_gdf['HARMFUL_EVENT_SEQ_2'].str.contains(r'PEDESTRIAN', flags=re.IGNORECASE)) | (denver_gdf['HARMFUL_EVENT_SEQ_3'].str.contains(r'PEDESTRIAN', flags=re.IGNORECASE)) | (denver_gdf['top_traffic_accident_offense'].str.contains(r'PEDESTRIAN', flags=re.IGNORECASE))

# In[110]:


denver_gdf[denver_gdf['pedestrian_related']==True]

# In[111]:


denver_gdf[denver_gdf['pedestrian_related']==True]['ROAD_DESCRIPTION'].unique()

# In[112]:


# Create a new column to identify if the accident occurred at an intersection
denver_gdf['at_intersection'] = denver_gdf['ROAD_DESCRIPTION'].str.contains('AT INTERSECTION|INTERSECTION RELATED|At Intersection|Intersection Related', regex=True, flags=re.IGNORECASE)


# In[113]:


# Create summary table for total pedestrian incidents
summary_table1 = denver_gdf[denver_gdf['pedestrian_related']==True].groupby(['month', 'year']).count()['object_id']
summary_table1 = summary_table1.reset_index().rename(columns={'object_id':'total_pedestrian_incidents'})
summary_table1

# In[124]:


# Create summary table for total incidents
denver_incidents_per_month = pd.merge(summary_table, summary_table1, how='outer', left_on=['month','year'], right_on=['month','year'])
denver_incidents_per_month

# In[125]:


# Create summary table for total pedestrian incidents at non-intersections
summary_table2 = denver_gdf[(denver_gdf['pedestrian_related']==True) & (denver_gdf['at_intersection']==False)].groupby(['month', 'year']).count()['object_id']
summary_table2 = summary_table2.reset_index().rename(columns={'object_id':'total_pedestrian_nonintersection_incidents'})
summary_table2

# In[126]:


# Merge summary tables together
denver_incidents_per_month = pd.merge(denver_incidents_per_month, summary_table2, how='outer', left_on=['month','year'], right_on=['month','year'])
denver_incidents_per_month

# In[127]:


# Calculate percentage of pedestrian incidents
denver_incidents_per_month['percentage_pedestrian_incidents'] = denver_incidents_per_month['total_pedestrian_incidents'] / denver_incidents_per_month['total_incidents']
denver_incidents_per_month

# In[128]:


# Calculate percentage of pedestrian incidents at non-intersections
denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'] = denver_incidents_per_month['total_pedestrian_nonintersection_incidents'] / denver_incidents_per_month['total_incidents']
denver_incidents_per_month

# In[129]:


# Sort by year and month
denver_incidents_per_month = denver_incidents_per_month.sort_values(by=['year','month'])
denver_incidents_per_month

# In[130]:


# Create feature for time (T)
denver_incidents_per_month = denver_incidents_per_month.reset_index().drop(columns='index').reset_index().rename(columns={'index':'T'})
denver_incidents_per_month

# In[131]:


# Create feature for intervention (D)
denver_incidents_per_month['D'] = denver_incidents_per_month.apply(lambda row: 1 if (row['month'] > 1 and row['year'] == 2023) or (row['year'] > 2023) else 0, axis=1)
denver_incidents_per_month

# In[132]:


# Create feature for time since intervention (P)
denver_incidents_per_month['P'] = denver_incidents_per_month['D'].cumsum()
denver_incidents_per_month

# In[133]:


# Create feature for time since intervention - 1 (P_prime) to test for multicollinearity in the model
denver_incidents_per_month['P_prime'] = denver_incidents_per_month.apply(lambda x: x['P'] - 1 if x['D'] != 0 else 0, axis=1)

# ## Modeling

# ### Pedestrian-Vehicle Crashes

# In[134]:


def test_stationarity(timeseries):
    """
    Perform the Augmented Dickey-Fuller test to check the stationarity of a time series.

    Parameters:
    timeseries (array-like): The time series data to be tested.

    Returns:
    None
    """
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is not stationary")

# In[135]:


test_stationarity(denver_incidents_per_month['percentage_pedestrian_incidents'])

# In[136]:


test_stationarity(denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'])

# In[137]:


plot_acf(denver_incidents_per_month['percentage_pedestrian_incidents'])
plt.show()

# ACF Plot shows autocorrelation at 1 and 2 lags so we will use ARIMA instead of OLS

# In[138]:


plot_acf(denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'])
plt.show()

# ACF Plot shows autocorrelation at 1 lag so we will use ARIMA instead of OLS

# In[47]:


# Test with no seasonality and P variable
auto_model_full = pm.auto_arima(denver_incidents_per_month['percentage_pedestrian_incidents'], 
                      X=denver_incidents_per_month[['T','D','P']],
                      start_p=0, start_q=0,
                      test='adf',       # Use ADF test to find optimal 'd'
                      max_p=5, max_q=5, # Maximum p and q
                      m=1,              # Frequency of the series
                      d=None,           # Let the model determine 'd'
                      seasonal=False,   # No seasonality
                      trace=True,       # Print status on the fits
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    # Stepwise selection to find the best model

auto_model_full.summary()

# In[158]:


# Test with seasonality and P variable
auto_model_full = pm.auto_arima(denver_incidents_per_month['percentage_pedestrian_incidents'], 
                      X=denver_incidents_per_month[['T','D','P']],
                      start_p=0, start_q=0,
                      test='adf',       # Use ADF test to find optimal 'd'
                      max_p=5, max_q=5, # Maximum p and q
                      m=12,              # Frequency of the series
                      d=None,           # Let the model determine 'd'
                      seasonal=True,   # No seasonality
                      trace=True,       # Print status on the fits
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    # Stepwise selection to find the best model

auto_model_full.summary()

# In[159]:


# Test with no seasonality and P_prime variable
auto_model_full = pm.auto_arima(denver_incidents_per_month['percentage_pedestrian_incidents'], 
                      X=denver_incidents_per_month[['T','D','P_prime']],
                      start_p=0, start_q=0,
                      test='adf',       # Use ADF test to find optimal 'd'
                      max_p=5, max_q=5, # Maximum p and q
                      m=1,              # Frequency of the series
                      d=None,           # Let the model determine 'd'
                      seasonal=False,   # No seasonality
                      trace=True,       # Print status on the fits
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    # Stepwise selection to find the best model

auto_model_full.summary()

# In[150]:


# Test with seasonality and P_prime variable
auto_model_full = pm.auto_arima(denver_incidents_per_month['percentage_pedestrian_incidents'], 
                      X=denver_incidents_per_month[['T','D','P_prime']],
                      start_p=0, start_q=0,
                      test='adf',       # Use ADF test to find optimal 'd'
                      max_p=12, max_q=12, # Maximum p and q
                      m=12,              # Frequency of the series
                      d=None,           # Let the model determine 'd'
                      seasonal=True,   # No seasonality
                      trace=True,       # Print status on the fits
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    # Stepwise selection to find the best model

auto_model_full.summary()

# In[37]:


# Check residuals
residuals = auto_model_full.resid()
plt.figure()
plt.hist(residuals)
plt.title('Histogram of Residuals')
plt.show()

# In[85]:


# Plot before and after intervention predicitons 
# auto_model_full is the model with no seasonality and P variable

start = 120
end = len(denver_incidents_per_month)

arima_results = ARIMA(denver_incidents_per_month['percentage_pedestrian_incidents'], denver_incidents_per_month[['T','D','P']], order=auto_model_full.order).fit()

predictions = arima_results.get_prediction(0, end-1)

# Use auto_arima to find the best ARIMA model parameters before the intervention
auto_model_cf = pm.auto_arima(denver_incidents_per_month['percentage_pedestrian_incidents'][:start], exogenous=denver_incidents_per_month[["T"][:start]],
                        start_p=0, start_q=0,
                        test='adf',       # Using ADF test to find optimal 'd'
                        max_p=5, max_q=5, # Search max bounds for p and q
                        m=1,              # Non-seasonal data
                        d=None,           # Let auto_arima determine 'd'
                        seasonal=False,   # No seasonality
                        stepwise=True,    # Use the stepwise algorithm
                        trace=True,       # Print status on the fits
                        error_action='ignore',  
                        suppress_warnings=True)

arima_cf = ARIMA(denver_incidents_per_month['percentage_pedestrian_incidents'][:start], denver_incidents_per_month["T"][:start], order=auto_model_cf.order).fit()

# Model predictions means
y_pred = predictions.predicted_mean

# Counterfactual mean and 95% confidence interval
y_cf = arima_cf.get_forecast(steps=end-start, exog=denver_incidents_per_month["T"][start:end]).summary_frame(alpha=0.05)

# Plot section
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(16,10))

# Plot bounce rate data
ax.scatter(denver_incidents_per_month["T"], denver_incidents_per_month["percentage_pedestrian_incidents"], facecolors='none', edgecolors='steelblue', label="Traffic accident data", linewidths=2)

# Plot model predictions
# ax.plot(incidents_per_month["T"][:end], y_pred, 'b-', label="Model prediction")
ax.plot(denver_incidents_per_month["T"][:start], y_pred[:start], 'b-', label="Model prediction")
ax.plot(denver_incidents_per_month["T"][start:end], y_pred[start:], 'b-')

# Plot counterfactual predictions with 95% confidence interval
ax.plot(denver_incidents_per_month["T"][start:end], y_cf["mean"], 'k.', label="counterfactual")
ax.fill_between(denver_incidents_per_month["T"][start:end], y_cf['mean_ci_lower'], y_cf['mean_ci_upper'], color='k', alpha=0.1, label="Counterfactual 95% CI")

# Intervention line
ax.axvline(x=denver_incidents_per_month["T"][start], color='r', linestyle='--', label='Intervention')

# Labels and legends
ax.legend(loc='best')
plt.xlabel("Months")
plt.ylabel("Pedestrian-involved traffic accidents (%)")
plt.show()

# ### Non-Intersection Pedestrian-Vehicle Crashes

# In[229]:


# Plot before and after intervention predicitons 
# auto_model_full is the model with no seasonality and P variable

auto_model_full = pm.auto_arima(denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'], 
                      X=denver_incidents_per_month[['T','D','P']],
                      start_p=0, start_q=0,
                      test='adf',       # Use ADF test to find optimal 'd'
                      max_p=20, max_q=20, # Maximum p and q
                      m=1,              # Frequency of the series
                      d=None,           # Let the model determine 'd'
                      seasonal=False,   # No seasonality
                      trace=True,       # Print status on the fits
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    # Stepwise selection to find the best model

print(auto_model_full.summary())

# Check residuals
residuals = auto_model_full.resid()
plt.figure()
plt.hist(residuals)
plt.title('Histogram of Residuals')
plt.show()
start = 120
end = len(denver_incidents_per_month)

arima_results = ARIMA(denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'], denver_incidents_per_month[['T','D','P']], order=auto_model_full.order).fit()

predictions = arima_results.get_prediction(0, end-1)

# Use auto_arima to find the best ARIMA model parameters before the intervention
auto_model_cf = pm.auto_arima(denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'][:start], exogenous=denver_incidents_per_month[["T"][:start]],
                        start_p=0, start_q=0,
                        test='adf',       # Using ADF test to find optimal 'd'
                        max_p=20, max_q=20, # Search max bounds for p and q
                        m=1,              # Non-seasonal data
                        d=None,           # Let auto_arima determine 'd'
                        seasonal=False,   # No seasonality
                        stepwise=True,    # Use the stepwise algorithm
                        trace=True,       # Print status on the fits
                        error_action='ignore',  
                        suppress_warnings=True)

arima_cf = ARIMA(denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'][:start], denver_incidents_per_month["T"][:start], order=auto_model_cf.order).fit()

# Model predictions means
y_pred = predictions.predicted_mean

# Counterfactual mean and 95% confidence interval
y_cf = arima_cf.get_forecast(steps=end-start, exog=denver_incidents_per_month["T"][start:end]).summary_frame(alpha=0.05)

# Plot section
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(16,10))

# Plot bounce rate data
ax.scatter(denver_incidents_per_month["T"], denver_incidents_per_month["percentage_pedestrian_nonintersection_incidents"], facecolors='none', edgecolors='steelblue', label="Traffic accident data", linewidths=2)

# Plot model predictions
# ax.plot(incidents_per_month["T"][:end], y_pred, 'b-', label="Model prediction")
ax.plot(denver_incidents_per_month["T"][:start], y_pred[:start], 'b-', label="Model prediction")
ax.plot(denver_incidents_per_month["T"][start:end], y_pred[start:], 'b-')

# Plot counterfactual predictions with 95% confidence interval
ax.plot(denver_incidents_per_month["T"][start:end], y_cf["mean"], 'k.', label="counterfactual")
ax.fill_between(denver_incidents_per_month["T"][start:end], y_cf['mean_ci_lower'], y_cf['mean_ci_upper'], color='k', alpha=0.1, label="Counterfactual 95% CI")

# Intervention line
ax.axvline(x=denver_incidents_per_month["T"][start], color='r', linestyle='--', label='Intervention')

# Labels and legends
ax.legend(loc='best')
plt.xlabel("Months")
plt.ylabel("Non-Intersection Pedestrian-involved traffic accidents (%)")
plt.show()

# In[141]:


# Model with no seasonality and P variable

auto_model_full = pm.auto_arima(denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'], 
                      X=denver_incidents_per_month[['T','D','P']],
                      start_p=0, start_q=0,
                      test='adf',       # Use ADF test to find optimal 'd'
                      max_p=20, max_q=20, # Maximum p and q
                      m=1,              # Frequency of the series
                      d=None,           # Let the model determine 'd'
                      seasonal=False,   # No seasonality
                      trace=True,       # Print status on the fits
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    # Stepwise selection to find the best model

auto_model_full.summary()

# In[155]:


# Model with no seasonality and P_prime variable

auto_model_full = pm.auto_arima(denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'], 
                      X=denver_incidents_per_month[['T','D','P']],
                      start_p=0, start_q=0,
                      test='adf',       # Use ADF test to find optimal 'd'
                      max_p=20, max_q=20, # Maximum p and q
                      m=1,              # Frequency of the series
                      d=None,           # Let the model determine 'd'
                      seasonal=False,   # No seasonality
                      trace=True,       # Print status on the fits
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    # Stepwise selection to find the best model

auto_model_full.summary()

# # Virginia Beach

# ## Read in data

# In[2]:


# https://drive.google.com/file/d/1DBFAzfgXdHNaCiXlfn859P3PSde_LokI/view?usp=drive_link
virginia_df = pd.read_csv("../data/input/CrashData_test_6478750435646127290.csv")
virginia_df

# In[3]:


# Filter to only Virginia Beach
vb_df = virginia_df[virginia_df['Physical Juris Name'].str.contains('Virginia Beach', flags=re.IGNORECASE)]

# In[4]:


vb_df

# ## Data Cleaning

# In[5]:


# Convert to datetime
vb_df['Crash Datetime'] = pd.to_datetime(vb_df['Crash Date'])

# In[6]:


# Create month column
vb_df['crash_month'] = vb_df['Crash Datetime'].dt.month

# In[7]:


# Create table with total crashes per month and year
vb_total_crashes_month_year = vb_df.groupby(['crash_month', 'Crash Year']).count()['OBJECTID']
vb_total_crashes_month_year = vb_total_crashes_month_year.reset_index().rename(columns={'OBJECTID':'total_crashes'})

# In[8]:


# Create table with total pedestrian crashes per month and year
vb_total_pedesrian_crashes_month_year = vb_df[vb_df['Pedestrian?']=='Yes'].groupby(['crash_month', 'Crash Year']).count()['OBJECTID']
vb_total_pedesrian_crashes_month_year = vb_total_pedesrian_crashes_month_year.reset_index().rename(columns={'OBJECTID':'total_pedestrian_crashes'})

# In[9]:


# Create table with total pedestrian crashes at non-intersections per month and year
vb_total_pedestrian_nonintersection_crashes_month_year = vb_df[(vb_df['Pedestrian?']=='Yes') & (vb_df['Intersection Type']=='1. Not at Intersection')].groupby(['crash_month', 'Crash Year']).count()['OBJECTID']
vb_total_pedestrian_nonintersection_crashes_month_year = vb_total_pedestrian_nonintersection_crashes_month_year.reset_index().rename(columns={'OBJECTID':'total_pedestrian_nonintersection_crashes'})

# In[10]:


# Merge tables together
vb_incidents_per_month = pd.merge(vb_total_crashes_month_year, vb_total_pedesrian_crashes_month_year, how='outer', left_on=['crash_month','Crash Year'], right_on=['crash_month','Crash Year'])
vb_incidents_per_month = pd.merge(vb_incidents_per_month, vb_total_pedestrian_nonintersection_crashes_month_year, how='outer', left_on=['crash_month','Crash Year'], right_on=['crash_month','Crash Year'])
vb_incidents_per_month

# In[11]:


# Calculate percentage of pedestrian crashes
vb_incidents_per_month['percentage_pedestrian_crashes'] = vb_incidents_per_month['total_pedestrian_crashes'] / vb_incidents_per_month['total_crashes']

# In[12]:


# Calculate percentage of pedestrian crashes at non-intersections
vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'] = vb_incidents_per_month['total_pedestrian_nonintersection_crashes'] / vb_incidents_per_month['total_crashes']

# In[13]:


# Sort by year and month
vb_incidents_per_month = vb_incidents_per_month.sort_values(by=['Crash Year','crash_month'])

# In[14]:


# Create T, D, and P features
vb_incidents_per_month = vb_incidents_per_month.reset_index().drop(columns='index').reset_index().rename(columns={'index':'T'})
vb_incidents_per_month['D'] = vb_incidents_per_month.apply(lambda row: 1 if (row['crash_month'] > 2 and row['Crash Year'] == 2021) or (row['Crash Year'] > 2021) else 0, axis=1)
vb_incidents_per_month['P'] = vb_incidents_per_month['D'].cumsum()

# In[15]:


vb_incidents_per_month

# In[92]:


# Create feature for time since intervention - 1 (P_prime) to test for multicollinearity in the model
vb_incidents_per_month['P_prime'] = vb_incidents_per_month.apply(lambda x: x['P'] - 1 if x['D'] != 0 else 0, axis=1)

# In[16]:


# Fill missing values
vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'].fillna(0, inplace=True)
vb_incidents_per_month['percentage_pedestrian_crashes'].fillna(0, inplace=True)

# ## Modeling

# In[71]:


original = vb_incidents_per_month.copy()

# In[83]:


vb_incidents_per_month = original.copy()

# ### Non-Intersection Pedestrian-Vehicle Crashes

# In[161]:


# Run ARIMA ITSA on the data
print(test_stationarity(vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes']))

plot_acf(vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'])
plt.show()

auto_model_full = pm.auto_arima(vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'], 
                      X=vb_incidents_per_month[['T','D','P']],
                      start_p=0, start_q=0,
                      test='adf',       # Use ADF test to find optimal 'd'
                      max_p=20, max_q=20, # Maximum p and q
                      m=1,              # Frequency of the series
                      d=None,           # Let the model determine 'd'
                      seasonal=False,   # No seasonality
                      trace=True,       # Print status on the fits
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    # Stepwise selection to find the best model

print(auto_model_full.summary())

# Check residuals
residuals = auto_model_full.resid()
plt.figure()
plt.hist(residuals)
plt.title('Histogram of Residuals')
plt.show()
start = 62
end = len(vb_incidents_per_month)

arima_results = ARIMA(vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'], vb_incidents_per_month[['T','D','P']], order=auto_model_full.order).fit()

predictions = arima_results.get_prediction(0, end-1)

# Use auto_arima to find the best ARIMA model parameters before the intervention
auto_model_cf = pm.auto_arima(vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'][:start], exogenous=vb_incidents_per_month[["T"][:start]],
                        start_p=0, start_q=0,
                        test='adf',       # Using ADF test to find optimal 'd'
                        max_p=20, max_q=20, # Search max bounds for p and q
                        m=1,              # Non-seasonal data
                        d=None,           # Let auto_arima determine 'd'
                        seasonal=False,   # No seasonality
                        stepwise=True,    # Use the stepwise algorithm
                        trace=True,       # Print status on the fits
                        error_action='ignore',  
                        suppress_warnings=True)

arima_cf = ARIMA(vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'][:start], vb_incidents_per_month["T"][:start], order=auto_model_cf.order).fit()

# Model predictions means
y_pred = predictions.predicted_mean

# Counterfactual mean and 95% confidence interval
y_cf = arima_cf.get_forecast(steps=end-start, exog=vb_incidents_per_month["T"][start:end]).summary_frame(alpha=0.05)

# Plot section
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(16,10))

# Plot bounce rate data
ax.scatter(vb_incidents_per_month["T"], vb_incidents_per_month["percentage_pedestrian_nonintersection_crashes"], facecolors='none', edgecolors='steelblue', label="Traffic accident data", linewidths=2)

# Plot model predictions
# ax.plot(incidents_per_month["T"][:end], y_pred, 'b-', label="Model prediction")
ax.plot(vb_incidents_per_month["T"][:start], y_pred[:start], 'b-', label="Model prediction")
ax.plot(vb_incidents_per_month["T"][start:end], y_pred[start:], 'b-')

# Plot counterfactual predictions with 95% confidence interval
ax.plot(vb_incidents_per_month["T"][start:end], y_cf["mean"], 'k.', label="counterfactual")
ax.fill_between(vb_incidents_per_month["T"][start:end], y_cf['mean_ci_lower'], y_cf['mean_ci_upper'], color='k', alpha=0.1, label="Counterfactual 95% CI")

# Intervention line
ax.axvline(x=vb_incidents_per_month["T"][start], color='r', linestyle='--', label='Intervention')

# Labels and legends
ax.legend(loc='best')
plt.xlabel("Months")
plt.ylabel("Non-Intersection Pedestrian-involved traffic accidents (%)")
plt.show()

# The ACF plot shows no autocorrelation so we can use OLS instead of ARIMA

# In[101]:


# ITS Analysis with OLS and P variable

# Define the intervention point
start = 62

# Fit unrestricted OLS model
X_full = sm.add_constant(vb_incidents_per_month[['T', 'D', 'P']])
y_full = vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes']
model_full = sm.OLS(y_full, X_full).fit()

# Fit restricted OLS model (using data up to 'start' and only 'T')
X_restricted = sm.add_constant(vb_incidents_per_month.loc[:start, ['T']])
y_restricted = vb_incidents_per_month.loc[:start, 'percentage_pedestrian_nonintersection_crashes']
model_restricted = sm.OLS(y_restricted, X_restricted).fit()

# Making predictions from both models
vb_incidents_per_month['predictions_full'] = model_full.predict(X_full)

# Properly structure X for predictions from the restricted model over the full dataset
X_restricted_full = sm.add_constant(vb_incidents_per_month[['T']])  # including constant term
predictions_restricted = model_restricted.get_prediction(X_restricted_full)
summary_frame = predictions_restricted.summary_frame(alpha=0.05)

# Plotting the data and models
plt.figure(figsize=(12, 8))
plt.plot(vb_incidents_per_month['T'], vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'], label='Actual Data')
plt.plot(vb_incidents_per_month['T'], vb_incidents_per_month['predictions_full'], label='Unrestricted Model Predictions', linestyle='--')
plt.plot(vb_incidents_per_month['T'], summary_frame['mean'], label='Restricted Model Predictions', color='red')
plt.fill_between(vb_incidents_per_month['T'], summary_frame['mean_ci_lower'], summary_frame['mean_ci_upper'], color='red', alpha=0.3, label='95% CI for Restricted Model')
plt.axvline(x=vb_incidents_per_month['T'][start], color='green', linestyle='--', label='Intervention')
plt.xlabel('Time')
plt.ylabel('Percentage of Pedestrian Crashes')
plt.title('OLS Model Predictions')
plt.legend()
plt.show()

# In[103]:


# ITS Analysis with OLS and P_prime variable
# Define the intervention point
start = 62

# Fit unrestricted OLS model
X_full = sm.add_constant(vb_incidents_per_month[['T', 'D', 'P_prime']])
y_full = vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes']
model_full = sm.OLS(y_full, X_full).fit()

# Fit restricted OLS model (using data up to 'start' and only 'T')
X_restricted = sm.add_constant(vb_incidents_per_month.loc[:start, ['T']])
y_restricted = vb_incidents_per_month.loc[:start, 'percentage_pedestrian_nonintersection_crashes']
model_restricted = sm.OLS(y_restricted, X_restricted).fit()

# Making predictions from both models
vb_incidents_per_month['predictions_full'] = model_full.predict(X_full)

# Properly structure X for predictions from the restricted model over the full dataset
X_restricted_full = sm.add_constant(vb_incidents_per_month[['T']])  # including constant term
predictions_restricted = model_restricted.get_prediction(X_restricted_full)
summary_frame = predictions_restricted.summary_frame(alpha=0.05)

# Plotting the data and models
plt.figure(figsize=(12, 8))
plt.plot(vb_incidents_per_month['T'], vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'], label='Actual Data')
plt.plot(vb_incidents_per_month['T'], vb_incidents_per_month['predictions_full'], label='Unrestricted Model Predictions', linestyle='--')
plt.plot(vb_incidents_per_month['T'], summary_frame['mean'], label='Restricted Model Predictions', color='red')
plt.fill_between(vb_incidents_per_month['T'], summary_frame['mean_ci_lower'], summary_frame['mean_ci_upper'], color='red', alpha=0.3, label='95% CI for Restricted Model')
plt.axvline(x=vb_incidents_per_month['T'][start], color='green', linestyle='--', label='Intervention')
plt.xlabel('Time')
plt.ylabel('Percentage of Pedestrian Crashes')
plt.title('OLS Model Predictions')
plt.legend()
plt.show()

# In[104]:


model_full.summary()

# In[64]:


model_restricted.summary()

# In[65]:


X_full = sm.add_constant(vb_incidents_per_month[['D']])
y_full = vb_incidents_per_month['percentage_pedestrian_crashes']
model_full = sm.OLS(y_full, X_full).fit()

model_full.summary()

# In[70]:


X_full = sm.add_constant(vb_incidents_per_month[['T','D']])
y_full = vb_incidents_per_month['percentage_pedestrian_crashes']
model_full = sm.OLS(y_full, X_full).fit()

model_full.summary()

# In[67]:


X_full = sm.add_constant(vb_incidents_per_month[['T','P']])
y_full = vb_incidents_per_month['percentage_pedestrian_crashes']
model_full = sm.OLS(y_full, X_full).fit()

model_full.summary()

# In[68]:


X_full = sm.add_constant(vb_incidents_per_month[['P']])
y_full = vb_incidents_per_month['percentage_pedestrian_crashes']
model_full = sm.OLS(y_full, X_full).fit()

model_full.summary()

# ### All Pedestrian-Vehicle Crashes

# In[205]:


# Run ARIMA ITSA on the data

print(test_stationarity(vb_incidents_per_month['percentage_pedestrian_crashes']))

plot_acf(vb_incidents_per_month['percentage_pedestrian_crashes'])
plt.show()

auto_model_full = pm.auto_arima(vb_incidents_per_month['percentage_pedestrian_crashes'], 
                      X=vb_incidents_per_month[['T','D','P']],
                      start_p=0, start_q=0,
                      test='adf',       # Use ADF test to find optimal 'd'
                      max_p=20, max_q=20, # Maximum p and q
                      m=1,              # Frequency of the series
                      d=None,           # Let the model determine 'd'
                      seasonal=False,   # No seasonality
                      trace=True,       # Print status on the fits
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)    # Stepwise selection to find the best model

print(auto_model_full.summary())

# Check residuals
residuals = auto_model_full.resid()
plt.figure()
plt.hist(residuals)
plt.title('Histogram of Residuals')
plt.show()
start = 62
end = len(vb_incidents_per_month)

arima_results = ARIMA(vb_incidents_per_month['percentage_pedestrian_crashes'], vb_incidents_per_month[['T','D','P']], order=auto_model_full.order).fit()

predictions = arima_results.get_prediction(0, end-1)

# Use auto_arima to find the best ARIMA model parameters before the intervention
auto_model_cf = pm.auto_arima(vb_incidents_per_month['percentage_pedestrian_crashes'][:start], exogenous=vb_incidents_per_month[["T"][:start]],
                        start_p=0, start_q=0,
                        test='adf',       # Using ADF test to find optimal 'd'
                        max_p=20, max_q=20, # Search max bounds for p and q
                        m=1,              # Non-seasonal data
                        d=None,           # Let auto_arima determine 'd'
                        seasonal=False,   # No seasonality
                        stepwise=True,    # Use the stepwise algorithm
                        trace=True,       # Print status on the fits
                        error_action='ignore',  
                        suppress_warnings=True)

arima_cf = ARIMA(vb_incidents_per_month['percentage_pedestrian_crashes'][:start], vb_incidents_per_month["T"][:start], order=auto_model_full.order).fit()

# Model predictions means
y_pred = predictions.predicted_mean

# Counterfactual mean and 95% confidence interval
y_cf = arima_cf.get_forecast(steps=end-start, exog=vb_incidents_per_month["T"][start:end]).summary_frame(alpha=0.05)

# Plot section
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(16,10))

# Plot bounce rate data
ax.scatter(vb_incidents_per_month["T"], vb_incidents_per_month["percentage_pedestrian_crashes"], facecolors='none', edgecolors='steelblue', label="Traffic accident data", linewidths=2)

# Plot model predictions
# ax.plot(incidents_per_month["T"][:end], y_pred, 'b-', label="Model prediction")
ax.plot(vb_incidents_per_month["T"][:start], y_pred[:start], 'b-', label="Model prediction")
ax.plot(vb_incidents_per_month["T"][start:end], y_pred[start:], 'b-')

# Plot counterfactual predictions with 95% confidence interval
ax.plot(vb_incidents_per_month["T"][start:end], y_cf["mean"], 'k.', label="counterfactual")
ax.fill_between(vb_incidents_per_month["T"][start:end], y_cf['mean_ci_lower'], y_cf['mean_ci_upper'], color='k', alpha=0.1, label="Counterfactual 95% CI")

# Intervention line
ax.axvline(x=vb_incidents_per_month["T"][start], color='r', linestyle='--', label='Intervention')

# Labels and legends
ax.legend(loc='best')
plt.xlabel("Months")
plt.ylabel("Pedestrian-involved traffic accidents (%)")
plt.show()

# In[160]:


# ITS Analysis with OLS and P variable 

# Define the intervention point
start = 62

# Fit unrestricted OLS model
X_full = sm.add_constant(vb_incidents_per_month[['T', 'D', 'P']])
y_full = vb_incidents_per_month['percentage_pedestrian_crashes']
model_full = sm.OLS(y_full, X_full).fit()

# Fit restricted OLS model (using data up to 'start' and only 'T')
X_restricted = sm.add_constant(vb_incidents_per_month.loc[:start, ['T']])
y_restricted = vb_incidents_per_month.loc[:start, 'percentage_pedestrian_crashes']
model_restricted = sm.OLS(y_restricted, X_restricted).fit()

# Making predictions from both models
vb_incidents_per_month['predictions_full'] = model_full.predict(X_full)

# Properly structure X for predictions from the restricted model over the full dataset
X_restricted_full = sm.add_constant(vb_incidents_per_month[['T']])  # including constant term
predictions_restricted = model_restricted.get_prediction(X_restricted_full)
summary_frame = predictions_restricted.summary_frame(alpha=0.05)

# Plotting the data and models
plt.figure(figsize=(12, 8))
plt.scatter(vb_incidents_per_month['T'], vb_incidents_per_month['percentage_pedestrian_crashes'], label='Actual Data')
plt.plot(vb_incidents_per_month['T'], vb_incidents_per_month['predictions_full'], label='Unrestricted Model Predictions', linestyle='--')
plt.plot(vb_incidents_per_month['T'], summary_frame['mean'], label='Restricted Model Predictions', color='red')
# plt.fill_between(vb_incidents_per_month['T'], summary_frame['mean_ci_lower'], summary_frame['mean_ci_upper'], color='red', alpha=0.3, label='95% CI for Restricted Model')
plt.axvline(x=vb_incidents_per_month['T'][start], color='green', linestyle='--', label='Intervention')
plt.xlabel('Time')
plt.ylabel('Percentage of Pedestrian Crashes')
plt.title('OLS Model Predictions')
plt.legend()
plt.show()

# In[95]:


model_full.summary()

# In[258]:


# ITS Analysis with OLS and P_prime variable

# Define the intervention point
start = 62

# Fit unrestricted OLS model
X_full = sm.add_constant(vb_incidents_per_month[['T', 'D', 'P_prime']])
y_full = vb_incidents_per_month['percentage_pedestrian_crashes']
model_full = sm.OLS(y_full, X_full).fit()

# Fit restricted OLS model (using data up to 'start' and only 'T')
X_restricted = sm.add_constant(vb_incidents_per_month.loc[:start, ['T']])
y_restricted = vb_incidents_per_month.loc[:start, 'percentage_pedestrian_crashes']
model_restricted = sm.OLS(y_restricted, X_restricted).fit()

# Making predictions from both models
vb_incidents_per_month['predictions_full'] = model_full.predict(X_full)

# Properly structure X for predictions from the restricted model over the full dataset
X_restricted_full = sm.add_constant(vb_incidents_per_month[['T']])  # including constant term
predictions_restricted = model_restricted.get_prediction(X_restricted_full)
summary_frame = predictions_restricted.summary_frame(alpha=0.05)

# Plotting the data and models
plt.figure(figsize=(12, 8))
plt.plot(vb_incidents_per_month['T'], vb_incidents_per_month['percentage_pedestrian_crashes'], label='Actual Data')
plt.plot(vb_incidents_per_month['T'], vb_incidents_per_month['predictions_full'], label='Unrestricted Model Predictions', linestyle='--')
plt.plot(vb_incidents_per_month['T'], summary_frame['mean'], label='Restricted Model Predictions', color='red')
plt.fill_between(vb_incidents_per_month['T'], summary_frame['mean_ci_lower'], summary_frame['mean_ci_upper'], color='red', alpha=0.3, label='95% CI for Restricted Model')
plt.axvline(x=vb_incidents_per_month['T'][start], color='green', linestyle='--', label='Intervention')
plt.xlabel('Time')
plt.ylabel('Percentage of Pedestrian Crashes')
plt.title('OLS Model Predictions')
plt.legend()
plt.show()

# # Conclusion

# In[264]:


start = 62

print(f"Virginia Beach Average percentage_pedestrian_crashes before intervention: {vb_incidents_per_month['percentage_pedestrian_crashes'][:start].mean()}")
print(f"Virginia Beach Average percentage_pedestrian_crashes after intervention: {vb_incidents_per_month['percentage_pedestrian_crashes'][start:].mean()}")
print(f"Virginia Beach Average percentage_pedestrian_crashes before intervention: {vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'][:start].mean()}")
print(f"Virginia Beach Average percentage_pedestrian_crashes after intervention: {vb_incidents_per_month['percentage_pedestrian_nonintersection_crashes'][start:].mean()}")

# In[266]:


start = 120

print(f"Denver Average percentage_pedestrian_crashes before intervention: {denver_incidents_per_month['percentage_pedestrian_incidents'][:start].mean()}")
print(f"Denver Average percentage_pedestrian_crashes after intervention: {denver_incidents_per_month['percentage_pedestrian_incidents'][start:].mean()}")
print(f"Denver Average percentage_pedestrian_crashes before intervention: {denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'][:start].mean()}")
print(f"Denver Average percentage_pedestrian_crashes after intervention: {denver_incidents_per_month['percentage_pedestrian_nonintersection_incidents'][start:].mean()}")

# In[97]:


model_full.summary()

# In[48]:


model_restricted.summary()

# In[52]:


X_full = sm.add_constant(vb_incidents_per_month[['D']])
y_full = vb_incidents_per_month['percentage_pedestrian_crashes']
model_full = sm.OLS(y_full, X_full).fit()

model_full.summary()

# In[53]:


X_full = sm.add_constant(vb_incidents_per_month[['P']])
y_full = vb_incidents_per_month['percentage_pedestrian_crashes']
model_full = sm.OLS(y_full, X_full).fit()

model_full.summary()

# In[ ]:



