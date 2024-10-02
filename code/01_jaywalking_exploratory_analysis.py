#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from copy import deepcopy
import seaborn as sns
from shapely.ops import unary_union
from shapely.geometry import Point
import folium
from geopandas import GeoDataFrame
from shapely import wkt
from folium.plugins import FloatImage
import base64

# ### Cleaning Data

# In[3]:


# criminal summonses dataset from Open Data (adding historic to year-to-date)

# downloaded
criminal_summonses_historic=pd.read_csv('../data/input/NYPD_Criminal_Court_Summons__Historic__20240610.csv')
criminal_summonses_year_to_date=pd.read_csv('../data/input/NYPD_Criminal_Court_Summons_Incident_Level_Data__Year_To_Date__20240806.csv')

# from API
# criminal_summonses_historic=pd.read_csv('https://data.cityofnewyork.us/resource/sv2w-rv3k.csv?$select=summons_key,summons_date,offense_description,law_section_number,age_group,sex,race,boro,precinct_of_occur,geocoded_column&$limit=99999999')
# criminal_summonses_year_to_date=pd.read_csv('https://data.cityofnewyork.us/resource/mv4k-y93f.csv?$select=summons_key,summons_date,offense_description,law_section_number,age_group,sex,race,boro,precinct_of_occur,geocoded_column&$limit=99999999')

criminal_summonses_open_data=pd.concat([criminal_summonses_historic,criminal_summonses_year_to_date])

# In[4]:


# creating quarter column for Open Data dataset
criminal_summonses_open_data['SUMMONS_DATE'] = pd.to_datetime(criminal_summonses_open_data['SUMMONS_DATE'])
criminal_summonses_open_data['quarter'] = criminal_summonses_open_data['SUMMONS_DATE'].dt.quarter.astype(str) 
criminal_summonses_open_data['quarter'] = 'Q' + criminal_summonses_open_data['quarter']
criminal_summonses_open_data['year'] = criminal_summonses_open_data['SUMMONS_DATE'].dt.year.astype(str)
criminal_summonses_open_data['quarter'] = criminal_summonses_open_data['year'] + ' ' + criminal_summonses_open_data['quarter']

# In[5]:


# finding names used to refer jaywalking in "offense description" column
criminal_summonses_open_data = criminal_summonses_open_data.dropna(subset=['OFFENSE_DESCRIPTION'], inplace=False)
sorted(criminal_summonses_open_data[criminal_summonses_open_data['OFFENSE_DESCRIPTION'].str.contains('WALK')]['OFFENSE_DESCRIPTION'].unique())

# In[6]:


jaywalking_desc = [ 
'CROSS NOT WITHIN CROSSWALK',
 'CROSS OTHER THAN CROSSWALK',
 'CROSS ROAD @OTHER THAN CROSSWALK',
 'CROSS ROAD AT OTHER THAN CROSSWALK',
 'CROSS ROAD OTHER THAN CROSSWALK',
 'CROSS ROAD WAY DONT WALK SYNBOL',
 'CROSSING AT OTHER THAN CROSSWALK',
 'CROSSING NOT AT CROSSWALK',
 'CROSSING OTHER THEN CROSSWALK',
 'CROSSING STREET NOT AT CROSSWALK',
 'CROSSING STREET NOT IN CROSSWALK',
 'CROSSROAD OTHER THAN CROSSWALK',
 "DISOBEYDON'T WALK",
 "DON'T WALK HAND STEADY RED",
 'DONT WALK ENTER/ CROSS STREET',
 'FAILED TO USE SIDEWALK',
 'J - WALKING',
 'J WALK',
 'J WALKING',
 'J-WALKING',
 'J. WALKING',
 'JAT WALKING',
 'JAY WALKING',
 'JAY WALKING`',
 'JAYWALK',
 'JAYWALK (CROSS TRAFFIC)',
 'JAYWALKING',
 'JAYWALKING - NOT IN CROSSWALK',
 'JAYWALKING AGAINST TRAFFIC',
 'JAYWALKING CROSS ROADWAY DIAGONALLY',
 'PEDESTRIAN FAIL TO USE SIDEWALK',
 'PEDESTRIAN STARTED ON STEADY DONT WALK',
 'PEDESTRIAN WALKING',
 'PEDESTRIAN WALKING INTO TRAFFIC UNSAFELY',
 "WALK AGAINST STEADY DON'T WALK",
 'WALK INTO PATH OF MARKD RMP',
 'WALK INTO PATH OF ONCOMING',
 'WALK INTO PATH OF ONCOMING VEH.',
 'WALK INTO PATH OF ONCOMING VEHICLE',
 'WALK INTO PATH OF VEH',
 'WALK INTO PATH ONCOMING AUTO',
 'WALK INTO TRAFFIC',
 'WALK ON ROADWAY',
 'WALK PATH OF ONCOMING AUTO',
 'WALKING IN PATH OF AUTO',
 'WALKING IN STREET',
 'WALKING IN STREET /SIDEWALK AVAILABLE',
 'WALKING IN STREET SIDEWALK',
 'WALKING IN STREET W/SIDEWALK AVALIBLE',
 'WALKING IN STREET WHEN SIDEWALK AVAIL.',
 'WALKING IN STREET WHEN SIDEWALK AVAILABL',
 'WALKING IN STREET WHEN SIDEWALK IS AVAIL',
 'WALKING IN STREET WHEN SIDEWALKS AVAILAB',
 'WALKING IN THE STREET SIDEWALK UNAVAILAB',
 'WALKING IN THE STREET WHEN SIDEWALK AVAI',
 'WALKING IN THE STREET WHEN SIDEWALK IS A',
 'WALKING INTO PATH OF ONCOMING TRAFFIC',
 'WALKING ON STATE HIGHWAY',
 'WALKING ON STREET / SIDEWALK AVAILABLE',
 'WALKING ON STREET WHEN SIDEWALK AVAIL',
 'WALKING ON STREET WHEN SIDEWALK AVAIL.',
 'WALKING ON STREET WHEN SIDEWALK AVAILABL',
 'WALKING ON STREET WHEN SIDEWALK AVAL',
 'WALKING ON STREET WHEN SIDEWALK IS AVAIL',
 'WALKING ON STREET WITH SIDEWALK AVAIL.',
 'WALKING ON STREET/SIDEWALK AVAILABLE',
 'WALKING ON THE STREET WHEN SIDEWALK IS A',
 'WLKING IN ST/SIDEWALK AVAIL'
 ]

# In[7]:


# law_section_number penal codes for jaywalking

# jaywalking_codes =  ['4-04', '4-04(B)(2)', '4-04(C)(3)']

# In[8]:


# subsetting dataset

# jaywalking_crim_summonses = criminal_summonses_open_data[criminal_summonses_open_data['LAW_SECTION_NUMBER'].isin(jaywalking_codes)] # using codes
jaywalking_crim_summonses = criminal_summonses_open_data[criminal_summonses_open_data['OFFENSE_DESCRIPTION'].isin(jaywalking_desc)] # using descriptions
jaywalking_crim_summonses_quarterly = jaywalking_crim_summonses.groupby('quarter').count().rename(columns={'SUMMONS_DATE':'total_summonses'})[['total_summonses']]

# jaywalking_crim_summonses.to_csv('../data/output/jaywalking_crim_summonses.csv')

# In[65]:


# summonses by borough

round((100*jaywalking_crim_summonses['BORO'].value_counts() / len(jaywalking_crim_summonses)),2)

# ### Charts

# In[9]:


# yearly numbers

yearly = jaywalking_crim_summonses.drop(columns=['SUMMONS_DATE']).groupby(jaywalking_crim_summonses['SUMMONS_DATE'].dt.year).count()[['SUMMONS_KEY']].rename(columns={'SUMMONS_KEY':'Total Jaywalking Summonses'}).rename_axis('Year').loc[2013:2023]
yearly

# yearly.to_csv('../data/output/yearly_jaywalking-summonses.csv')

# In[10]:


# 4-year average

yearly.loc[2019:2023]['Total Jaywalking Summonses'].mean()
#yearly.loc[2019:2023]['Total Jaywalking Summonses'].median()

# In[11]:


# visualizing offenses over time

x_crim = jaywalking_crim_summonses_quarterly.index
y_crim = jaywalking_crim_summonses_quarterly['total_summonses']

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()

plt.plot(x_crim, y_crim)

plt.xlabel("$\bf{y=e^{x}}$")
spacing = 0.100
fig.subplots_adjust(bottom=spacing)

ax.xaxis.set_major_locator(plt.MaxNLocator(20))

plt.xlabel('Quarter')
plt.xticks(rotation=45)
plt.ylabel('Summonses')
plt.title('NYPD Criminal Summonses for Jaywalking Offenses')

plt.grid(True)
plt.style.use("ggplot")

%matplotlib inline
plt.show()

# jaywalking_crim_summonses_quarterly.to_csv('../data/output/jaywalking_crim_summonses_quarterly.csv')

# In[328]:


# investigate the spike
# happened in 2014 Q1
# there was a spike in precinct 44

jaywalking_2014Q1 = jaywalking_crim_summonses[jaywalking_crim_summonses['quarter'] == '2014 Q1']

round((100*jaywalking_2014Q1['PRECINCT_OF_OCCUR'].value_counts() / len(jaywalking_2014Q1)),2).sort_values(ascending=False).head(10)

# In[12]:


# racial breakdown of jaywalking summonses (2017-2023)
# including 'Unknown'

# jaywalking summonses (2017-2023)

jaywalking_crim_summonses_2017 = jaywalking_crim_summonses[jaywalking_crim_summonses['quarter'] >= '2017 Q1']

jaywalking_crim_summonses_2017['RACE'].replace(['BLACK HISPANIC','WHITE HISPANIC'], 'HISPANIC', inplace=True) # collapsing hispanic categories
jaywalking_crim_summonses_2017['RACE'].replace('AMERICAN INDIAN/ALASKAN NATIVE', 'OTHER', inplace=True) 
jaywalking_crim_summonses_2017['RACE'].replace('(null)', 'UNKNOWN', inplace=True) 

racial_breakdown_jaywalking_2017 = pd.DataFrame(round((100*jaywalking_crim_summonses_2017['RACE'].value_counts() / jaywalking_crim_summonses_2017['RACE'].count()),2)).rename(columns={'count':'jaywalking_summons'})

racial_breakdown_jaywalking_2017

# In[13]:


# white vs non-white 2017-2023
# excluding 'Unknown'
# can't really look farther back in time than this... the race was listed as UNKNOWN for the vast majority entries (usually almost 100%) until 2016 Q3

jaywalking_white_v_nonwhite = jaywalking_crim_summonses_2017[jaywalking_crim_summonses_2017['RACE'] != 'UNKNOWN'].reset_index().drop(columns=['index'])
jaywalking_white_v_nonwhite['RACE'].replace(['BLACK', 'HISPANIC', 'ASIAN / PACIFIC ISLANDER','OTHER'], 'Non-White', inplace=True) # collapsing hispanic categories
jaywalking_white_v_nonwhite['RACE'].replace('WHITE', 'White', inplace=True) # collapsing hispanic categories

# replacing missing values
# new_row = {'RACE': 'White','quarter': '2020 Q1'}
# jaywalking_white_v_nonwhite.loc[len(jaywalking_white_v_nonwhite)+1] = new_row

jaywalking_white_v_nonwhite_groupby = jaywalking_white_v_nonwhite.groupby(['RACE', 'quarter']).count()[['SUMMONS_KEY']].rename(columns={'SUMMONS_KEY':'Count'}).reset_index()
jaywalking_white_v_nonwhite_groupby['quarter'] = jaywalking_white_v_nonwhite_groupby['quarter'].sort_values()

count_dict = pd.Series(jaywalking_white_v_nonwhite_groupby.groupby('quarter').sum()['Count'].values,index=jaywalking_white_v_nonwhite_groupby.groupby('quarter').sum()['Count'].index).to_dict()
jaywalking_white_v_nonwhite_groupby['Total Summonses This Quarter'] = jaywalking_white_v_nonwhite_groupby['quarter'].str.findall('|'.join([fr'\b{w}\b' for w in count_dict.keys()])).apply(", ".join).map(count_dict)
jaywalking_white_v_nonwhite_groupby['Percent of Total'] = round((100*jaywalking_white_v_nonwhite_groupby['Count'] / jaywalking_white_v_nonwhite_groupby['Total Summonses This Quarter']),2)

fig, ax = plt.subplots()
ax = sns.lineplot(x="quarter", y="Percent of Total", hue="RACE", data=jaywalking_white_v_nonwhite_groupby)

ax.xaxis.set_major_locator(plt.MaxNLocator(20))

plt.xlabel("$\bf{y=e^{x}}$")
spacing = 0.100
fig.subplots_adjust(bottom=spacing)

plt.xlabel('Quarter')
plt.xticks(rotation=45)
plt.ylabel('Percent (%)')
plt.title('Racial Breakdown of Jaywalking Criminal Summonses')
plt.legend(fontsize='10')

plt.grid(True)
plt.style.use("ggplot")

plt.show()

# jaywalking_white_v_nonwhite_groupby.to_csv('../data/output/jaywalking_white_v_nonwhite.csv')

# In[14]:


# what % of all summonses are jaywalking, by race

criminal_summonses_2017 = criminal_summonses_open_data[criminal_summonses_open_data['SUMMONS_DATE'] > '2017-01-01'] # filtering to date range that both datasets have in common

jaywalking_share_by_race = deepcopy(criminal_summonses_open_data)

jaywalking_share_by_race['RACE'].replace(['BLACK HISPANIC','WHITE HISPANIC'], 'HISPANIC', inplace=True) # collapsing hispanic categories
jaywalking_share_by_race['RACE'].replace('AMERICAN INDIAN/ALASKAN NATIVE', 'OTHER', inplace=True) 
jaywalking_share_by_race['RACE'].replace('(null)', 'UNKNOWN', inplace=True) 

jaywalking_share_by_race['Jaywalking Offense?'] = np.where(jaywalking_share_by_race['OFFENSE_DESCRIPTION'].isin(jaywalking_desc), 1, 0)
jaywalking_share_by_race = jaywalking_share_by_race.drop(columns=['SUMMONS_DATE']).groupby('RACE').agg({'OFFENSE_DESCRIPTION':'count','Jaywalking Offense?':'sum'}).rename(columns={'OFFENSE_DESCRIPTION':'total_summonses','Jaywalking Offense?':'jaywalking_summonses'})
jaywalking_share_by_race['% of Total Summonses'] = round((100*jaywalking_share_by_race['jaywalking_summonses'] / jaywalking_share_by_race['total_summonses']),2)

# In[15]:


# sort the DataFrame by the percentage of total summonses that are for jaywalking
jaywalking_share_by_race = jaywalking_share_by_race.sort_values(by='% of Total Summonses', ascending=False)

# create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=jaywalking_share_by_race.reset_index(), x='RACE', y='% of Total Summonses', palette='viridis')

# add titles and labels
plt.title('Jaywalking Share of All Criminal Summonses by Race')
plt.xlabel('Race')
plt.ylabel('% of Total Summonses')

# display the plot
plt.xticks(rotation=45)
plt.show()

# jaywalking_share_by_race.to_csv('../data/output/jaywalking_share_by_race.csv')

# ### Maps

# In[17]:


# uploading precinct data

precinct_data = pd.read_csv('../data/input/precinct-geographies_2021.csv').set_index('policeprct')

jaywalking_summonses_2023 = jaywalking_crim_summonses_2017[(jaywalking_crim_summonses_2017['SUMMONS_DATE'] >= '2023-01-01') & (jaywalking_crim_summonses_2017['SUMMONS_DATE'] < '2024-01-01')] # just for 2023
jaywalking_summonses_2023_by_precinct = jaywalking_summonses_2023.groupby('PRECINCT_OF_OCCUR').count().rename(columns={'SUMMONS_KEY':'Jaywalking Summonses'})[['Jaywalking Summonses']]
jaywalking_summonses_2023_by_precinct['Total population'] = precinct_data['Total population']
jaywalking_summonses_2023_by_precinct['Jaywalking Summonses per 10K'] = round((10000*jaywalking_summonses_2023_by_precinct['Jaywalking Summonses'] / jaywalking_summonses_2023_by_precinct['Total population']),1)
jaywalking_summonses_2023_by_precinct['geometry'] = precinct_data['geometry']

jaywalking_summonses_2023_by_precinct['geometry'] = jaywalking_summonses_2023_by_precinct['geometry'].apply(wkt.loads)
jaywalking_summonses_2023_by_precinct = GeoDataFrame(jaywalking_summonses_2023_by_precinct, crs="EPSG:4326", geometry='geometry')
jaywalking_summonses_2023_by_precinct['PRECINCT_OF_OCCUR'] = jaywalking_summonses_2023_by_precinct.index


# In[18]:


# choropleth map of regions with greatest rate of jaywalking criminal summonses

# function that encodes an image file with base64

def b64_image(image_filename):

# image_filename: path to file
    
    with open(image_filename, 'rb') as f:
        image = f.read()
        
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

# map specifications
zoom = 9.8
lat = 40.706000
lon = -73.976300

# Data Team Logo, to be added to map
data_team_logo = '../assets/data-team-logo.png' 

# starting map

fig = folium.Map(location=[lat, lon], zoom_start=zoom, tiles='cartodbpositron') # blank map

# setting column values and pop-up aliases for all variables with MOEs and CVs

columns = ['PRECINCT_OF_OCCUR','Jaywalking Summonses per 10K'] # uses 'for map' columns so only regions w/ reliable values (based on CV) are colored in based on gen_reliable_col() output df
aliases= ['Precinct','Summonses per 10K']

choropleth = folium.Choropleth(
    geo_data = jaywalking_summonses_2023_by_precinct,
    name = 'choropleth',
    data = jaywalking_summonses_2023_by_precinct,
    columns = columns,
    key_on = 'feature.properties.PRECINCT_OF_OCCUR',
    fill_color = 'Blues',
    fill_opacity = 0.7,
    line_opacity = 0.2,
    nan_fill_color='grey',
    use_jenks = True, # check with Rose
    legend_name = 'Jaywalking Criminal Summonses per 10,000 people (2023)',
    highlight = True,
).add_to(fig)

choropleth.color_scale.width=500 # setting legend width

choropleth.geojson.add_child( 
    folium.features.GeoJsonTooltip(columns, aliases=aliases, labels=True))

# adding Data Team logo

# FloatImage(b64_image(data_team_logo), bottom=1, left=1).add_to(fig)
      
fig

# In[19]:


# creating a column for jaywalking % of total summonses

criminal_summonses_2023 = criminal_summonses_open_data[(criminal_summonses_open_data['SUMMONS_DATE'] >= '2023-01-01') & (criminal_summonses_open_data['SUMMONS_DATE'] < '2024-01-01')]
criminal_summonses_2023_by_precinct = criminal_summonses_2023.groupby('PRECINCT_OF_OCCUR').count().rename(columns={'SUMMONS_KEY':'All Summonses'})[['All Summonses']]

jaywalking_summonses_2023_by_precinct['All Summonses'] = criminal_summonses_2023_by_precinct['All Summonses']
jaywalking_summonses_2023_by_precinct['Jaywalking % of Total Summonses'] = round((100*jaywalking_summonses_2023_by_precinct['Jaywalking Summonses'] / jaywalking_summonses_2023_by_precinct['All Summonses']),2)

# In[20]:


# choropleth map showing areas with greatest jaywalking share of total summonses

# function that encodes an image file with base64

def b64_image(image_filename):

# image_filename: path to file
    
    with open(image_filename, 'rb') as f:
        image = f.read()
        
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

# map specifications
zoom = 9.8
lat = 40.706000
lon = -73.976300

# Data Team Logo, to be added to map
data_team_logo = '../assets/data-team-logo.png' 

# starting map

fig = folium.Map(location=[lat, lon], zoom_start=zoom, tiles='cartodbpositron') # blank map

# setting column values and pop-up aliases for all variables with MOEs and CVs

columns = ['PRECINCT_OF_OCCUR','Jaywalking % of Total Summonses'] # uses 'for map' columns so only regions w/ reliable values (based on CV) are colored in based on gen_reliable_col() output df
aliases= ['Precinct','% of Total']

choropleth = folium.Choropleth(
    geo_data = jaywalking_summonses_2023_by_precinct,
    name = 'choropleth',
    data = jaywalking_summonses_2023_by_precinct,
    columns = columns,
    key_on = 'feature.properties.PRECINCT_OF_OCCUR',
    fill_color = 'Blues',
    fill_opacity = 0.7,
    line_opacity = 0.2,
    nan_fill_color='grey',
    use_jenks = True, # check with Rose
    legend_name = 'Jaywalking Share of Total Summonses (%)',
    highlight = True,
).add_to(fig)

choropleth.color_scale.width=500 # setting legend width

choropleth.geojson.add_child( 
    folium.features.GeoJsonTooltip(columns, aliases=aliases, labels=True))

# adding Data Team logo

# FloatImage(b64_image(data_team_logo), bottom=1, left=1).add_to(fig)
      
fig

# jaywalking_summonses_2023_by_precinct.drop(columns=['geometry']).to_csv('../data/output/jaywalking_summonses_2023_by_precinct.csv')

# In[21]:


# preparing to look at areas with worst racial disparity

prct_racial_breakdown = precinct_data[[
    
           '% Hispanic or Latino','% American Indian or Alaska Native alone, not Hispanic or Latino',
           '% Asian alone, not Hispanic or Latino','% Black or African American alone, not Hispanic or Latino',
           '% White alone, not Hispanic or Latino', '% Some other race alone, not Hispanic or Latino',
           '% Two or more races, not Hispanic or Latino'
    
        ]]

# matching categories used in the criminal summonses dataset

prct_racial_breakdown['Asian / Pacific Islander'] = precinct_data['% Asian alone, not Hispanic or Latino'] + precinct_data['% Native Hawaiian or other Pacific Islander alone, not Hispanic or Latino']
prct_racial_breakdown['Other'] = prct_racial_breakdown['% American Indian or Alaska Native alone, not Hispanic or Latino'] + prct_racial_breakdown['% Some other race alone, not Hispanic or Latino'] + prct_racial_breakdown['% Two or more races, not Hispanic or Latino']

prct_racial_breakdown = prct_racial_breakdown.rename(columns={'% Hispanic or Latino': 'Hispanic',
                                                              '% Black or African American alone, not Hispanic or Latino': 'Black',
                                                              '% White alone, not Hispanic or Latino': 'White'
                                                         }).drop(columns=['% American Indian or Alaska Native alone, not Hispanic or Latino',
                                                                          '% Asian alone, not Hispanic or Latino',
                                                                          '% Some other race alone, not Hispanic or Latino',
                                                                          '% Two or more races, not Hispanic or Latino'])


prct_racial_breakdown['Non-White'] = prct_racial_breakdown['Hispanic'] + prct_racial_breakdown['Black'] + prct_racial_breakdown['Asian / Pacific Islander'] + prct_racial_breakdown['Other']

white_nonwhite_by_prct = prct_racial_breakdown[['White','Non-White']].rename(columns={'Non-White':'Non-White Precinct','White':'White Precinct'})

# In[22]:


# racial breakdown for each precinct

# jaywalking_white_v_nonwhite_2023 = jaywalking_white_v_nonwhite[(jaywalking_white_v_nonwhite['SUMMONS_DATE'] >= '2023-01-01') & (jaywalking_white_v_nonwhite['SUMMONS_DATE'] < '2024-01-01')]
jaywalking_white_v_nonwhite_2017 = jaywalking_white_v_nonwhite[(jaywalking_white_v_nonwhite['SUMMONS_DATE'] >= '2017-01-01')]
jaywalking_precinct_total_summonses = jaywalking_white_v_nonwhite_2017.groupby('PRECINCT_OF_OCCUR').count().rename(columns={'SUMMONS_KEY':'Total Jaywalking Summonses'})[['Total Jaywalking Summonses']].reset_index()
jaywalking_white_v_nonwhite_2017_groupby = jaywalking_white_v_nonwhite_2017.groupby(['PRECINCT_OF_OCCUR','RACE']).count().rename(columns={'SUMMONS_KEY':'Count'})[['Count']]
jaywalking_white_v_nonwhite_2017_groupby = jaywalking_white_v_nonwhite_2017_groupby.reset_index().merge(jaywalking_precinct_total_summonses, left_on='PRECINCT_OF_OCCUR', right_on='PRECINCT_OF_OCCUR')
jaywalking_white_v_nonwhite_2017_groupby['% of Total'] = round((100*jaywalking_white_v_nonwhite_2017_groupby['Count'] / jaywalking_white_v_nonwhite_2017_groupby['Total Jaywalking Summonses']),1)
jaywalking_white_v_nonwhite_2017_groupby['PRECINCT_OF_OCCUR'] = jaywalking_white_v_nonwhite_2017_groupby['PRECINCT_OF_OCCUR'].astype('int')
jaywalking_white_v_nonwhite_2017_groupby = jaywalking_white_v_nonwhite_2017_groupby.set_index(['PRECINCT_OF_OCCUR','RACE'])

# In[23]:


# creating table comparing jaywalking race breakdown to precinct demographic breakdown

jaywalking_white_v_nonwhite_2017_prct = jaywalking_white_v_nonwhite_2017_groupby.pivot_table(
                                               values='% of Total', 
                                               index='PRECINCT_OF_OCCUR', 
                                               columns='RACE'
                                              ).rename(columns={'Non-White':'Non-White Jaywalking','White':'White Jaywalking'})

jaywalking_count = pd.DataFrame(jaywalking_white_v_nonwhite_2017_groupby.reset_index()[['PRECINCT_OF_OCCUR', 'Total Jaywalking Summonses']].groupby('PRECINCT_OF_OCCUR').mean()) # adding column that has total count of CJRA summonses
merged_white_nonwhite_2017 = jaywalking_white_v_nonwhite_2017_prct.reset_index().merge(white_nonwhite_by_prct.reset_index(), left_on='PRECINCT_OF_OCCUR', right_on='policeprct')
merged_white_nonwhite_2017 = merged_white_nonwhite_2017.drop(columns=['PRECINCT_OF_OCCUR']).set_index('policeprct')
merged_white_nonwhite_2017 = merged_white_nonwhite_2017.reset_index().merge(jaywalking_count.reset_index(), left_on='policeprct', right_on='PRECINCT_OF_OCCUR').drop(columns={'PRECINCT_OF_OCCUR'}).rename(columns={'Count':'CJRA Summonses Count'}).set_index('policeprct')
merged_white_nonwhite_2017 = merged_white_nonwhite_2017.fillna(0)

merged_white_nonwhite_2017['Non-White Difference'] = round((merged_white_nonwhite_2017['Non-White Jaywalking'] - merged_white_nonwhite_2017['Non-White Precinct']),1)
merged_white_nonwhite_2017['Absolute % Difference'] = abs(merged_white_nonwhite_2017['Non-White Difference'])
merged_white_nonwhite_2017['Non-White Jaywalking'] = round(merged_white_nonwhite_2017['Non-White Jaywalking'],1)
merged_white_nonwhite_2017['Non-White Precinct'] = round(merged_white_nonwhite_2017['Non-White Precinct'],1)
merged_white_nonwhite_2017['Disproportionately'] = np.where(merged_white_nonwhite_2017['Non-White Difference'] > 0, 'Non-White', 'White')

# In[24]:


# preparing for map 

merged_white_nonwhite_2017['geometry'] = precinct_data['geometry']
merged_white_nonwhite_2017=merged_white_nonwhite_2017.reset_index()

merged_white_nonwhite_2017['geometry'] = merged_white_nonwhite_2017['geometry'].apply(wkt.loads)
merged_white_nonwhite_2017 = GeoDataFrame(merged_white_nonwhite_2017, crs="EPSG:4326", geometry='geometry')

# In[25]:


# finding areas with biggest racial disparities, then sorting those by total number of jay-walking summonses 
# trying to avoid identifying regions that only have high disparities because there only only a few jaywalking summonses
# most jaywalking summonses racial makeups are disproportionately Non-White

merged_white_nonwhite_2017.sort_values('Absolute % Difference', ascending=False)[['policeprct', 'Non-White Jaywalking', 'Non-White Precinct', 'Total Jaywalking Summonses', 'Absolute % Difference', 'Disproportionately']].head(10).sort_values('Total Jaywalking Summonses', ascending=False)

# In[28]:


# finding areas with greatest number of jaywalking summonses, then sorting those by racial disparity 
# most still have a disparity that disproportionately impacts non-white people

high_summones_high_disparities = merged_white_nonwhite_2017.sort_values('Total Jaywalking Summonses', ascending=False)[['policeprct', 'Non-White Jaywalking', 'Non-White Precinct', 'Total Jaywalking Summonses', 'Absolute % Difference', 'Disproportionately']].head(10).sort_values('Absolute % Difference', ascending=False)
high_summones_high_disparities 

# high_summones_high_disparities.to_csv('../data/output/racial-disparities_precincts-w-top-jaywalking-summonses.csv')

# In[29]:


# just looking at top half 

top_half_jaywalking_summonses = merged_white_nonwhite_2017.sort_values('Total Jaywalking Summonses', ascending=False)[['policeprct', 'Non-White Jaywalking', 'Non-White Precinct', 'Total Jaywalking Summonses', 'Absolute % Difference', 'Disproportionately']].head(37)

print('Average disparity for white targeted areas:', top_half_jaywalking_summonses[top_half_jaywalking_summonses['Disproportionately'] == 'White']['Absolute % Difference'].median(), '%')
print('Average disparity for non-white targeted areas:', top_half_jaywalking_summonses[top_half_jaywalking_summonses['Disproportionately'] == 'Non-White']['Absolute % Difference'].median(), '%')

# In[40]:


top_half_disparities = merged_white_nonwhite_2017.sort_values('Absolute % Difference', ascending=False).head(37)
print('Average # of jaywalking summonses for precincts with highest racial disparities targeting white people:', top_half_disparities[top_half_disparities['Disproportionately'] == 'White']['Total Jaywalking Summonses'].median(), 'summonses')
print('Average # of jaywalking summonses for precincts with highest racial disparities targeting non-white people:', top_half_disparities[top_half_disparities['Disproportionately'] == 'Non-White']['Total Jaywalking Summonses'].median(), 'summonses')


# In[31]:


# map showing racial disparities in jaywalking summonses relative to population (2017-2023)

fig = folium.Map(location=[lat, lon], zoom_start=zoom, tiles='cartodbpositron')

# define columns and aliases
columns = ['policeprct', 'Disproportionately', 'Absolute % Difference', 'Non-White Jaywalking', 'Non-White Precinct', 'Total Jaywalking Summonses']
aliases = ['Precinct', 'Disproportionately', 'Difference (%)', 'Non-White Share of Jaywalking Summonses (%)', 'Non-White Share of Population (%)', 'Total Jaywalking Summonses']

# define customized diverging bins: [min (-) value, 50th percentile (-) value, 0, 50th percentile (+) value, max (+) value]

negative_df = pd.DataFrame(merged_white_nonwhite_2017[merged_white_nonwhite_2017['Non-White Difference'] < 0]['Non-White Difference'].describe())
positive_df = pd.DataFrame(merged_white_nonwhite_2017[merged_white_nonwhite_2017['Non-White Difference'] > 0]['Non-White Difference'].describe())

divering_palette = [negative_df['Non-White Difference'].loc['min'], negative_df['Non-White Difference'].loc['25%'],
                    negative_df['Non-White Difference'].loc['50%'], negative_df['Non-White Difference'].loc['75%'], 0, 
                    positive_df['Non-White Difference'].loc['25%'], positive_df['Non-White Difference'].loc['50%'], 
                    positive_df['Non-White Difference'].loc['75%'], positive_df['Non-White Difference'].loc['max']] 

# create a Choropleth map
choropleth = folium.Choropleth(
    geo_data=merged_white_nonwhite_2017,
    name='choropleth',
    data=merged_white_nonwhite_2017,
    columns=['policeprct', 'Non-White Difference'],
    key_on='feature.properties.policeprct',
    fill_color='RdYlBu',  # Set a temporary colormap here
    fill_opacity=0.7,
    line_opacity=0.2,
    nan_fill_color='grey',
    bins=divering_palette,
    highlight=True,
    legend_name='Non-White Share of Jaywalking Summonses Subtracted by Non-White Share of Population (%) (2017-2023)',
).add_to(fig)

# add the tooltip
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(fields=columns, aliases=aliases, labels=True)
)

choropleth.color_scale.width=550 # setting legend width

# add Data Team logo
# data_team_logo = '../assets/data-team-logo.png'  
# FloatImage(b64_image(data_team_logo), bottom=1, left=1).add_to(fig)

# display the map
fig

# merged_white_nonwhite_2017.to_csv('../data/output/jaywalking_merged_white_nonwhite_2017.csv')
# fig.save('../visuals/jaywalking-racial-disparity_by-precinct_map_2017-2023.html')

# ### Priority Zones

# In[43]:


# taking a look at vision zero priority zones (high number of pedestrian KSI)

priority_zones = pd.read_csv('https://data.cityofnewyork.us/resource/6pav-h8qv.csv') #uploading

# create GeoDataFrame
priority_zones['the_geom'] = priority_zones['the_geom'].apply(wkt.loads)
priority_zones = gpd.GeoDataFrame(priority_zones, crs="EPSG:4326", geometry='the_geom')

# creating one polygon for all priority zones
combined_geometry = unary_union(priority_zones.the_geom) 

# In[121]:


# uploading crash data

collision_dataset = gpd.read_file('../data/input/vz_nyc_vehicle-collisions_2263.geojson').drop(columns=['Unnamed: 0']) # downloaded version

# In[122]:


# create GeoDataFrame

collision_dataset_gdf = collision_dataset.to_crs({'init': 'epsg:4326'}) 

# In[264]:


# uploading demographic data at the point level

demographics = pd.read_csv('../data/input/raw_bbl_estimates_2021.csv').drop(columns=['Unnamed: 0'])

# In[265]:


# creating column indicating whether or not summons was given in a priority corridor
jaywalking_crim_summonses_2017['point_geom'] = [Point(xy) for xy in zip(jaywalking_crim_summonses_2017['Longitude'], jaywalking_crim_summonses_2017['Latitude'])]
jaywalking_crim_summonses_2017['priority_corridor'] = jaywalking_crim_summonses_2017['point_geom'].apply(
    lambda point: 1 if combined_geometry.contains(point) else 0
)

# now for collisions (using different method because above code is too slow for a dataset this size)
collision_dataset_2017 = collision_dataset_gdf[collision_dataset_gdf['crash_date'] >= pd.to_datetime('01-01-2017')]
collision_dataset_2017 = gpd.sjoin(collision_dataset_2017, priority_zones[['the_geom', 'sq_mi']], how='left').rename(columns={'sq_mi':'priority_corridor'}).drop(columns=['index_right'])
collision_dataset_2017['priority_corridor'] = np.where(collision_dataset_2017['priority_corridor'].isnull(), 0, 1)

# now for demographics
demographics['point_geom'] = [Point(xy) for xy in zip(demographics['longitude'], demographics['latitude'])]
demographics = GeoDataFrame(demographics, crs="EPSG:4326", geometry='point_geom')
demographics = gpd.sjoin(demographics, priority_zones[['the_geom', 'sq_mi']], how='left').rename(columns={'sq_mi':'priority_corridor'})#.drop(columns=['index_right'])
demographics['priority_corridor'] = np.where(demographics['priority_corridor'].isnull(), 0, 1)

# In[266]:


# creating dataset locating all of the pedestrian-involved car accidents

collision_dataset_2017['pedestrian_ksi'] = collision_dataset_2017['number_of_pedestrians_injured'] + collision_dataset_2017['number_of_pedestrians_killed']	
pedestrian_collisions_2017 = collision_dataset_2017[collision_dataset_2017['pedestrian_ksi'] != 0]


# In[267]:


# looking at racial breakdown 

non_white_variables = ['pop_est_DP05_0071PE','pop_est_DP05_0079PE', 'pop_est_DP05_0080PE', 'pop_est_DP05_0078PE',
                       'pop_est_DP05_0081PE', 'pop_est_DP05_0082PE', 'pop_est_DP05_0083PE']

white_variables = ['pop_est_DP05_0077PE']

demographics_white_nonwhite = demographics[non_white_variables + white_variables + ['pop_estimate_pluto', 'priority_corridor']].groupby('priority_corridor').sum()

demographics_white_nonwhite['pop_non_white'] = demographics_white_nonwhite[non_white_variables].sum(axis=1)
demographics_white_nonwhite['pop_white'] = demographics_white_nonwhite[white_variables].sum(axis=1)
demographics_white_nonwhite['frac_non_white'] = round((100*demographics_white_nonwhite['pop_non_white'] / demographics_white_nonwhite['pop_estimate_pluto']),2)
demographics_white_nonwhite['frac_white'] = round((100*demographics_white_nonwhite['pop_white'] / demographics_white_nonwhite['pop_estimate_pluto']),2)

# In[268]:


print('Non-white proportion of jaywalking summonses recipients:', round((100*len(jaywalking_white_v_nonwhite[jaywalking_white_v_nonwhite['RACE'] == 'Non-White']) / len(jaywalking_white_v_nonwhite)),1), '%')

# In[110]:


# comparing

print('Percent of total population living in priority corridors:', round((100*demographics_white_nonwhite['pop_estimate_pluto'].loc[1] / demographics_white_nonwhite['pop_estimate_pluto'].sum()),1), '%')
print('Percent of pedestrian KSI occuring in priority corridors:', round((100*len(pedestrian_collisions_2017[pedestrian_collisions_2017['priority_corridor'] == 1]) / len(pedestrian_collisions_2017)),1), '%')
print('Percent of jaywalking summonses occuring in priority corridors:', round((100*len(jaywalking_crim_summonses_2017[jaywalking_crim_summonses_2017['priority_corridor'] == 1]) / len(jaywalking_crim_summonses_2017)),1), '%')

# In[308]:


# comparing

non_white_list = ['HISPANIC', 'BLACK', 'ASIAN / PACIFIC ISLANDER', 'OTHER']

print('Non-white population in priority corridors:', demographics_white_nonwhite.loc[1]['frac_non_white'], '%')
print('Non-White proportion of jaywalking summonses in priority corridors', round((100*len(jaywalking_crim_summonses_2017[(jaywalking_crim_summonses_2017['priority_corridor'] == 1) & (jaywalking_crim_summonses_2017['RACE'].isin(non_white_list))]) / len(jaywalking_crim_summonses_2017[(jaywalking_crim_summonses_2017['priority_corridor'] == 1)])),1), '%')
print('Non-white population elsewhere:', demographics_white_nonwhite.loc[0]['frac_non_white'], '%')
print('Non-White proportion of jaywalking summonses elsewhere', round((100*len(jaywalking_crim_summonses_2017[(jaywalking_crim_summonses_2017['priority_corridor'] == 0) & (jaywalking_crim_summonses_2017['RACE'].isin(non_white_list))]) / len(jaywalking_crim_summonses_2017[(jaywalking_crim_summonses_2017['priority_corridor'] == 0)])),1), '%')


# In[112]:


# map with priority corridors and jaywalking summonses

jaywalking_crim_summonses_2017_gdf = gpd.GeoDataFrame(jaywalking_crim_summonses_2017, crs="EPSG:4326", geometry='point_geom')

m = priority_zones.explore(color = 'grey')
jaywalking_crim_summonses_2017_gdf[['BORO', 'PRECINCT_OF_OCCUR', 'RACE','SEX','AGE_GROUP','OFFENSE_DESCRIPTION', 'LAW_SECTION_NUMBER', 'point_geom']].explore(m=m)


# In[119]:


# collisions vs jaywalking summonses by precinct

pedestrian_collisions_2017 = GeoDataFrame(pedestrian_collisions_2017, crs="EPSG:4326", geometry='geometry')
precinct_data['geometry'] = precinct_data['geometry'].apply(wkt.loads)
precinct_data = GeoDataFrame(precinct_data, crs="EPSG:4326", geometry='geometry')
collisions_by_precinct = gpd.sjoin(pedestrian_collisions_2017, precinct_data[['precinct','geometry']]).drop(columns=['index_right'])
collisions_by_precinct = collisions_by_precinct.groupby('precinct').count()[['pedestrian_ksi']].reset_index()

# In[150]:


# comparing regions with high KSI v high jaywalking summonses

summonses_v_ksi = pd.merge(collisions_by_precinct, jaywalking_white_v_nonwhite_2017_groupby.reset_index()[['PRECINCT_OF_OCCUR','Total Jaywalking Summonses']], right_on='PRECINCT_OF_OCCUR', left_on='precinct').drop(columns=['PRECINCT_OF_OCCUR']).rename(columns={'Total Jaywalking Summonses':'total_jaywalking_summonses'}).drop_duplicates()
# summonses_v_ksi.to_csv('../data/output/summonses-vs-ksi_by-precinct_2017-2024.csv')
summonses_v_ksi['percent_pedestrian_ksi'] = round((100*summonses_v_ksi['pedestrian_ksi'] / summonses_v_ksi['pedestrian_ksi'].sum()),2)
summonses_v_ksi['percent_jaywalking_summonses'] = round((100*summonses_v_ksi['total_jaywalking_summonses'] / summonses_v_ksi['total_jaywalking_summonses'].sum()),2)


# In[148]:


summonses_v_ksi

# In[121]:


# to determine percentiles

summonses_v_ksi['percent_jaywalking_summonses'].describe()

# In[122]:


# to determine percentiles

summonses_v_ksi['percent_pedestrian_ksi'].describe()

# In[123]:


# highest and lowest

highest_25th_summonses = summonses_v_ksi[summonses_v_ksi['percent_jaywalking_summonses'] >= 1.89].sort_values('percent_jaywalking_summonses', ascending=False) # above 75th percentile
highest_10_summonses = summonses_v_ksi.sort_values('percent_jaywalking_summonses', ascending=False).head(10) # top 10

highest_25th_ksi = summonses_v_ksi[summonses_v_ksi['percent_pedestrian_ksi'] >= 1.7].sort_values('percent_pedestrian_ksi', ascending=False) # above 75th percentile
highest_10_ksi = summonses_v_ksi.sort_values('percent_pedestrian_ksi', ascending=False).head(10) # top 10

lowest_25th_ksi = summonses_v_ksi[summonses_v_ksi['percent_pedestrian_ksi'] <= 0.9525].sort_values('percent_pedestrian_ksi', ascending=False) # bottom 25th percentile
lowest_10_ksi = summonses_v_ksi.sort_values('percent_pedestrian_ksi').head(10) # lowest 10

# In[124]:


# how many precincts with highest share of summonses also have the among lowest share of pedestrian KSI? (top 25th v bottom 25th)

for p in highest_25th_summonses['precinct'].values:
    if p in lowest_25th_ksi['precinct'].values:
        print(p) 

# In[125]:


summonses_v_ksi[summonses_v_ksi['precinct'].isin([76, 33, 28, 9])]

# In[126]:


# how many precincts with highest share of summonses also have the among highest share of pedestrian KSI? (top 10 v top 10)

for p in highest_10_summonses['precinct'].values:
    if p in highest_10_ksi['precinct'].values:
        print(p) 

# In[127]:


summonses_v_ksi[summonses_v_ksi['precinct'] == 44]

# ### Comparing to CJRA Offenses

# In[128]:


urination = ['16-118(6)','16-1186'] 
littering = ['16-118(1)'] # there's also 16-118, (2), and (4)... do they count?
alcohol = ['10-125(2B)','10-125'] # -> '10-125' as well?
noise = ['24-218(A)'] # -> '24-218' as well?
park = [ # just using exact matches with CJRA spreadsheet law section numbers (file:///Users/ravram/Downloads/CJRA%20Quarterly%20Report%20Vol%206%20No%204_Final.pdf)
        '1-03C (1)','1-03(C) 2','103.C2','1-04K','104-K','1-04(C)(1)','1-04(D)',
        '1-04(Q)','1-04(I)','1-04(B) 1','1-04(O)','1-05(F) 1','1-05(F)(1)','1-05N',
        '1-05(I)','1-05I','1-05(D)(1)','1-05(D)(2)','1-05(M)(1)'
       ]

code_list = urination + littering + alcohol + noise + park

# updated df with CJRA law codes

drop_na = criminal_summonses_2017.dropna(subset=['LAW_SECTION_NUMBER'])
CJRA_law_codes = drop_na[drop_na['LAW_SECTION_NUMBER'].isin(code_list)]


# In[129]:


# racial breakdown of jaywalking vs CJRA vs non-CJRA summonses (2017-2023)
# including 'Unknown'

# CJRA summonses (2017-2023)

CJRA_law_codes['RACE'].replace(['BLACK HISPANIC','WHITE HISPANIC'], 'HISPANIC', inplace=True) # collapsing hispanic categories
CJRA_law_codes['RACE'].replace('AMERICAN INDIAN/ALASKAN NATIVE', 'OTHER', inplace=True) 
CJRA_law_codes['RACE'].replace('(null)', 'UNKNOWN', inplace=True) 

racial_breakdown_CJRA = pd.DataFrame(round((100*CJRA_law_codes['RACE'].value_counts() / CJRA_law_codes['RACE'].count()),2)).rename(columns={'count':'CJRA'})

# non-CJRA summonses (2017-2023)

# list of non_CJRA/ jaywalking law codes
non_CJRA_list = [var for var in criminal_summonses_2017['LAW_SECTION_NUMBER'].unique() if var not in code_list + jaywalking_codes]
non_CJRA_law_codes = criminal_summonses_2017[criminal_summonses_2017['LAW_SECTION_NUMBER'].isin(non_CJRA_list)]

non_CJRA_law_codes['RACE'].replace(['BLACK HISPANIC','WHITE HISPANIC'], 'HISPANIC', inplace=True) # collapsing hispanic categories
non_CJRA_law_codes['RACE'].replace('AMERICAN INDIAN/ALASKAN NATIVE', 'OTHER', inplace=True) 
non_CJRA_law_codes['RACE'].replace('(null)', 'UNKNOWN', inplace=True) 

racial_breakdown_CJRA['Non-CJRA'] = round((100*non_CJRA_law_codes['RACE'].value_counts() / non_CJRA_law_codes['RACE'].count()),2)

# jaywalking summonses (2017-2023)

jaywalking_crim_summonses_2017['RACE'].replace(['BLACK HISPANIC','WHITE HISPANIC'], 'HISPANIC', inplace=True) # collapsing hispanic categories
jaywalking_crim_summonses_2017['RACE'].replace('AMERICAN INDIAN/ALASKAN NATIVE', 'OTHER', inplace=True) 
jaywalking_crim_summonses_2017['RACE'].replace('(null)', 'UNKNOWN', inplace=True) 

racial_breakdown_CJRA['Jaywalking'] = round((100*jaywalking_crim_summonses_2017['RACE'].value_counts() / jaywalking_crim_summonses_2017['RACE'].count()),2)

racial_breakdown_CJRA.to_csv('../data/output/racial_breakdown_CJRA_vs_jaywalking.csv')

# In[130]:


racial_breakdown_CJRA

# #### Update for 2024 Q1-Q2

# In[203]:


jaywalking_crim_summonses_2024 = jaywalking_crim_summonses[jaywalking_crim_summonses['quarter'].isin(['2024 Q1', '2024 Q2'])]

# In[204]:


# for Q1 97.22% Non-White, for Q2 93.40% Non-White
# pretty normal compared to other quarters

pd.DataFrame(round((100*jaywalking_crim_summonses_2024[jaywalking_crim_summonses_2024['quarter'].isin(['2024 Q1', '2024 Q2'])]['RACE'].value_counts() / len(jaywalking_crim_summonses[jaywalking_crim_summonses['quarter'].isin(['2024 Q1', '2024 Q2'])])),2)).rename(columns={'count':'Share of Summonses'})

# In[273]:


collision_dataset_updated = pd.read_csv('../data/input/Motor_Vehicle_Collisions_-_Crashes_20240828.csv')
collision_dataset_updated = collision_dataset_updated[collision_dataset_updated['LATITUDE'].notnull()]
collision_dataset_updated['geometry'] = collision_dataset_updated.apply(lambda x: Point(x['LONGITUDE'], x['LATITUDE']), axis=1)
collision_dataset_updated = gpd.GeoDataFrame(collision_dataset_updated, crs="EPSG:4326", geometry='geometry')


# In[274]:


# creating quarter column 
collision_dataset_updated['CRASH DATE'] = pd.to_datetime(collision_dataset_updated['CRASH DATE'])
collision_dataset_updated['quarter'] = collision_dataset_updated['CRASH DATE'].dt.quarter.astype(str) 
collision_dataset_updated['quarter'] = 'Q' + collision_dataset_updated['quarter']
collision_dataset_updated['year'] = collision_dataset_updated['CRASH DATE'].dt.year.astype(str)
collision_dataset_updated['quarter'] = collision_dataset_updated['year'] + ' ' + collision_dataset_updated['quarter']


# In[275]:


# pedestrian deaths over time

collision_dataset_updated['CRASH DATE'] = pd.to_datetime(collision_dataset_updated['CRASH DATE'])
collision_dataset_updated['NUMBER OF PEDESTRIANS KILLED'].groupby(collision_dataset_updated['CRASH DATE'].dt.to_period('Q')).sum().loc[:'2024Q2'].plot()

# In[278]:


# uploading precinct data

jaywalking_summonses_2024_by_precinct = jaywalking_crim_summonses_2024.groupby('PRECINCT_OF_OCCUR').count().rename(columns={'SUMMONS_KEY':'Jaywalking Summonses'})[['Jaywalking Summonses']]
jaywalking_summonses_2024_by_precinct['Total population'] = precinct_data['Total population']
jaywalking_summonses_2024_by_precinct['Jaywalking Summonses per 10K'] = round((10000*jaywalking_summonses_2024_by_precinct['Jaywalking Summonses'] / jaywalking_summonses_2024_by_precinct['Total population']),1)
jaywalking_summonses_2024_by_precinct['geometry'] = precinct_data['geometry']

# jaywalking_summonses_2024_by_precinct['geometry'] = jaywalking_summonses_2024_by_precinct['geometry'].apply(wkt.loads)
jaywalking_summonses_2024_by_precinct = GeoDataFrame(jaywalking_summonses_2024_by_precinct, crs="EPSG:4326", geometry='geometry')
jaywalking_summonses_2024_by_precinct['PRECINCT_OF_OCCUR'] =jaywalking_summonses_2024_by_precinct.index


# In[279]:


# choropleth map of regions with greatest rate of jaywalking criminal summonses

# map specifications
zoom = 9.8
lat = 40.706000
lon = -73.976300

# starting map

fig = folium.Map(location=[lat, lon], zoom_start=zoom, tiles='cartodbpositron') # blank map

# setting column values and pop-up aliases for all variables with MOEs and CVs

columns = ['PRECINCT_OF_OCCUR','Jaywalking Summonses per 10K'] # uses 'for map' columns so only regions w/ reliable values (based on CV) are colored in based on gen_reliable_col() output df
aliases= ['Precinct','Summonses per 10K']

choropleth = folium.Choropleth(
    geo_data = jaywalking_summonses_2024_by_precinct,
    name = 'choropleth',
    data = jaywalking_summonses_2024_by_precinct,
    columns = columns,
    key_on = 'feature.properties.PRECINCT_OF_OCCUR',
    fill_color = 'Blues',
    fill_opacity = 0.7,
    line_opacity = 0.2,
    nan_fill_color='grey',
    use_jenks = True, # check with Rose
    legend_name = 'Jaywalking Criminal Summonses per 10,000 people (2024 Q1 & Q2)',
    highlight = True,
).add_to(fig)

choropleth.color_scale.width=500 # setting legend width

choropleth.geojson.add_child( 
    folium.features.GeoJsonTooltip(columns, aliases=aliases, labels=True))

# adding Data Team logo

# FloatImage(b64_image(data_team_logo), bottom=1, left=1).add_to(fig)
      
fig

# In[280]:


# precinct_data['geometry'] = precinct_data['geometry'].apply(wkt.loads)
# precinct_data = gpd.GeoDataFrame(precinct_data, crs="EPSG:4326", geometry='geometry')

collision_dataset_updated = collision_dataset_updated.sjoin(precinct_data[['precinct', 'geometry']])
collision_dataset_updated = collision_dataset_updated.merge(precinct_data.rename(columns={'geometry':'precinct_geom'})[['precinct','precinct_geom']], on='precinct')

# In[281]:


collision_dataset_2024 = collision_dataset_updated[collision_dataset_updated['quarter'].isin(['2024 Q1', '2024 Q2'])]
collision_dataset_by_precinct_2024 = collision_dataset_2024.groupby('precinct').agg({'NUMBER OF PEDESTRIANS KILLED':'sum'})

# In[282]:


# finding ratio of summonses to deaths

jaywalking_and_collisions_2024 = collision_dataset_by_precinct_2024.reset_index().merge(jaywalking_summonses_2024_by_precinct.drop(columns=['PRECINCT_OF_OCCUR']).reset_index()[['PRECINCT_OF_OCCUR','Jaywalking Summonses','geometry']], left_on='precinct', right_on='PRECINCT_OF_OCCUR')
jaywalking_and_collisions_2024 = jaywalking_and_collisions_2024.drop(columns=['PRECINCT_OF_OCCUR'])
jaywalking_and_collisions_2024['ratio'] = jaywalking_and_collisions_2024['Jaywalking Summonses'] / jaywalking_and_collisions_2024['NUMBER OF PEDESTRIANS KILLED'] 
# jaywalking_and_collisions_2024 = jaywalking_and_collisions_2024.replace(np.inf, np.nan)
jaywalking_and_collisions_2024.loc[jaywalking_and_collisions_2024['ratio'] == np.inf, 'ratio'] = jaywalking_and_collisions_2024['Jaywalking Summonses']
jaywalking_and_collisions_2024.index = jaywalking_and_collisions_2024['precinct']

# In[283]:


jaywalking_and_collisions_2024 = gpd.GeoDataFrame(jaywalking_and_collisions_2024)

fig = folium.Map(location=[lat, lon], zoom_start=zoom, tiles='cartodbpositron') # blank map

# setting column values and pop-up aliases for all variables with MOEs and CVs

columns = ['precinct', 'ratio', 'NUMBER OF PEDESTRIANS KILLED', 'Jaywalking Summonses'] # uses 'for map' columns so only regions w/ reliable values (based on CV) are colored in based on gen_reliable_col() output df
aliases = ['Precinct','Summonses Per Pedestrian Death', 'Pedestrians Killed', 'Jaywalking Summonses']

choropleth = folium.Choropleth(
    geo_data = jaywalking_and_collisions_2024,
    name = 'choropleth',
    data = jaywalking_and_collisions_2024,
    columns = columns,
    key_on = 'feature.properties.precinct',
    fill_color = 'Blues',
    fill_opacity = 0.7,
    line_opacity = 0.2,
    nan_fill_color='grey',
    use_jenks = True, # check with Rose
    legend_name = 'Summonses Per Pedestrian Death (2024 Q1 & Q2)',
    highlight = True,
).add_to(fig)

choropleth.color_scale.width=500 # setting legend width

choropleth.geojson.add_child( 
    folium.features.GeoJsonTooltip(columns, aliases=aliases, labels=True))

      
fig

# In[284]:


# creating column indicating whether or not summons was given in a priority corridor
jaywalking_crim_summonses_2024['point_geom'] = [Point(xy) for xy in zip(jaywalking_crim_summonses_2024['Longitude'], jaywalking_crim_summonses_2024['Latitude'])]
jaywalking_crim_summonses_2024['priority_corridor'] = jaywalking_crim_summonses_2024['point_geom'].apply(
    lambda point: 1 if combined_geometry.contains(point) else 0
)

# now for collisions (using different method because above code is too slow for a dataset this size)
collision_dataset_updated = gpd.sjoin(collision_dataset_updated, priority_zones[['the_geom', 'sq_mi']], how='left').rename(columns={'sq_mi':'priority_corridor'}).drop(columns=['index_right'])
collision_dataset_updated['priority_corridor'] = np.where(collision_dataset_updated['priority_corridor'].isnull(), 0, 1)

# In[285]:


# creating dataset locating all of the pedestrian-involved car accidents

collision_dataset_updated['pedestrian_ksi'] = collision_dataset_updated['NUMBER OF PEDESTRIANS INJURED'] + collision_dataset_updated['NUMBER OF PEDESTRIANS KILLED']	
pedestrian_collisions_2024 = collision_dataset_updated[collision_dataset_updated['pedestrian_ksi'] != 0]


# In[289]:


# comparing

print('Percent of total population living in priority corridors:', round((100*demographics_white_nonwhite['pop_estimate_pluto'].loc[1] / demographics_white_nonwhite['pop_estimate_pluto'].sum()),1), '%')
print('Percent of pedestrian KSI occuring in priority corridors:', round((100*len(pedestrian_collisions_2024[pedestrian_collisions_2024['priority_corridor'] == 1]) / len(pedestrian_collisions_2024)),1), '%')
print('Percent of jaywalking summonses occuring in priority corridors:', round((100*len(jaywalking_crim_summonses_2024[jaywalking_crim_summonses_2024['priority_corridor'] == 1]) / len(jaywalking_crim_summonses_2024)),1), '%')

# In[307]:


# comparing

non_white_list = non_white_list = ['BLACK HISPANIC', 'WHITE HISPANIC', 'BLACK', 'ASIAN / PACIFIC ISLANDER', 'OTHER']

print('Non-white population in priority corridors:', demographics_white_nonwhite.loc[1]['frac_non_white'], '%')
print('Non-White proportion of jaywalking summonses in priority corridors', round((100*len(jaywalking_crim_summonses_2024[(jaywalking_crim_summonses_2024['priority_corridor'] == 1) & (jaywalking_crim_summonses_2024['RACE'].isin(non_white_list))]) / len(jaywalking_crim_summonses_2024[(jaywalking_crim_summonses_2024['priority_corridor'] == 1)])),1), '%')
print('Non-white population elsewhere:', demographics_white_nonwhite.loc[0]['frac_non_white'], '%')
print('Non-White proportion of jaywalking summonses elsewhere', round((100*len(jaywalking_crim_summonses_2024[(jaywalking_crim_summonses_2024['priority_corridor'] == 0) & (jaywalking_crim_summonses_2024['RACE'].isin(non_white_list))]) / len(jaywalking_crim_summonses_2024[(jaywalking_crim_summonses_2024['priority_corridor'] == 1)])),1), '%')


# In[ ]:


jaywalking_crim_summonses_2024[(jaywalking_crim_summonses_2024['priority_corridor'] == 1) & (jaywalking_crim_summonses_2024['RACE'].isin(non_white_list))]





# In[290]:


# map with priority corridors and jaywalking summonses

jaywalking_crim_summonses_2024_gdf = gpd.GeoDataFrame(jaywalking_crim_summonses_2024, crs="EPSG:4326", geometry='point_geom')

m = priority_zones.explore(color = 'grey')
jaywalking_crim_summonses_2024_gdf[['BORO', 'PRECINCT_OF_OCCUR', 'RACE','SEX','AGE_GROUP','OFFENSE_DESCRIPTION', 'LAW_SECTION_NUMBER', 'point_geom']].explore(m=m)

