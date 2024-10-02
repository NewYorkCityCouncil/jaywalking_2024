### Jaywalking Analysis 2024
Data analysis for Int 0346-2024 ['Pedestrian crossing guidelines and right of way.'](https://legistar.council.nyc.gov/LegislationDetail.aspx?ID=6557803&GUID=7D6F4CEC-85C3-4E00-9E54-36641179493B&Options=&Search=)

***  

#### Data Sources 
- Traffic Accidents (Offenses), Denver Open Data Catalogue, Accessed July 31st 2024, [link](https://opendata-geospatialdenver.hub.arcgis.com/datasets/db00bd99ea534d8987e0913a191ebe19_325/about)
- Crash Data, Virginia Roads, Accessed August 1st 2024, [link](https://www.virginiaroads.org/maps/VDOT::crash-data-1/about)
- Motor Vehicle Collisions - Person, New York City Open Data, Accessed TBD, [link](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu/about_data)
- Motor Vehicle Collisions - Crashes, New York City Open Data, Accessed TBD, [link](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)
- NYPD Criminal Court Summons Incident Level Data (Year To Date), New York City Open Data, Accessed TBD, [link](https://data.cityofnewyork.us/Public-Safety/NYPD-Criminal-Court-Summons-Incident-Level-Data-Ye/mv4k-y93f/about_data)
- NYPD Criminal Court Summons Incident Level Data (Historic), New York City Open Data, Accessed TBD, [link](https://data.cityofnewyork.us/Public-Safety/NYPD-Criminal-Court-Summons-Historic-/sv2w-rv3k/about_data)


#### Methodology 

##### Summary & Intention
Pedestrians throughout New York City navigate busy streets daily, but when jaywalking laws are inconsistently enforced or disproportionately applied, concerns around safety and fairness arise. Jaywalking, which refers to crossing streets in violation of traffic laws, has been enforced since the 1920s but is now rarely a focus in NYC, with only 467 summonses issued in 2023. However, 92% of these summonses were given to Black or Latino individuals, prompting questions about the role of racial profiling in traffic enforcement.

Walking against traffic signals was decriminalized statewide in Virginia in March 2021. Then in January 2023, it was decriminalized in the city of Denver, CO. On July 13, 2013, and again on February 28th, 2024, the New York City Council introduced new legislation to amend the administrative code of the city of New York, in relation to pedestrian crossing guidelines and right of way. 

The data team analyzed citation and crash data from NYC and other cities in order to:
- Test if legalization is a predictor of a change in pedestrian incidents in other cities
- Identify if tickets are correlated to a decreased number of pedestrian collisions in NYC
- Evaluate the NYPD’s claim that they target neighborhoods with high pedestrian KSI with citations
- Determine if there is a racial disparity in how jaywalking summons are administered
- Assess if the number of tickets given out is a predictor of the number of collisions in NYC

#### Main Takeaways
- **Legalization in Denver, CO and Virginia Beach, VA had no impact on the predictions of the percentage of pedestrian collisions or the percentage of non-intersection pedestrian collisions each month.** We used the pedestrian collisions divided by all collisions (the share) to normalize for natural fluctuations in the total number of collisions over time. We assume that factors such as weather and traffic volume would lead to similar increases in pedestrian and non-pedestrian collisions, thus minimizing their impact on our models.  All of the models we tested revealed either no statistical significance or very small coefficients (β = 0.0008) for the features we included to represent legalization.
- **We found weak to moderate statistically significant positive relationships between the quarterly number of criminal summonses issued for jaywalking and pedestrian-involved/ jaywalking-involved collisions (with a one quarter lag), suggesting that summonses and incidents rise and fall together.** A negative relationship, where increases in summonses issued correlate with decreases in accidents and vice versa, would have been much more suggestive of jaywalking summonses protecting against accidents. 
- **In NYC, there is a statistically significant moderate relationship between pedestrians killed and injured (PKI) and jaywalking summons issued per 10,000 people per precinct. This suggests that there is some relationship between the sites of severe accidents and the locations targeted with jaywalking summonses.** However, some precincts, like 9 (MN), 28 (MN), 33 (MN), and 76 (BK), are exceptions, as they fall in the top 25th percentile for their share of the city’s jaywalking summonses and the bottom 25th percentile for their share of the city’s total PKI.
- 52% of all pedestrian-related car crashes occur in Vision Zero Priority Zones, and 58% of jaywalking summonses are given in priority zones. **However, there is a strong racial disparity in these regions. The priority zones are about 73% non-white, but 92% of jaywalking summonses in these areas are given to non-white people.**
- **The results of our Granger causality test revealed that the number of citations in the previous quarter is associated with changes in the pedestrian-involved share of collisions in the next quarter. However, it is important to note that Granger causality does not necessarily imply true causality.** Granger causality is rooted in predictability and correlation rather than a direct cause-and-effect relationship. **In the case of citations predicting the share of pedestrian collisions, it is especially unlikely to indicate true causality** given the effect is small (β = -0.0039) and the model is not very predictive. The mean magnitude of error is 124% indicating that the error is often larger than that actual value itself.  



#### Legislation
In 2023 and again in 2024, City Council introduced legislation to decriminalize jaywalking, aiming to address the disproportionate enforcement of jaywalking citations, which city data shows are overwhelmingly issued to Black and Latino New Yorkers. The proposed bill, which passed the Transportation and Infrastructure Committee, sought to stop the NYPD from penalizing pedestrians for crossing streets outside of crosswalks. After further revisions, the Council has postponed a full vote, leaving the future of jaywalking decriminalization uncertain.
