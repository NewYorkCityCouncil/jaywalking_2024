### Jaywalking Analysis 2024
Data analysis for Int 0346-2024 ['Pedestrian crossing guidelines and right of way.'](https://legistar.council.nyc.gov/LegislationDetail.aspx?ID=6557803&GUID=7D6F4CEC-85C3-4E00-9E54-36641179493B&Options=&Search=)

***  

#### Data Sources 
- Traffic Accidents (Offenses), Denver Open Data Catalogue, Accessed July 31st 2024, [link](https://data.cityofnewyork.us/Transportation/Bus-Breakdown-and-Delays/ez4e-fazm)
- Crash Data, Virginia Roads, Accessed August 1st 2024, [link](https://www.virginiaroads.org/maps/VDOT::crash-data-1/about)
- Motor Vehicle Collisions - Person, New York City Open Data, Accessed TBD, [link](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu/about_data)
- Motor Vehicle Collisions - Crashes, New York City Open Data, Accessed TBD, [link](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)
- NYPD Criminal Court Summons Incident Level Data (Year To Date), New York City Open Data, Accessed TBD, [link](https://data.cityofnewyork.us/Public-Safety/NYPD-Criminal-Court-Summons-Incident-Level-Data-Yea/8h9b-rp9u/about_data)
- NYPD Criminal Court Summons Incident Level Data (Historic), New York City Open Data, Accessed TBD, [link](https://data.cityofnewyork.us/Public-Safety/NYPD-Criminal-Court-Summons-Historic-/sv2w-rv3k/about_data)


#### Methodology 

##### Summary & Intention
Pedestrians throughout New York City navigate busy streets daily, but when jaywalking laws are inconsistently enforced or disproportionately applied, concerns around safety and fairness arise. Jaywalking, which refers to crossing streets in violation of traffic laws, has been enforced since the 1920s but is now rarely a focus in NYC, with only 467 summonses issued in 2023. However, 92% of these summonses were given to Black or Latino individuals, prompting questions about the role of racial profiling in traffic enforcement.

Walking against traffic signals was decriminalized statewide in Virginia in March 2021. Then in January 2023, it was decriminalized in the city of Denver, CO. On July 13, 2013, and again on February 28th, 2024, the New York City Council introduced new legislation to amend the administrative code of the city of New York, in relation to pedestrian crossing guidelines and right of way. 

The data team analyzed citation and crash data from NYC and other cities in order to:
- Test if legalization is a predictor of a change in pedestrian incidents in other cities
- Identify if ticks are correlated to a decreased number of pedestrian collisions in NYC
- Evaluate the NYPD’s claim that they target neighborhoods with high pedestrian KSI with citations
- Assess if the number of tickets given out is a predictor of the number of collisions in NYC

#### Main Takeaways
- **Legalization in Denver, CO and Virginia Beach, VA had no impact on the predictions of the percentage of pedestrian collisions or the percentage of non-intersection pedestrian collisions each month.** We chose percentage of pedestrian collisions to avoid increases caused by weather or more cars being on the road on a given month. We assume that if the weather or more cars being on the road lead to more pedestrian collisions it would also lead to a similar number of non-pedestrian collisions. All of models revealed either no statistical significance or very small coefficients. 
- **There is no relationship between the number of citations each quarter and the pedestrian-involved incidents to imply that giving more citations leads to less pedestrian incidents in NYC.**
- **In NYC, there is a statistically significant moderate relationship between pedestrian KSI and jaywalking summons per 10,000 people per precinct.** 
- **The number of citations in the previous quarter granger causes the pedestrian share of collisions in the next quarter. However this does not imply true causality. The effect is small and the model is not very predictive.** 


#### Legislation
In 2023 and again in 2024, City Council introduced legislation to decriminalize jaywalking, aiming to address the disproportionate enforcement of jaywalking citations, which city data shows are overwhelmingly issued to Black and Latino New Yorkers. The proposed bill, which passed the Transportation and Infrastructure Committee, sought to stop the NYPD from penalizing pedestrians for crossing streets outside of crosswalks. After further revisions, the Council has postponed a full vote, leaving the future of jaywalking decriminalization uncertain.
