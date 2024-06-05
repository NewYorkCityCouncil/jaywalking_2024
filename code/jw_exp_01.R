source("code/00_load_dependencies.R")
# using open data from nypd summons
# https://data.cityofnewyork.us/Public-Safety/NYPD-Criminal-Court-Summons-Incident-Level-Data-Ye/mv4k-y93f/about_data
# year to data - is this a good time frame?
# year over year too
# by race
# normalized by population
# relative to race breakdown all summons - is this worse better 
# looking at other descr of summons for same place and time 
# june 25 hearing 
# quality of life - disorderly 
# traffic deaths


# using the community created sub table for jwalking to get desc used
odc <- vroom("https://data.cityofnewyork.us/resource/f7n3-uvc8.csv") %>% 
  as.data.table()
jwd <- unique(odc$offense_description)
# using ytd -- 268 tickets/1.13% of all summons ytd

sumsdt <- vroom("https://data.cityofnewyork.us/resource/mv4k-y93f.csv?$limit=99999999") %>% 
  as.data.table()
sumsdt[offense_description %in% jwd, length(unique(summons_key))]/sumsdt[, length(unique(summons_key))] * 100
subsums <- sumsdt[offense_description %in% jwd, ]
subsums[, length(unique(summons_key)), by = "race"][order(V1, decreasing = TRUE)]
sums <- st_read("https://data.cityofnewyork.us/resource/mv4k-y93f.geojson?$limit=9999999999") %>%
  st_as_sf()
sumss <- sums[sums$offense_description %in% jwd, ]
sumss %>% st_as_sf() %>% mapview()

sumsdt[, length(unique(summons_key)), by = offense_description][order(V1, decreasing = TRUE)][1:30, ]
 
