---
layout: post
author: YJ Park
title:  "Ride-sharing behaviors of New York City's taxi and rideshare users"
date:   2020-04-11 07:50:00 -700
categories: jekyll update
tags: ride-sharing behaviors, New York City's taxi, New York City's rideshare, ride-sharing ratios
---
<head>
	<!-- Global site tag (gtag.js) - Google Analytics -->
	<script async src="https://www.googletagmanager.com/gtag/js?id=UA-127453746-1"></script>
	<script>
		  window.dataLayer = window.dataLayer || [];
		  function gtag(){dataLayer.push(arguments);}
		  gtag('js', new Date());

		  gtag('config', 'UA-127453746-1');
	</script>
</head>

# Project: Ride-sharing behaviors of New York City's taxi and rideshare users

## Introduction
---

The purpose of this sub-project is to explore sharing patters of passengers who are taking New York City's taxis and rideshares, including Green Boro taxis, Yellow Medallion taxis, Uber and Lyft. 

I found that:
1) When examining only ride areas with greater than 10,000 occurrences during rush hours, 102 data points are available. However, this number is dominated by rideshares such as Uber and Lyft (85 out of 102), not taxis. It seems that between July 2017 and July 2019, rideshares were popular during rush hours. The other interpretation can be the data for taxis may not be complete so less number of occurrences could have been recorded.

2) Shared ratios for green taxis seems to be lowest, consistently between 12.81% and 21.07%. However, the interpretation here is quite difficult to make because the comparison between rideshare and two taxis is not a fair one. The only clear points here are: where there is one passenger only, these rides are not shared; and the ride-sharing ratio of yellow taxis seems to be higher than that of green taxis.

3) The average shared ratios for Brooklyn seems to be higher than those of Manhattan and Queens. For rush hours, not much data is available for the shared ratios for Bronx.

![map_over_10k](../../../../../../assets/images/map_over_10k.png)


## Import packages
---

Prior to starting any analysis and visualization, I will import relevant packages to be used in this project.

```
#To get automatic reloading and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import numpy as np
import pandas as pd
#graph visualization
import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt

import folium #folium map visualization

```

## Create dataframes from available csv files
---

The New York City (hereafter, NYC)'s taxi and ride-share data is available here: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

In my case, I had New York City's Bike, MTA, Green, Yellow and Rideshare data available. However, because of their vast amount, I decided to focus on transportation modes where a single passenger would most likely use but having a sharing capacity, hence taxis and ridesharing methods such as Uber and Lyft.

To explore the data, I import the data of these transportation modes into csv files into Pandas dataframe first.

```
nyc_green_taxi_df = pd.read_csv('path_to_your_data/nyc_green_taxi.csv')
nyc_yellow_taxi_df = pd.read_csv('path_to_your_data/nyc_yellow_taxi.csv')
rideshare_df = pd.read_csv('path_to_your_datanyc_rideshare.csv')
```

I also need to create the dataframe for the mapping between the location IDs and borough (and zone) for NYC for visualization tasks later.
There are two datasets I use to this task: 1) NYC's taxi and Limousine Commission (TLC) zone data; and 2) NYC's geographical mapping between Neighborhood Tabulation Areas (NTA) codes and geographical coordinates.

```
tlc_df = pd.read_csv('path_to_your_data/tlc_zones.csv')
geo = pd.read_csv('path_to_your_data/geographic.csv')
```

From TLC data, location_id, borough, zone and nta_code are available.

![tlc_df](../../../../../../assets/images/tlc_df.PNG)

TLC data shows there are 69 zones in Manhattan and Queens and other major zones are 61 in Brooklyn and 43 in Bronx.

![tlc_df_count](../../../../../../assets/images/tlc_df_count.PNG)

Geo data can provide nta codes with their boundary coordinates.

![geo_df](../../../../../../assets/images/geo_df.PNG)


When I join nta_code between these two dataframes, I will be able to map from location_id to boough and zone to geographical coordinates for analysis and visualization later.
To make this task simple, I created an utility dataframe to hold first x and y coordinates of each nta_code.

```
nta_df = pd.DataFrame(columns=['nta', 'longitude', 'latitude'])
nta_df['nta'] = geo.columns
nta_df['longitude'] = list(geo.iloc[0])
nta_df['latitude'] = list(geo.iloc[1])
```

The simplified dataframe of geo_df is now transferred to nta_df.

![nta_df](../../../../../../assets/images/nta_df.png)


## Explore basic data available for taxi and rideshare information
---

Rideshare dataframe includes pick-up and drop-off location IDs. It also includes the indication of this ride was shared or not. The data is missing a lot of coordinates of these location ids. In addition, shared_ride indicates that whether this particular ride is shared and the value is boolean rather than displaying the number of passengers shared this ride.

![rideshare_df_example](../../../../../../assets/images/rideshare_df_example.png)

Data collected from using taxis seems to be more extensive include the location information about geogrphical coordinates, trip distance and the number of passengers. However, it is missing a lot of IDs of pick up and drop off locations.

![green_taxi_df_example](../../../../../../assets/images/green_taxi_example.png)

![yellow_taxi_df_example](../../../../../../assets/images/yellow_taxi_example.png)

How many data points are available? I would like to look at the number of data points that are not NaN in the dataframe for each transport.

![number_of_datapoints](../../../../../../assets/images/number_of_datapoints.png)

Overall, it seems that the inconsistency using location coordinates or IDs needs to be addressed to combine the data from rideshare and taxis together.

## Working with the data as it is - dropping NaN on missing location IDs
---

I decided to work with the data as it is so I am dropping all NaN values.

```
filtered_nyc_yellow_df = nyc_yellow_taxi_df.dropna(subset=['pickup_location_id', 'dropoff_location_id'])
filtered_nyc_green_df = nyc_green_taxi_df.dropna(subset=['pickup_location_id', 'dropoff_location_id'])
filtered_rideshare_df = rideshare_df.dropna(subset=['pickup_location_id', 'dropoff_location_id'])
```

![drop_na_example](../../../../../../assets/images/drop_na_example.png)

When I looked at the rideshare data after dropping all NaN values, unfortunately, the shared_ride indicator is much short than other fields. Why is this the case?

------------

According to the data schema, this indicator for shared ride was added from 2017 July. To make the data consistent across rideshare and taxis, I will reduce the time period of analysis between 2017 July to the end of 2019. Because this is the most recent available time periods.

## Filter the data period between 2017 and 2019 and make a new field, shared_ride, in taxi dataframes based on passenger_count
---

### Restrict the time period for three dataframes

First, I changed pick up and drop off datetime fields into pandas date time data type so that we can work with limiting the time period easier.


```
#yellow taxis
filtered_nyc_yellow_df['pickup_datetime'] = pd.to_datetime(filtered_nyc_yellow_df['pickup_datetime'])
filtered_nyc_yellow_df['dropoff_datetime'] = pd.to_datetime(filtered_nyc_yellow_df['dropoff_datetime'])
#green taxis
filtered_nyc_green_df['pickup_datetime'] = pd.to_datetime(filtered_nyc_green_df['pickup_datetime'])
filtered_nyc_green_df['dropoff_datetime'] = pd.to_datetime(filtered_nyc_green_df['dropoff_datetime'])
#rideshare
filtered_rideshare_df['pickup_datetime'] = pd.to_datetime(filtered_rideshare_df['pickup_datetime'])
filtered_rideshare_df['dropoff_datetime'] = pd.to_datetime(filtered_rideshare_df['dropoff_datetime'])
```

Then, I masked only limited time periods, in this case, 2017 July to 2019 December.

```
#decide start and end dates
start_date = '2017-07-01'
end_date = '2019-12-31'

#mask other dates
yellow_mask = (filtered_nyc_yellow_df['pickup_datetime'] >= start_date) & (filtered_nyc_yellow_df['dropoff_datetime'] <= end_date)
green_mask = (filtered_nyc_green_df['pickup_datetime'] > start_date) & (filtered_nyc_green_df['dropoff_datetime'] <= end_date)
rideshare_mask = (filtered_rideshare_df['pickup_datetime'] >= start_date) & (filtered_rideshare_df['dropoff_datetime'] <= end_date)

#get filtered time periods
filtered_nyc_yellow_df = filtered_nyc_yellow_df.loc[yellow_mask]
filtered_nyc_green_df = filtered_nyc_green_df.loc[green_mask]
filtered_rideshare_df = filtered_rideshare_df.loc[rideshare_mask]
```

It turned out that actual 2019 data is up to June 30th rather than December 31st for all datasets.

```
time_filtered_nyc_yellow_df['pickup_datetime'].min(), time_filtered_nyc_yellow_df['pickup_datetime'].max(),\
time_filtered_nyc_yellow_df['dropoff_datetime'].min(), time_filtered_nyc_yellow_df['dropoff_datetime'].max()
```

![actual_timeperiod](../../../../../../assets/images/actual_timeperiod.png)

In the end, I have approximately data points of 440k for yellow taxis, 200k for green taxis and 2.3 mil for rideshares.

![time_filtered](../../../../../../assets/images/time_filtered.png)

### Replace NaN to 0 value for the shared_ride indicator in rideshare_df

Because currently the shared_ride indicator in rideshare_df only has the value of 1 if shared, its values are counted as 600k rather than 2.3 mil. I need to fill in 0 for this indicator so that it is a binary indicator.

```
time_filtered_rideshare_df = time_filtered_rideshare_df.fillna(0)
```

Now, shared_ride is counted as the same number with other fields.
![shared_ride](../../../../../../assets/images/shared_ride.png)

### Create a shared_ride indicator for taxis for a comparison

Finally, I will create a shared_ride indicator for taxi dataframes based on the passenger count. If the number of passengers are larger than 1, this will be the indicator 1, otherwise 0.

This is not strictly a like-to-like comparison because passengers greater than 1 may not be a sharing; for example, a couple in a taxi would be counted as 2 but still not sharing with other passengers.
Similarly, for rideshare such as Uber and Lyft, the shared_ride indicator 0 may represent a group of passengers greater than one.


```
def classify_shared_ride(row):
    if row['passenger_count']>1.0:
        return 1
    elif row['passenger_count']==1.0:
        return 0
    else:
        return np.NaN

time_filtered_nyc_yellow_df['shared_ride'] = filtered_nyc_yellow_df.apply(lambda row: classify_shared_ride(row), axis=1)
time_filtered_nyc_green_df['shared_ride'] = filtered_nyc_green_df.apply(lambda row: classify_shared_ride(row), axis=1)
```

After creating a ride-sharing indicator, I will drop all NaN values because there are cases where the number of passenger was recorded as 0. 
For example, 407,454 shared-ride indicators are available out of 411,437 occurrences.

![example_yellow_num_shared_ride](../../../../../../assets/images/example_yellow_num_shared_ride.png)

These instances will be removed from the dataframes.

```
#drop the empty columns
time_filtered_nyc_yellow_df = time_filtered_nyc_yellow_df.drop(['pickup_longitude', 'pickup_latitude', 
                                                      'dropoff_longitude', 'dropoff_latitude'], axis=1)
time_filtered_nyc_green_df = time_filtered_nyc_green_df.drop(['pickup_longitude', 'pickup_latitude', 
                                                      'dropoff_longitude', 'dropoff_latitude'], axis=1)
#filter non-empty shared_ride indicator
time_filtered_nyc_yellow_df = time_filtered_nyc_yellow_df[time_filtered_nyc_yellow_df['shared_ride'].notnull()]
time_filtered_nyc_green_df = time_filtered_nyc_green_df[time_filtered_nyc_green_df['shared_ride'].notnull()]
```

After cleaning the dataframes, there are 407,454 and 184.370 data points for yellow and green taxis, respectively while 2,214,027 data points for the rideshare transports.

![remaining_data](../../../../../../assets/images/remaining_data.png)

Then, to make a ride-sharing ratio, make an occurrence to 1 to count all occurrences of the rides on rush hours by each transportation mode later.

```
time_filtered_nyc_yellow_df['occurence'] = 1
time_filtered_nyc_green_df['occurence'] = 1
time_filtered_rideshare_df['occurence'] = 1
``` 

## Explore behavioral patterns of rideshares and taxis
---

First, I created the shared ride ratio values by each transport mode.

```
#create ride-sharing ratio for rideshares such as Uber and Lyft
top10_rideshare_df = time_filtered_rideshare_df[['pickup_location_id', 'shared_ride','occurence']]\
                    .groupby(['pickup_location_id'])\
                    .sum()\
                    .reset_index()\
                    .sort_values(['occurence'], ascending=False)
top10_rideshare_df['shared_ride_ratio'] = round((top10_rideshare_df['shared_ride']/top10_rideshare_df['occurence'])*100, 2)

#create ride-sharing ratio for yellow taxis
top10_yellow_df = time_filtered_nyc_yellow_df[['pickup_location_id', 'shared_ride','occurence']]\
                    .groupby(['pickup_location_id'])\
                    .sum()\
                    .reset_index()\
                    .sort_values(['occurence'], ascending=False)
top10_yellow_df['shared_ride_ratio'] = round((top10_yellow_df['shared_ride']/top10_yellow_df['occurence'])*100, 2)

#create ride-sharing ratio for green taxis
top10_green_df = time_filtered_nyc_green_df[['pickup_location_id', 'shared_ride','occurence']]\
                    .groupby(['pickup_location_id'])\
                    .sum()\
                    .reset_index()\
                    .sort_values(['occurence'], ascending=False)
top10_green_df['shared_ride_ratio'] = round((top10_green_df['shared_ride']/top10_green_df['occurence'])*100, 2)
```

Now, let us look at which boroughs and zones are having better or worse ride-sharing ratios. To do so, I need to join the prepared dataframes with the tlc zone dataframe, like below.

```
top10_rideshare_df_loc = top10_rideshare_df.join(tlc_df.set_index('location_id')[['borough', 'zone', 'service_zone', 'nta_code']], on='pickup_location_id')
top10_yellow_df_loc = top10_yellow_df.join(tlc_df.set_index('location_id')[['borough', 'zone', 'service_zone', 'nta_code']], on='pickup_location_id')
top10_green_df_loc = top10_green_df.join(tlc_df.set_index('location_id')[['borough', 'zone', 'service_zone', 'nta_code']], on='pickup_location_id')
```

For the purpose of the analysis, I look at the instances of more than 10,000 occurrences of pick ups, look at the top 50 ratio of shared ride across rideshares and taxis.
Three findings arise from looking at the ride-sharing ratio out of its occurrences (i.e. how much percentage of the ride-sharing occurs during rush hours) by each transport mode.

1) When examining only ride areas with more_than_10k_occurences, 102 data points are available. However, this number is dominated by rideshares (85 out of 102), not taxis.
It seems that between July 2017 and July 2019, rideshares were popular during rush hours. The other interpretation can be the data for taxis may not be complete so less number of occurrences could have been recorded.

![more_than_10k_occurences](../../../../../../assets/images/more_than_10k_occurences.png)

2) Shared ratios for green taxis seems to be lowest, consistently between 12.81% and 21.07%. However, the interpretation here is quite difficult to make because the comparison between rideshare and two taxis is not a fair one. The only clear points here are: where there is one passenger only, these rides are not shared; and the ride-sharing ratio of yellow taxis seems to be higher than that of green taxis.

![violin_chart_mode](../../../../../../assets/images/violin_chart_mode.png)

3) The average shared ratios for Brooklyn seems to be higher than those of Manhattan and Queens. For rush hours, not much data is available for the shared ratios for Bronx.

![violin_chart_borough](../../../../../../assets/images/violin_chart_borough.png)  

Let us now look at problem areas - areas that contain large occurrences during rush-hours, but having a low sharing ratio. To investigate this, bottom 20 pick up areas are selected based on its ratio and plot these areas with the occurrences. If the size of scatter dots are larger, the occurrences of that particular pick up location are greater.

![bottom_20](../../../../../../assets/images/bottom_20.png)

It is clear that the highest occurrences locations for the worst sharing ratio encompasses: 1) TriBeCa/Civic Center, 2) Times Sq/Theatre District, 3) SoHo, 4) JFK Airport, and 5) LaGuardia Airport. During rush hours of July 2017 to July 2019, the occurrences show more than 14,000 trips from these locations but the sharing ratio is around 10-18% only via rideshares.

The last comparison analysis is to visualize the ride-sharing ratio by its percentage and its source. The code below is using folium map with CartoDB dark_matter to visualize the ratio with the size of circles.

```
#util function for deciding the size of the circles to represent shared ratio
def get_size(shared_percent):
    if shared_percent >= 27:
        size = 10
    elif shared_percent >= 25 and shared_percent < 27:
        size = 8
    elif shared_percent >= 23 and shared_percent < 25:
        size = 6
    elif shared_percent >= 21 and shared_percent < 23: 
        size = 4
    elif shared_percent < 21:
        size = 2
    return size

#using folium map, visualize apprximate longitude and attitude of tlc zones for ride occurrences greater than 10,000 
nyc_map = folium.Map(location=[40.693943, -73.985880],
                        zoom_start=10,
                        tiles="CartoDB dark_matter")
for idx, row in all_df_10k_loc.iterrows():
    
    if row["longitude"] and row["latitude"]:

        shared_percent = row["shared_ride_ratio"]
        if row['source']=='rideshare':
            color = "#736AFF" #purple
        elif row['source']=='green_taxi':
            color = "#00CC66" #yellow
        elif row['source']=='yellow_taxi':
            color = "#CCCC00" #green
        
        size = get_size(shared_percent)

        popup_text = "Source: {}<br>Zone: {}<br>Shared_ratio: {}<br>"
        popup_text = popup_text.format(row["source"],
                      row["zone"], str(shared_percent)+'%')

        folium.CircleMarker(location=(row["latitude"],
                                      row["longitude"]),
                            radius=size,
                            color=color,
                            popup=folium.Popup(popup_text, parse_html=False),
                                fill=True).add_to(nyc_map)

nyc_map
```

With the code above, the ride sharing ratios are represented on the New York City's map 1) by the different size of circles (the larger circles represent larger ride-sharing ratios); and 2) by the different transport mode (purple: rideshares, yellow: yellow taxis, green: green taxis). This map is interactive and the source of ride, zone and its sharing ratio can be seen by clicking a particular circle on the map.

![map_over_10k](../../../../../../assets/images/map_over_10k.png)


## Conclusion and limitation of this analysis
---

After going through the analysis, I found that rideshares may not have been used to deliver 'true' ride-sharing in certain areas during rush hours. Overall, increase in ‘true’ ride-sharing may be required to make NYC’s rides more efficient during rush hours since a large disparity from ride-sharing ratios is observed between the areas during rush hours (e.g. Lenox Hill East: 40.04% vs JFK Airport: 11.42%).

It is noted that ride-sharing ratio cannot be compared fairly across different transportation methods currently due to a lack of information in the datasets of Yellow taxi and Green taxi. To prepare a fair analysis for a better traffic situation in NYC, it would be helpful to record a ride-share indicator for taxis.