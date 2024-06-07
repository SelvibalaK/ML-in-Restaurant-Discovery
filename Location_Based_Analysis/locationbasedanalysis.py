# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:08:26 2024

@author: Selvibala
"""
#step 1 => importing the necessary libraries
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# step 2 => loading the dataset
data = pd.read_csv(r'C:\Users\Selvibala\Downloads\Dataset Cognifyz.csv')

# step 3 => visualizing the distribution of restaurants on a map
map_restaurants = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=10)

marker_cluster = MarkerCluster().add_to(map_restaurants)

for idx, row in data.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Restaurant Name']).add_to(marker_cluster)

# step 4 => saving the map
map_restaurants.save('restaurants_map.html')

# step 5 => grouping the restaurants by city or locality
grouped_by_city = data.groupby('City')

# step 6 => analyzing the concentration of restaurants in different areas
restaurant_counts = grouped_by_city.size()

# step 7 => calculating the statistics such as average ratings, cuisines, or price ranges by city or locality
avg_ratings = grouped_by_city['Aggregate rating'].mean()

# step 8 => defining a function to get the most common cuisine or return none if there are no cuisines
def get_common_cuisine(x):
    try:
        return x.value_counts().index[0]
    except IndexError:
        return None

common_cuisines = grouped_by_city['Cuisines'].apply(get_common_cuisine)
avg_price_range = grouped_by_city['Price range'].mean()

# step 9 => identifing the interesting insights or patterns related to the locations of the restaurants

print("Restaurant Counts by City:")
print(restaurant_counts)
print("\nAverage Ratings by City:")
print(avg_ratings)
print("\nMost Common Cuisines by City:")
print(common_cuisines)
print("\nAverage Price Range by City:")
print(avg_price_range)
