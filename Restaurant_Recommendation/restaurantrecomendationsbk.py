# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:42:26 2024

@author: Selvibala
"""
#step 1 => importing the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# step 2 => loading the dataset
data = pd.read_csv(r'C:\Users\Selvibala\Downloads\Dataset Cognifyz.csv')

# step 3 => handling the missing values
data.fillna("", inplace=True)

# step 4 => converting the numerical columns to strings
data['Cuisines'] = data['Cuisines'].astype(str)
data['Price range'] = data['Price range'].astype(str)

# step 5 => encoding the categorical variables
tfidf = TfidfVectorizer(stop_words='english')
restaurant_matrix = tfidf.fit_transform(data['Cuisines'] + ' ' + data['Price range'])

# step 6 => creating a function to recommend restaurants based on user preferences
def recommend_restaurants(user_preferences, top_n=5):
    # transforming the user preferences into TF-IDF vector
    user_pref_vector = tfidf.transform([user_preferences])
    
    # calculating the cosine similarity between user preferences and restaurant descriptions
    cosine_similarities = linear_kernel(user_pref_vector, restaurant_matrix).flatten()
    
    # getting the indices of top similar restaurants
    top_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    
    # returning the top recommended restaurants
    return data.iloc[top_indices]

# step 7 => making the list of unique cuisines
cuisine_list = data['Cuisines'].unique()
print("List of available cuisines:")
for cuisine in cuisine_list:
    print(cuisine)

# step 8 => asking the user for preference
user_cuisine = input("Enter your Preferred Cuisine from the List above: ")
user_price_range = input("Enter your Preferred Price Range (E.g : Cheap, Moderate, Expensive): ")

# step 9 => combining the user preferences
user_preferences = user_cuisine + ' ' + user_price_range

# step 10 => displaying the recommendations to the user according to their preferences
recommended_restaurants = recommend_restaurants(user_preferences)
print(recommended_restaurants[['Restaurant Name', 'Cuisines', 'Price range']])
