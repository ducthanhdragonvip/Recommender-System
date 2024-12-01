import re

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

def weighted_rating(df):
    """
    Calculates the IMDB's Weighted Rating using the following formula:
        (v / (v+m) * R) + (m / (m+v) * C)

    where:
    - v is the number of votes for the movie;
    - m is the minimum votes required to be listed in the chart;
    - R is the average rating of the movie; And
    - C is the mean vote across the whole report
    """
    v = df['count_rating']
    m = df['minimum_no_of_ratings']
    R = df['avg_rating']
    C = df['overall_avg_rating']

    return (v / (v+m) * R) + (m / (m+v) * C)

def filtered_dataset(genre, ratings_movies_df):
    """Filters list of movies to show only movies of the chosen genre. Calculates
    the 95th quantile of No. of Ratings to keep only relevant movies.

    Args:
        genre (str): Genre name
        ratings_movies_df (pd.DataFrame): Dataframe with Movie Ratings

    Returns:
        genre_df[pd.DataFrame]: Filtered Dataframe with relevant movies for the genre
    """
    # Keep only the selected genre
    genre_df = ratings_movies_df[ratings_movies_df['genres'].str.contains(genre)]

    # Calculate the 95th quantile and the weighted rating
    minimum_no_of_ratings = genre_df['count_rating'].quantile(0.95)
    genre_df['minimum_no_of_ratings'] = minimum_no_of_ratings
    genre_df['weighted_rating'] = genre_df.apply(weighted_rating, axis=1)

    # Remove movies with not enough ratings
    genre_df = genre_df[genre_df['count_rating'] >= minimum_no_of_ratings]

    # Sorted it by weighted rating so we have the highest ratings on the top
    genre_df = genre_df.sort_values('weighted_rating', ascending=False)
    genre_df = genre_df.reset_index(drop=True)

    # Keep certain relevant columns
    genre_df = genre_df[['movieId', 'title',
                         'genres', 'count_rating',
                         'minimum_no_of_ratings', 'weighted_rating']]

    return genre_df

@st.cache(suppress_st_warning=True, show_spinner=False)
def movie_picture(imdb_id):
    """A function to find a movie's picture from IMDB

    Args:
        movie_name (str): Movie Name

    Returns:
        movie_image_link[str]: IMDB's Link to JPEG Image of the Movie
    """
    # Create IMDB movie link
    new_url = 'https://www.imdb.com/title/tt' + str(imdb_id)

    # Define headers
    HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '\
                         'AppleWebKit/537.36 (KHTML, like Gecko) '\
                         'Chrome/75.0.3770.80 Safari/537.36'}

    # Go Movie's Page
    movie_page = requests.get(new_url, headers=HEADERS)
    soup = BeautifulSoup(movie_page.content, 'html.parser')

    # Movie JPEG link
    movie_image_link = soup.find(class_='ipc-image').attrs['src']

    return movie_image_link

def rename_title(title):
    """Function to rename a title

    Args:
        title (str): Original Title in wrong format

    Returns:
        new_title[str]: Fixed movie title in the correct format
    """
    # Extract year from title
    year = title.strip()[-5:][:-1]

    # Confirms if it's a valid year number
    if len(re.findall("[0-9]{4}", year)) != 1:
        year = np.nan
    if year != np.nan:
        title = title[:-6].strip()

    # Create new title
    new_title = " ".join(title.split(", ")[::-1])

    return new_title
