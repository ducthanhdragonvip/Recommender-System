import warnings

import numpy as np
import pandas as pd
import streamlit as st
from surprise import Dataset, Reader

from functions import filtered_dataset, movie_picture

warnings.filterwarnings('ignore')

# Import required data
current_user_ratings = pd.read_csv('./app/data/user_movie_ratings.csv')
movie_ratings = pd.read_csv('./app/data/movies_by_rating.csv')
model = pd.read_pickle('./app/data/svd.pkl')['algo']
imdb_links = pd.read_csv('./app/data/links.csv', dtype={'movieId': 'str',
														'imdbId': 'str',
														'tmdbId': 'str'})

# Set default values
n_recommendations = 5
n_of_movies_to_rate = 5
default_user_id = 999999
movie_image_links = ['nolink' for i in range(n_of_movies_to_rate)]

# Set page title
st.set_page_config(page_title="Movies Recommendation", page_icon=':clapper:')
st.write('# What movies to watch next? :clapper:')
st.subheader("Select a genre, rate five movies and I'm going to tell you what to watch next :crystal_ball:")

# Ask first question
st.write('#### :one: Please select your favorite genre')
genres = ['-', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
		  'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
selected_genre = st.selectbox('Genres:', genres)

# If user hasn't chosen a genre, prompt the user to choose one
if selected_genre == '-':
	st.warning('Please choose a genre')
	st.stop()

# Show the selected genre
st.write('Selected genre:', selected_genre)

# Filter dataset given the selected genre
favorite_genre_movies = filtered_dataset(selected_genre, movie_ratings)

# Keep the highest rated movies by keeping the top 20, shuffling and choosing 5
favorite_genre_movies_filtered = favorite_genre_movies.iloc[:20].sample(frac=1, random_state=111)
favorite_genre_movies_filtered = favorite_genre_movies_filtered.iloc[:n_of_movies_to_rate]

# Grab Ratings - Don't let users move on if no ratings
st.write('#### :two: Please rate the following movies from 1 to 5')
ratings_options = ("Haven't watched it yet.", '1', '2', '3', '4', '5')

# Store User Ratings on New Movies
input_ratings = []
submitted = False

# Add form to collect multiple ratings at the same time and load pictures faster
with st.form(key='form'):

	# Create five rows of elements
	cols = st.columns(5)
	for i, col in enumerate(cols):

		# Extract Movie ID and Title
		movie_to_rate_id = favorite_genre_movies_filtered.iloc[i]['movieId']
		movie_to_rate_title = favorite_genre_movies_filtered.iloc[i]['title']
		imdb_id = imdb_links[imdb_links['movieId'].astype(float) == movie_to_rate_id].iloc[0]['imdbId']

		# For each movie create three columns
		row1_0, row1_1, row1_2 = st.columns((1,2,3))

		# The first will contain the image
		with row1_1:
			# If there's no image link then we need to calculate it before showing image
			if movie_image_links[i] == 'nolink':
				# Create the movie image link from IMDB
				movie_img_link = movie_picture(imdb_id)
				movie_image_links[i] = movie_img_link

			st.image(movie_image_links[i], width=150)


		# Second will have the rating radio form
		with row1_2:
			user_movie_rating = st.radio(movie_to_rate_title,
										 ratings_options,
										 key=i)
			imdb_link = 'https://www.imdb.com/title/tt' + imdb_id
			st.markdown(f'[IMDb link]({imdb_link})')
		# Add rating to list
		input_ratings.append(user_movie_rating)

	# Add form submit button
	submitted = st.form_submit_button('Show recommendations!')

# Recommendations will run only after the user submits
if submitted:
	with st.spinner('Wait for it...'):

		# Create empty list to store ratings with more info
		new_ratings = []

		for idx, movie_rating in enumerate(input_ratings):
			if movie_rating != "Haven't watched it yet.":
				new_ratings.append({'userId': default_user_id,
									'movieId': favorite_genre_movies_filtered.iloc[idx]['movieId'],
									'rating': movie_rating})
			else:
				new_ratings.append({'userId': np.nan,
										'movieId': np.nan,
										'rating': np.nan})

		# Append new ratings to existing dataset
		new_ratings_df = pd.DataFrame(new_ratings)
		updated_df = pd.concat([new_ratings_df, current_user_ratings])
		updated_df = updated_df.dropna()

		# Transform data set
		reader = Reader()
		new_data = Dataset.load_from_df(updated_df, reader)
		new_dataset = new_data.build_full_trainset()

		# Retrain the model
		model.fit(new_dataset)

		# Make predictions for the user
		predictions = []
		for movie_id in favorite_genre_movies['movieId'].to_list():
			predicted_score = model.predict(default_user_id, movie_id)[3]
			predictions.append((movie_id, predicted_score))

		# order the predictions from highest to lowest rated
		ranked_movies = pd.DataFrame(predictions, columns=['movieId', 'predicted_score'])
		ranked_movies = ranked_movies[~ranked_movies['movieId'].isin(new_ratings_df['movieId'])]
		ranked_movies = ranked_movies.sort_values('predicted_score', ascending=False).reset_index(drop=True)
		ranked_movies = pd.merge(ranked_movies, movie_ratings, on='movieId')
		ranked_movies = ranked_movies[ranked_movies['movieId'].isin(favorite_genre_movies['movieId'])]

		# Show the recommendations
		st.write('### :three: Here are your recommendations')


		# If there aren't enough movies to recommend then show only what's in there
		if len(ranked_movies) < n_recommendations:
			n_recommendations = len(ranked_movies)

		# For each movie create three columns
		cols = st.columns(5)

		# Show recommendations
		for row in range(n_recommendations):
			movie_id = ranked_movies.iloc[row]['movieId']
			imdb_id = imdb_links[imdb_links['movieId'].astype(float) == movie_id].iloc[0]['imdbId']
			imdb_link = 'https://www.imdb.com/title/tt' + imdb_id
			recommended_title = movie_ratings[movie_ratings['movieId'] == movie_id]['title'].item()

			# The first will contain the image
			with cols[row]:
				# Create the movie image link from IMDB
				movie_img_link = movie_picture(imdb_id)
				st.image(movie_img_link, width=125)
				st.markdown(f'###### #{row+1} [{recommended_title}]({imdb_link})')




		st.success('Done!')
