import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide")

st.title('Game Publication Analysis')

# --- Data Loading and Preprocessing ---
# Loading the dataset
# Assuming 'game_info_cleaned.csv' is available in the same directory as the app.py file
try:
    df_am12_bb20_rj43 = pd.read_csv('game_info_cleaned.csv', low_memory=False) # Added low_memory=False
except FileNotFoundError:
    st.error("Error: 'game_info_cleaned.csv' not found. Please make sure the file is in the same directory.")
    st.stop()

# Handle the DtypeWarning by specifying dtype for problematic column(s) if known,
# or by using low_memory=False (already used in previous execution)
# For simplicity here, we'll proceed with the loaded data assuming it's handled.

# Drop specified columns
df_am12_bb20_rj43_dropped = df_am12_bb20_rj43.drop([
    'slug', 'tba', 'updated', 'playtime', 'website', 'suggestions_count',
    'game_series_count', 'achievements_count', 'added_status_yet',
    'added_status_owned', 'added_status_beaten', 'added_status_toplay',
    'added_status_dropped','added_status_playing'
], axis=1)

# Drop rows with any null values after dropping columns
df_am12_bb20_rj43_cleaned = df_am12_bb20_rj43_dropped.dropna().copy() # Use .copy() to avoid SettingWithCopyWarning

# --- Feature Engineering ---
# 1. Create 'release_year' column
df_am12_bb20_rj43_cleaned['released'] = pd.to_datetime(df_am12_bb20_rj43_cleaned['released'], errors='coerce')
df_am12_bb20_rj43_cleaned['release_year'] = df_am12_bb20_rj43_cleaned['released'].dt.year.fillna(0).astype(int)

# 2. Create 'game_age' column
df_am12_bb20_rj43_cleaned['game_age'] = 2025 - df_am12_bb20_rj43_cleaned['release_year']
df_am12_bb20_rj43_cleaned.loc[df_am12_bb20_rj43_cleaned['release_year'] == 0, 'game_age'] = np.nan

# 3. Create 'metacritic_category' column
metacritic_bins = [0, 50, 80, 100]
metacritic_labels = ['low', 'medium', 'high']
df_am12_bb20_rj43_cleaned['metacritic_category'] = pd.cut(df_am12_bb20_rj43_cleaned['metacritic'], bins=metacritic_bins, labels=metacritic_labels, right=False)

# 4. Create 'ratings_category' column
ratings_bins = [0, 2, 4, 5.1]
ratings_labels = ['low', 'medium', 'high']
df_am12_bb20_rj43_cleaned['ratings_category'] = pd.cut(df_am12_bb20_rj43_cleaned['rating'], bins=ratings_bins, labels=ratings_labels, right=False)

# 5. Create 'genre_count' and 'genre_count_category' columns
df_am12_bb20_rj43_cleaned['genre_count'] = df_am12_bb20_rj43_cleaned['genres'].str.split('\|\|').str.len()
df_am12_bb20_rj43_cleaned['genre_count_category'] = df_am12_bb20_rj43_cleaned['genre_count'].apply(lambda x: 'single' if x == 1 else 'multiple')


# --- Query 1: Most Highly Rated Game ---
st.markdown("### Query 1: Most Highly Rated Game")

# Find the game with the highest metacritic score
df_am12_bb20_rj43_highest_metacritic_game = df_am12_bb20_rj43_cleaned.loc[df_am12_bb20_rj43_cleaned['metacritic'].idxmax()]
st.write("Game with the highest Metacritic score:")
st.dataframe(df_am12_bb20_rj43_highest_metacritic_game[['name', 'metacritic','release_year','publishers']].to_frame().T)

# Find the game with the highest rating
df_am12_bb20_rj43_highest_rating_game = df_am12_bb20_rj43_cleaned.loc[df_am12_bb20_rj43_cleaned['rating'].idxmax()]
st.write("Game with the highest rating:")
st.dataframe(df_am12_bb20_rj43_highest_rating_game[['name', 'rating','release_year','publishers']].to_frame().T)

# Plot comparing Metacritic and Audience Ratings for Top Games
st.markdown("#### Metacritic Score vs. Rating for Top Games by Metacritic")
top_games_metacritic = df_am12_bb20_rj43_cleaned.nlargest(10, 'metacritic').reset_index(drop=True)

fig1 = px.bar(top_games_metacritic, x='name', y=['metacritic', 'rating'],
              labels={'name': 'Game Name', 'value': 'Score'},
              title='Metacritic Score vs. Rating for Top Games by Metacritic')
fig1.update_layout(barmode='group')
st.plotly_chart(fig1, use_container_width=True)


# --- Query 2: Games Loved by Critics and Audiences ---
st.markdown("### Query 2: Games Loved by Critics and Audiences")

# Filter for games with high metacritic_category and high ratings_category
df_am12_bb20_rj43_loved_games = df_am12_bb20_rj43_cleaned[(df_am12_bb20_rj43_cleaned['metacritic_category'] == 'high') & (df_am12_bb20_rj43_cleaned['ratings_category'] == 'high')]

st.write("Games loved by both critics and audiences:")
st.dataframe(df_am12_bb20_rj43_loved_games[['name','metacritic','rating']])


# --- Query 3: Publishers with the Highest Number of Games ---
st.markdown("### Query 3: Publishers with the Highest Number of Games Published")

# Count the number of games per publisher
df_am12_bb20_rj43_publisher_counts = df_am12_bb20_rj43_cleaned['publishers'].value_counts()

st.write("Publishers with the highest number of games published:")
st.dataframe(df_am12_bb20_rj43_publisher_counts.head(10).to_frame()) # Displaying the top 10 publishers

# Create a bar chart for top publishers
top_n = 10
top_publishers = df_am12_bb20_rj43_publisher_counts.head(top_n)

fig2 = px.bar(top_publishers, x=top_publishers.index, y=top_publishers.values,
              labels={'x': 'Publisher', 'y': 'Number of Games'},
              title=f'Top {top_n} Publishers by Number of Games Published')
st.plotly_chart(fig2, use_container_width=True)


# --- Query 4: Number of Games Published Each Year ---
st.markdown("### Query 4: Number of Games Published Each Year")

# Count the number of games published each year
games_per_year = df_am12_bb20_rj43_cleaned['release_year'].value_counts().sort_index()

# Filter out the year 0, which represents missing release dates
games_per_year = games_per_year[games_per_year.index != 0]

st.write("Number of games published every year:")
st.dataframe(games_per_year.to_frame())

# Create a line plot for games per year
fig3 = px.line(x=games_per_year.index, y=games_per_year.values,
               labels={'x': 'Year', 'y': 'Number of Games'},
               title='Number of Games Published Each Year')
fig3.update_traces(mode='lines+markers')
st.plotly_chart(fig3, use_container_width=True)


# --- Query 5: Average Metacritic and Audience Ratings Over the Years ---
st.markdown("### Query 5: Average Metacritic and Audience Ratings Over the Years")

# Group by release year and calculate mean ratings
average_ratings_per_year = df_am12_bb20_rj43_cleaned.groupby('release_year')[['metacritic', 'rating']].mean().reset_index()

# Filter out the year 0
average_ratings_per_year = average_ratings_per_year[average_ratings_per_year['release_year'] != 0]

st.write("Average Metacritic and Audience ratings over the years:")
st.dataframe(average_ratings_per_year)

# Create a line plot for average ratings over the years
fig4 = px.line(average_ratings_per_year, x='release_year', y=['metacritic', 'rating'],
               labels={'release_year': 'Year', 'value': 'Average Score'},
               title='Average Metacritic and Audience Ratings Over the Years')
fig4.update_traces(mode='lines+markers')
st.plotly_chart(fig4, use_container_width=True)
