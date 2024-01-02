import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew
import math
import ast
import random
from Useful_Functions import *
from user_based_functions import findUser_SameMovies
from Index_Model import Index_Model
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.1f' % x)

def find_closest_games_w_NLP(played_games, products, model=Index_Model):
    closest_games = []
    for game in played_games:
        game_title = products.loc[products["product_id"] == game, "app_name"].iloc[0]
        nlp_based_similars = list(model(game_title)[0])
        similar_game = products.loc[nlp_based_similars, "product_id"].reset_index(drop=True).head(20).tolist()
        closest_games.extend(similar_game)
        closest_games = list(set(closest_games))
    return closest_games

def create_item_pivot_table(df, Users_SameMovies):
    user_pivot_table = df.pivot_table(index=["user_id"], columns=["product_id"], values="time_supported_hours")
    user_pivot_table = user_pivot_table.loc[user_pivot_table.index.isin(Users_SameMovies)]
    #user_pivot_table = user_pivot_table[played_games]
    user_pivot_table = user_pivot_table.fillna(0)
    return user_pivot_table
def create_similarity_matrix_item(user_pivot_table):
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(user_pivot_table.T)

    # Convert the similarity matrix to a DataFrame for easier handling
    similarity_df = pd.DataFrame(similarity_matrix, index=user_pivot_table.columns, columns=user_pivot_table.columns)
    return similarity_df
def find_top_5_similar_games(game_id, similarity_matrix):
    # Get the similarity scores for the given game and sort them
    similar_games = similarity_matrix[game_id].sort_values(ascending=False)

    # Find the top 5 most similar games, excluding the game itself
    top_5_similar = similar_games.index[similar_games.index != game_id][0:5]
    return top_5_similar
def create_similar_games_df(similarity_matrix, played_games_df, general_df, weight_similarity=0.7):
    # Dictionary to hold the top 5 similar games for each game
    top_5_similar_games_dict = {}

    # Find the top 5 similar games for each game in the dataframe
    for game in played_games_df["product_id"]:
        top_5_similar_games_dict[game] = find_top_5_similar_games(game, similarity_matrix)

    data_for_df = []
    # Iterating through each game and its top 5 similar games to collect the required information
    for game_id, similar_games in top_5_similar_games_dict.items():
        for similar_game_id in similar_games:
            similarity_ratio = similarity_matrix.loc[game_id, similar_game_id]
            data_for_df.append({'product_id': game_id, 'Similar Game ID': similar_game_id, 'Similarity Ratio': similarity_ratio})

    # Creating the DataFrame
    similar_games_df = pd.DataFrame(data_for_df)
    similar_games_df = similar_games_df.merge(general_df[["product_id", "time_supported_hours"]],
                                              on="product_id", how="left")

    weight_time_supported_hours = 1 - weight_similarity

    # Calculating the combined rating
    similar_games_df['Rating'] = (weight_similarity * similar_games_df['Similarity Ratio'] +
                                         weight_time_supported_hours * similar_games_df['time_supported_hours'])
    similar_games_df = similar_games_df.drop_duplicates(subset=['product_id', 'Similar Game ID'])
    similar_games_df.sort_values(by="Rating", axis=0, ascending=False, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 10))
    # Scale the 'Rating' column
    similar_games_df['Rating'] = scaler.fit_transform(similar_games_df[['Rating']])
    return similar_games_df
    # Display the updated DataFrame with the new 'Rating' column
def finalize_item_based_recommendation(similar_games_df, products, num_games=30):
    similar_games_df.drop(["product_id", 'Similarity Ratio', 'time_supported_hours'], axis=1, inplace=True)
    similar_games_df.rename(columns={"Similar Game ID": "product_id"}, inplace=True)
    item_based_recommendations = similar_games_df.merge(products[["product_id", "app_name"]],
                                              on="product_id", how="left")
    item_based_recommendations = item_based_recommendations.drop_duplicates(subset=['product_id'])
    MMScaler = MinMaxScaler(feature_range=(0, 10))
    item_based_recommendations["Rating"] = MMScaler.fit_transform(item_based_recommendations[["Rating"]])
    item_based_recommendations.reset_index(inplace=True)
    return item_based_recommendations.head(num_games)

def item_based_suggestions(general_df, random_user, played_games_df, products):
    played_games = played_games_df["product_id"].tolist()
    Users_SameMovies = findUser_SameMovies(general_df, random_user, played_games, game_count=1)
    user_pivot_table = create_item_pivot_table(general_df, Users_SameMovies=Users_SameMovies)
    similarity_matrix = create_similarity_matrix_item(user_pivot_table)
    similar_games_df = create_similar_games_df(similarity_matrix, played_games_df, general_df, weight_similarity=0.97)
    closest_games = find_closest_games_w_NLP(played_games, products)
    similar_games_df["Rating"] = similar_games_df[similar_games_df["product_id"].isin(closest_games)]["Rating"] * 1.3
    item_based_recommendations = finalize_item_based_recommendation(similar_games_df, products, num_games=30)
    return item_based_recommendations


