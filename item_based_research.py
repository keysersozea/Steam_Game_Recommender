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
from Index_Model import Index_Model

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
pd.set_option('display.max_rows', 500)


def read():
    general_df = pd.read_csv("dataset/lastDfs/general_df.csv")
    products = pd.read_csv("dataset/lastDfs/products.csv")
    users = pd.read_csv("dataset/lastDfs/users.csv")
    return general_df, products, users
general_df, products, users = read()

def take_user(df):
    # games played more than 2000 user
    play_count_user = pd.DataFrame(df["user_id"].value_counts())
    valid_users = play_count_user[play_count_user["count"] > 10].index.tolist()
    random_user = random.sample(valid_users, 1)
    return random_user[0]
random_user = take_user(general_df)

def replace_thresholds(dataframe, variable, up_limit=200):
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit # Üst thresholddan büyük değerleri thresholda eşitler
replace_thresholds(general_df, "hours")

def log_transform(df):
    df["log_hours"] = np.log1p(df[["hours"]])
log_transform(general_df)

def weight_with_monthsago(df, weighting_factor=0.05):
    # Implementing inverse weighting on the dataframe
    # Adding a new column for the weighting factor
    df.loc[:, 'weight'] = 1 / (1 + weighting_factor * df['months_ago'])
    # Applying the weight to the hours column to create a new weighted hours column
    df.loc[:, 'time_supported_hours'] = df.loc[:, 'log_hours'] * df.loc[:, 'weight']
weight_with_monthsago(general_df)

def create_playedGames_df(df, user):
    played_games_df = ((df[df["user_id"] == user][["product_id", 'time_supported_hours']]).
                       merge(products[["product_id", "app_name", "tags", "specs"]],
                             on="product_id", how="left"))
    played_games_df = played_games_df.sort_values(by="time_supported_hours", axis=0, ascending=False)
    return played_games_df.head(10)
played_games_df = create_playedGames_df(general_df, random_user)
played_games = played_games_df["product_id"].tolist()
played_games_names = played_games_df["app_name"]

def find_closest_games_w_NLP(played_games, products, model=Index_Model):
    closest_games = []
    for game in played_games:
        game_title = products.loc[products["product_id"] == game, "app_name"].iloc[0]
        nlp_based_similars = list(model(game_title)[0])
        similar_game = products.loc[nlp_based_similars, "product_id"].reset_index(drop=True).head(20).tolist()
        closest_games.extend(similar_game)
        closest_games = list(set(closest_games))
    return closest_games
closest_games = find_closest_games_w_NLP(played_games, products)


def findUser_SameMovies(df, user, played_games, game_count=3):
    # Filter the DataFrame for rows where the game is in gameList
    filtered_df = df[df['product_id'].isin(played_games)]
    user_game_counts = filtered_df.groupby('user_id')['product_id'].nunique()
    User_SameMovies = user_game_counts[user_game_counts >= game_count].index.tolist()
    User_SameMovies.append(user)
    return User_SameMovies
Users_SameMovies = findUser_SameMovies(general_df, random_user, played_games, game_count=1)



def create_item_pivot_table(df, Users_SameMovies, played_games):
    user_pivot_table = df.pivot_table(index=["user_id"], columns=["product_id"], values="time_supported_hours")
    user_pivot_table = user_pivot_table.loc[user_pivot_table.index.isin(Users_SameMovies)]
    #user_pivot_table = user_pivot_table[played_games]
    user_pivot_table = user_pivot_table.fillna(0)
    return user_pivot_table
user_pivot_table = create_item_pivot_table(general_df, Users_SameMovies=Users_SameMovies, played_games=played_games)


def create_similarity_matrix_item(user_pivot_table):
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(user_pivot_table.T)

    # Convert the similarity matrix to a DataFrame for easier handling
    similarity_df = pd.DataFrame(similarity_matrix, index=user_pivot_table.columns, columns=user_pivot_table.columns)
    return similarity_df

similarity_matrix = create_similarity_matrix_item(user_pivot_table)

def find_top_5_similar_games(game_id, similarity_matrix):
    # Get the similarity scores for the given game and sort them
    similar_games = similarity_matrix[game_id].sort_values(ascending=False)

    # Find the top 5 most similar games, excluding the game itself
    top_5_similar = similar_games.index[similar_games.index != game_id][0:5]
    return top_5_similar

def create_similar_games_df(similarity_matrix, played_games_df, weight_similarity=0.7):
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
similar_games_df = create_similar_games_df(similarity_matrix, played_games_df, weight_similarity=0.97)

similar_games_df["Rating"] = similar_games_df[similar_games_df["product_id"].isin(closest_games)]["Rating"] * 1.3

def finalize_item_based_recommendation(similar_games_df, products):
    similar_games_df1 = similar_games_df.drop(["product_id", 'Similarity Ratio', 'time_supported_hours'], axis=1)
    similar_games_df1.rename(columns={"Similar Game ID": "product_id"}, inplace=True)
    item_based_recommendations = similar_games_df1.merge(products[["product_id", "app_name"]],
                                              on="product_id", how="left")
    item_based_recommendations = item_based_recommendations.drop_duplicates(subset=['product_id'])
    MMScaler = MinMaxScaler(feature_range=(0, 10))
    item_based_recommendations["rating"] = MMScaler.fit_transform(item_based_recommendations[["rating"]])
    return item_based_recommendations
item_based_recommendations = finalize_item_based_recommendation(similar_games_df, products)





