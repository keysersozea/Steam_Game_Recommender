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

users[users["user_id"] == random_user]

def create_playedGames_df(df, user):
    played_games_df = ((df[df["user_id"] == user][["product_id"]]).
                       merge(products[["product_id", "app_name", "tags", "specs"]],
                             on="product_id", how="left"))
    return played_games_df
played_games_df = create_playedGames_df(general_df, random_user)
played_games = played_games_df["product_id"].tolist()
played_games_names = played_games_df["app_name"]

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

def findUser_SameMovies(df, user, played_games, game_count=3):
    # Filter the DataFrame for rows where the game is in gameList
    filtered_df = df[df['product_id'].isin(played_games)]
    user_game_counts = filtered_df.groupby('user_id')['product_id'].nunique()
    User_SameMovies = user_game_counts[user_game_counts >= game_count].index.tolist()
    User_SameMovies.append(user)
    return User_SameMovies
Users_SameMovies = findUser_SameMovies(general_df, random_user, played_games)

def create_user_pivot_table(df, Users_SameMovies, played_games):
    user_pivot_table = df.pivot_table(index=["user_id"], columns=["product_id"], values="time_supported_hours")
    user_pivot_table = user_pivot_table.loc[user_pivot_table.index.isin(Users_SameMovies)]
    user_pivot_table = user_pivot_table[played_games]
    user_pivot_table = user_pivot_table.fillna(0)
    return user_pivot_table
user_pivot_table = create_user_pivot_table(general_df, Users_SameMovies=Users_SameMovies, played_games=played_games)

def create_similarity_matrix(pivot_table):
    SS = StandardScaler()
    # Normalize weighted hours to obtain scores
    normalized_array = SS.fit_transform(pivot_table)

    normalized_table = pd.DataFrame(normalized_array, index=pivot_table.index)

    # Obtaain similarity matrix
    similarity_matrix = pd.DataFrame(cosine_similarity(normalized_table),
                                     index=normalized_table.index,
                                     columns=normalized_table.index)
    return similarity_matrix
similarity_matrix = create_similarity_matrix(user_pivot_table)

def create_similarUser_df(similarity_matrix, user, threshold=0.01):
    user_similarity_scores = similarity_matrix.loc[user]
    similar_user_df = pd.DataFrame(user_similarity_scores).reset_index()
    similar_user_df.columns = ['user_id', 'similarity']
    similar_user_1 = similar_user_df[similar_user_df['user_id'] != user]
    similar_user_filtered = similar_user_1[similar_user_1['similarity'] > threshold]
    return similar_user_filtered
similar_user_df = create_similarUser_df(similarity_matrix, random_user, threshold=0.1)

def create_recommendation_df(similar_user_df, general_df, products):
    recommendation_df = similar_user_df.merge(general_df[["user_id", "product_id", "time_supported_hours"]], how="left", on="user_id")
    recommendation_df = recommendation_df.merge(products[["product_id", "tags", "specs"]], on="product_id", how="left")
    return recommendation_df
recommendation_df = create_recommendation_df(similar_user_df, general_df, products)

def find_common_tags(played_games_df, threshold=0.75):
    # Safely evaluate string representations of lists
    played_games_df['tags'] = played_games_df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Explode the 'tags' column
    exploded_tags = played_games_df.explode('tags')

    # Count occurrences of each tag
    spec_counts = exploded_tags['tags'].value_counts()

    # Find common tags based on threshold
    common_tags = spec_counts[spec_counts > (spec_counts.max() * threshold)].index.tolist()

    return common_tags
common_tags = find_common_tags(played_games_df, 0.6)

def find_common_specs(played_games_df, threshold=0.75):
    # Safely evaluate string representations of lists
    played_games_df['specs'] = played_games_df['specs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Explode the 'specs' column
    exploded_specs = played_games_df.explode('specs')

    # Count occurrences of each tag
    spec_counts = exploded_specs['specs'].value_counts()

    # Find common specs based on threshold
    common_specs = spec_counts[spec_counts > (spec_counts.max() * threshold)].index.tolist()

    return common_specs
common_specs = find_common_specs(played_games_df, 0.7)

def scale_w_tags(rec_df, common_tags, base_scale=1.05):
    # 'tags' sütununun NaN olup olmadığını kontrol et
    if pd.isna(rec_df['tags']):
        game_tags = []
    else:
        # String ifadeyi gerçek bir listeye dönüştür
        game_tags = ast.literal_eval(rec_df['tags'])

    # Ortak tag sayısını say
    common_count = sum(tag in common_tags for tag in game_tags)

    # Ortak tag sayısına göre ölçeklendirme faktörünü hesapla
    # Örneğin, base_scale'in karesi ile çarparak (burada 1.05 varsayılan değer)
    scale_factor = base_scale ** common_count

    # Sonuç değerini hesapla
    rec_df["tags_supported_hours"] = rec_df['time_supported_hours'] * scale_factor
    return rec_df
recommendation_df = recommendation_df.apply(lambda rec_df: scale_w_tags(rec_df, common_tags), axis=1)

def scale_w_specs(rec_df, common_specs):
    # Check if 'tags' column is NaN or not
    if pd.isna(rec_df['specs']):
        game_specs = []
    else:
        # Convert the string representation of the list to an actual list
        game_specs = ast.literal_eval(rec_df['specs'])

    # Count the number of common specs
    common_count = sum(tag in common_specs for tag in game_specs)

    # Determine the scaling factor based on the count of common specs
    if common_count >= 3:
        scale_factor = 1.15
    elif common_count == 2:
        scale_factor = 1.1
    elif common_count == 1:
        scale_factor = 1.05
    else:
        scale_factor = 1.0  # No scaling if no common specs
    rec_df["supported_hours"] = rec_df['tags_supported_hours'] * scale_factor
    return rec_df
recommendation_df = recommendation_df.apply(lambda rec_df: scale_w_specs(rec_df, common_specs), axis=1)

def scale_w_similarity(rec_df, exponent=1):
    # Adjust with Similarity
    # exponent: Adjust the exponent to control the influence
    rec_df["rating"] = (rec_df["similarity"] ** exponent) * rec_df["supported_hours"]
    return rec_df
    #  ** exponent)
recommendation_df = scale_w_similarity(recommendation_df)


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
recommendation_df["rating"] = recommendation_df[recommendation_df["product_id"].isin(closest_games)]["rating"] * 1.3


def finalize_recommendation_df(rec_df, played_games, num_games=30):
    user_based_rec_df = (rec_df.sort_values(by="rating", ascending=False))[["product_id", "rating"]].reset_index(drop=True)
    user_based_rec_df = user_based_rec_df[~user_based_rec_df["product_id"].isin(played_games)]
    user_based_rec_df = user_based_rec_df.drop_duplicates(subset="product_id")
    user_based_rec_df = user_based_rec_df.merge(products, on="product_id", how="left")
    MMScaler = MinMaxScaler(feature_range=(0, 10))
    user_based_rec_df["rating"] = MMScaler.fit_transform(user_based_rec_df[["rating"]])
    user_based_rec_df = user_based_rec_df[["product_id", "rating", "app_name"]].head(num_games)
    user_based_rec_df = user_based_rec_df.head(num_games)
    return user_based_rec_df
user_based_rec_df = finalize_recommendation_df(recommendation_df, played_games)

def suggest_user_based(user_based_rec_df, num_games=10, exponent=1.5, save=False):

    # Normalize the ratings to get probabilities
    # USe exponent to give priority to games with highest rating
    user_based_rec_df["exp_rating"] = user_based_rec_df['rating'] ** exponent
    total_rating = user_based_rec_df['exp_rating'].sum()
    user_based_rec_df['probability'] = user_based_rec_df['rating'] / total_rating

    # Select games randomly based on the rating probabilities
    selected_games = user_based_rec_df.sample(n=num_games, weights='probability', replace=False)
    print(selected_games["app_name"].reset_index(drop=True))
    if save == True:
        return selected_games
suggest_user_based(user_based_rec_df, exponent=1)


def user_based_suggestions(user, df):
    played_games_df = create_playedGames_df(df, user)
    played_games = played_games_df["product_id"].tolist()
    played_games_names = played_games_df["app_name"]
    replace_thresholds(df, "hours")
    log_transform(df)
    weight_with_monthsago(df)
    Users_SameMovies = findUser_SameMovies(df, user, played_games)
    user_pivot_table = create_user_pivot_table(df, Users_SameMovies=Users_SameMovies, played_games=played_games)
    similarity_matrix = create_similarity_matrix(user_pivot_table)
    similar_user_df = create_similarUser_df(similarity_matrix, user, threshold=0.1)
    recommendation_df = create_recommendation_df(similar_user_df, general_df, products)
    common_tags = find_common_tags(played_games_df, 0.7)
    common_specs = find_common_specs(played_games_df, 0.8)
    recommendation_df_1 = recommendation_df.apply(lambda rec_df: scale_w_tags(rec_df, common_tags), axis=1)
    recommendation_df_2 = recommendation_df_1.apply(lambda rec_df: scale_w_specs(rec_df, common_specs), axis=1)
    recommendation_df_3 = scale_w_similarity(recommendation_df_2, exponent=1.5)
    user_based_rec_df = finalize_recommendation_df(recommendation_df_3, played_games)
    print(user_based_rec_df)
    print(played_games_df)

random_user = take_user(general_df)
user_based_suggestions(random_user, general_df)
print(played_games_names)







