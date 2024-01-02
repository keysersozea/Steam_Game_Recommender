import pandas as pd
import numpy as np
import datetime as dt
from Useful_Functions import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.0f' % x)

games = pd.read_csv("dataset/starting/games_data.csv")
bundle = pd.read_csv("dataset/starting/bundle_data.csv")
reviews = pd.read_csv("dataset/starting/reviews_data.csv")


### games preprocessins ###
games.head()
games.rename(columns={"id": "product_id"}, inplace=True)

#### reviews processes ###
reviews.head()
check_df(reviews)
comman = reviews[reviews["user_id"].notnull()].drop("Unnamed: 0", axis=1).reset_index(drop=True)

comman.drop(axis=1, columns=["page", "page_order", "compensation", "products"], inplace=True)
comman.drop(axis=1, columns="found_funny", inplace=True)    # %95 of data is 0
comman.drop(axis=1, columns="early_access", inplace=True)    # not useful
comman["text"].dropna(inplace=True)
check_df(comman)
comman.dropna(inplace=True)

comman["username"].nunique()
comman["date"] = pd.to_datetime(comman["date"])
today = comman["date"].max()
comman["months_ago"] = round((today - comman["date"]).dt.days / 30)
comman.drop("date", axis=True, inplace=True)
comman["username"].value_counts()


### New DF's ###

users = pd.DataFrame(comman.set_index("user_id")["username"])
users = users.drop_duplicates()
users.reset_index(inplace=True)
filtered_users = users.groupby("user_id").agg({"username": "count"}).sort_values(by="username")
userList = filtered_users[~(filtered_users["username"] > 1)].index.tolist()

products = comman[["product_id"]]
products = products.merge(games[["product_id", "app_name", "tags", "specs"]], how="left", on="product_id")
products = products.drop_duplicates(subset="product_id")
productList = products["product_id"].tolist()

general_df = comman[["user_id", "product_id", "hours", "months_ago"]]
general_df = general_df.sort_values(by=['user_id', 'product_id']).reset_index(drop=True)

comments = comman[["user_id", "product_id", "text"]]
comments = comments.drop_duplicates(subset=["user_id", "product_id", "text"]).reset_index(drop=True)

users = users[users["user_id"].isin(userList)]
general_df = general_df[general_df["user_id"].isin(userList)]
comments = comments[comments["user_id"].isin(userList)]

general_df = general_df[general_df["product_id"].isin(productList)]
comments = comments[comments["product_id"].isin(productList)]

# 10den az oyun oynamış insanları çıkar
# 100'dan az kişinin oynadığı oyunları çıkar

def create_valid_df(df, product_count_th=100, user_count_th=10):
    play_count_product = pd.DataFrame(df["product_id"].value_counts())
    # games played more than 2000 user
    valid_games = play_count_product[play_count_product["count"] > product_count_th].index.tolist()

    play_count_user = pd.DataFrame(df["user_id"].value_counts())
    # users playing more than 10 games
    valid_users = play_count_user[play_count_user["count"] > user_count_th].index.tolist()

    # upgrade with valid games
    general_df = df[df["product_id"].isin(valid_games)]
    # upgrade with valid users
    general_df = general_df[general_df["user_id"].isin(valid_users)]
    general_df = general_df.drop_duplicates(subset=['user_id', 'product_id'])
    return general_df, valid_users, valid_games

general_df, valid_users, valid_games = create_valid_df(general_df)
users = users[users["user_id"].isin(valid_users)]
products = products[products["product_id"].isin(valid_games)]
comments = comments[comments["product_id"].isin(valid_games)]

users.to_csv("users.csv", index=False)
products.to_csv("products.csv", index=False)
comments.to_csv('comments.csv', index=False)
general_df.to_csv("general_df.csv", index=False)






