import pandas as pd
from user_based_functions import *
from item_based_functions import *
from final_suggestions import final_suggestions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.1f' % x)

random_user = take_user(general_df)
user_based_recommendations, general_df, played_games_df, products = user_based_suggestions(random_user, general_df)
item_based_recommendations = item_based_suggestions(general_df, random_user, played_games_df, products)
final_recommendations = final_suggestions(user_based_recommendations, item_based_recommendations)

if __name__ == "__main__":
    print("İşlem başladı")
    random_user = take_user(general_df)
    user_based_recommendations, general_df, played_games_df, products = user_based_suggestions(random_user, general_df)
    item_based_recommendations = item_based_suggestions(general_df, random_user, played_games_df, products)
    final_recommendations = final_suggestions(user_based_recommendations, item_based_recommendations)



