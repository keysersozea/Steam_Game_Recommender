import pandas as pd
def final_suggestions(user_based_recommendations, item_based_recommendations, boost=1.3):
    common_product_ids = list(set(item_based_recommendations['product_id']).intersection(set(user_based_recommendations['product_id'])))
    common_item_df = item_based_recommendations[item_based_recommendations['product_id'].isin(common_product_ids)]
    common_user_df = user_based_recommendations[user_based_recommendations['product_id'].isin(common_product_ids)]
    common_merged_df = pd.merge(common_item_df, common_user_df, on='product_id', suffixes=('_item', '_user'))
    common_merged_df['average_rating'] = common_merged_df[['Rating', 'rating']].mean(axis=1) * boost
    final_common_df = common_merged_df[['product_id', 'app_name_item', 'average_rating']]
    final_common_df = final_common_df.rename(columns={"app_name_item": "Game_Name", "average_rating": "Rating"})


    non_common_product_ids_item = set(item_based_recommendations['product_id']) - set(common_product_ids)
    non_common_product_ids_user = set(user_based_recommendations['product_id']) - set(common_product_ids)


    non_common_item_df = item_based_recommendations[item_based_recommendations['product_id'].isin(non_common_product_ids_item)]
    non_common_user_df = user_based_recommendations[user_based_recommendations['product_id'].isin(non_common_product_ids_user)]

    # Ortak olmayan oyunların DataFrame'lerini düzenleme ve birleştirme
    non_common_item_df = non_common_item_df[['product_id', 'app_name', 'Rating']].rename(columns={'app_name': 'Game_Name'})
    non_common_user_df = non_common_user_df[['product_id', 'app_name', 'rating']].rename(columns={'app_name': 'Game_Name', 'rating': 'Rating'})

    # Ortak ve ortak olmayan oyunları birleştirme
    final_recommendation = pd.concat([final_common_df, non_common_item_df, non_common_user_df])
    final_recommendation.sort_values(by="Rating", axis=0, ascending=False, inplace=True)
    final_recommendation.reset_index(drop=True, inplace=True)
    print(final_recommendation.head(25))
    return final_recommendation.head(25)
