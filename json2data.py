import ast
import pandas as pd
import json
import gzip

games_json = "dataset/2/steam_games.json.gz"
bundle_json = "dataset/2/bundle_data.json.gz"
reviews_json = "dataset/2/steam_reviews.json.gz"

# Function to convert Python dictionary string to standard JSON string
def convert_to_json_string(py_dict_str):
    try:
        # Convert Python dictionary string to actual dictionary
        dict_obj = ast.literal_eval(py_dict_str)
        # Convert dictionary to JSON string
        json_str = json.dumps(dict_obj)
        return json_str
    except Exception as e:
        # Return None if there's an error
        return None

# Reading and converting the file content
games_data = []; bundle_data = []; reviews_data = []
file = [(games_json, games_data), (bundle_json, bundle_data), (reviews_data, reviews_json)]
for jsonfile, datafile in file:
    with gzip.open(jsonfile, 'rt', encoding='utf-8') as file:
        for line in file:
            json_str = convert_to_json_string(line)
            if json_str:
                datafile.append(json.loads(json_str))


with gzip.open(reviews_json, 'rt', encoding='utf-8') as file:
    for line in file:
        json_str = convert_to_json_string(line)
        if json_str:
            reviews_data.append(json.loads(json_str))

reviews_data = pd.DataFrame(reviews_data)
reviews_data.to_csv("dataset/2/reviews_data.csv")

# Converting the list of dictionaries to a DataFrame
bundle_data = pd.DataFrame(bundle_data)
bundle_data.to_csv("dataset/2/bundle_data.csv")

games_data = pd.DataFrame(games_data)
games_data.to_csv("dataset/2/games_data.csv")


reviews_data["user_id"].isnull().sum()