import pandas as pd
import numpy as np
import datetime as dt
from Useful_Functions import *
import matplotlib.pyplot as plt
from collections import Counter
import ast

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.0f' % x)

games = pd.read_csv("dataset/starting/games_data.csv")
bundle = pd.read_csv("dataset/starting/bundle_data.csv")
reviews = pd.read_csv("dataset/starting/reviews_data.csv")

def visualize_tag_distribution(dataframe, tag_column, top_n=20, save_path=None):
    """
    Visualizes the distribution of tags in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    tag_column (str): The name of the column containing the tags.
    top_n (int): The number of top tags to display.

    Returns:
    None: The function plots a bar chart of the tag distribution.
    """
    # Handle NaN values and convert the string representation of lists to actual lists
    dataframe[tag_column] = dataframe[tag_column].fillna("[]").apply(ast.literal_eval)

    # Flatten the list of tags into a single list
    all_tags = [tag for sublist in dataframe[tag_column] for tag in sublist]

    # Count the frequency of each tag
    tag_counts = Counter(all_tags)

    # Convert to a DataFrame for easier plotting
    tag_counts_df = pd.DataFrame(tag_counts.items(), columns=['Tag', 'Count']).sort_values('Count', ascending=False)

    # Display the top N tags for visualization
    top_tags = tag_counts_df.head(top_n)
    top_tags.plot(kind='bar', x='Tag', y='Count', figsize=(15, 7), color='skyblue')

    plt.title(f'Top {top_n} Tags in Dataset')
    plt.xlabel('Tag')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

visualize_tag_distribution(games, 'tags', save_path='C:/Users/yilma/PycharmProjects/RecommendationProject/visuals/tag_distribution.png')

def visualize_publisher_representation(dataframe, publisher_column, top_n=20, figsize=(15, 7), bar_color='green', save_path=None):
    """
    Visualizes the number of games published by different publishers in a bar chart format.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    publisher_column (str): The name of the column containing the publishers.
    top_n (int): The number of top publishers to display.
    figsize (tuple): The size of the figure (width, height).
    bar_color (str): The color of the bars in the bar chart.

    Returns:
    None: The function plots a bar chart of the number of games published by different publishers.
    """
    # Count the number of games published by each publisher
    publisher_counts = dataframe[publisher_column].value_counts()

    # Focus on the top N publishers for readability
    top_publishers = publisher_counts.head(top_n)

    # Plotting
    plt.figure(figsize=figsize)
    top_publishers.plot(kind='bar', color=bar_color)

    plt.title(f'Top {top_n} Publishers by Number of Games Published')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Games Published')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

visualize_publisher_representation(dataframe=games, publisher_column='publisher', top_n=20, save_path='C:/Users/yilma/PycharmProjects/RecommendationProject/visuals/publisher_representation.png')


def plot_games_released_last_n_years(dataframe, date_column, last_n_years=5, figsize=(10, 6), bar_color='navy', save_path=None):
    """
    Plots the number of games released in the last N years.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    date_column (str): The name of the column containing the release dates.
    last_n_years (int): The number of years to look back from the latest year in the dataset.
    figsize (tuple): The size of the figure (width, height).
    bar_color (str): The color of the bars in the bar chart.

    Returns:
    None: The function plots a bar chart of the number of games released in the last N years.
    """
    # Convert 'date_column' to datetime and handle NaN values by excluding them
    dataframe[date_column] = pd.to_datetime(dataframe[date_column], errors='coerce')

    # Identifying the latest year in the data
    latest_year = 2017

    # Determine the range for the last N years
    start_year = latest_year - last_n_years + 1

    # Filter the dataset for the last N years
    filtered_data = dataframe[
        (dataframe[date_column].dt.year >= start_year) & (dataframe[date_column].dt.year <= latest_year)]

    # Count the number of games released each year in the last N years
    yearly_counts = filtered_data[date_column].dt.year.value_counts().sort_index()

    # Plotting
    plt.figure(figsize=figsize)
    yearly_counts.plot(kind='bar', color=bar_color)

    plt.title(f'Number of Games Released in the Last {last_n_years} Years ({start_year}-{latest_year})')
    plt.xlabel('Year')
    plt.ylabel('Number of Games Released')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

plot_games_released_last_n_years(dataframe=games, date_column='release_date', last_n_years=5, save_path='C:/Users/yilma/PycharmProjects/RecommendationProject/visuals/games_released_last_n_years.png')

def plot_hours_played_distribution(dataframe, hours_column, bins=50, figsize=(12, 6), color='blue', edgecolor='black', x_limit=1000, save_path=None):
    """
    Plots and optionally saves a histogram showing the distribution of hours played by users, with a specified x-axis limit.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    hours_column (str): The name of the column containing the hours played.
    bins (int): The number of bins for the histogram.
    figsize (tuple): The size of the figure (width, height).
    color (str): The color of the bars in the histogram.
    edgecolor (str): The edge color of the bars in the histogram.
    x_limit (int): The upper limit for the x-axis.
    save_path (str, optional): The file path to save the histogram. If None, the histogram is not saved.

    Returns:
    None: The function plots a histogram and optionally saves it to a file.
    """
    plt.figure(figsize=figsize)
    plt.hist(dataframe[hours_column], bins=bins, color=color, edgecolor=edgecolor, range=(0, x_limit))

    plt.title('Distribution of Hours Played')
    plt.xlabel('Hours')
    plt.ylabel('Number of Users')
    plt.grid(axis='y')
    plt.xlim(0, x_limit)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

plot_hours_played_distribution(dataframe=reviews, hours_column='hours', bins=50, color='darkgreen', edgecolor='black', x_limit=1000, save_path='C:/Users/yilma/PycharmProjects/RecommendationProject/visuals/hour_distributions.png')


def analyze_missing_values(dataframe, column_name, save_path=None):
    """
    Analyzes missing values in a specified column of a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to analyze for missing values.

    Returns:
    None: The function prints the analysis and plots a bar chart.
    """
    total_rows = len(dataframe)
    missing_count = dataframe[column_name].isna().sum()
    missing_percentage = (missing_count / total_rows) * 100

    print(f"Total rows: {total_rows}")
    print(f"Missing values in '{column_name}': {missing_count} ({missing_percentage:.2f}%)")

    # Data for plotting
    categories = ['Missing', 'Not Missing']
    values = [missing_count, total_rows - missing_count]

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color=['red', 'green'])

    plt.title(f'Missing Value Analysis for "{column_name}"')
    plt.ylabel('Count')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

analyze_missing_values(dataframe=reviews, column_name='found_funny', save_path='C:/Users/yilma/PycharmProjects/RecommendationProject/visuals/found_funny_missing_values.png')

analyze_missing_values(dataframe=reviews, column_name='compensation', save_path='C:/Users/yilma/PycharmProjects/RecommendationProject/visuals/compensation_missing_values.png')

analyze_missing_values(dataframe=games, column_name='metascore', save_path='C:/Users/yilma/PycharmProjects/RecommendationProject/visuals/metascore_missing_values.png')