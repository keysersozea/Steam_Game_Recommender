import streamlit as st
import pandas as pd
from item_based_functions import item_based_suggestions
from user_based_functions import take_user, general_df, users, user_based_suggestions, create_playedGames_df
from final_suggestions import final_suggestions

# Function that gets called when the button is clicked
def on_button_click():
    random_user = take_user(general_df)
    username = users.loc[users["user_id"] == random_user, "username"].iloc[0]
    played_games_df = create_playedGames_df(general_df, random_user)
    # Display the username
    st.sidebar.write(f"Username: {username}")

    # Display the DataFrame

    st.sidebar.dataframe(played_games_df[["app_name"]], width=1500, height=800)

    st.session_state.username = username
    st.session_state.played_games_df = played_games_df

    return random_user

# Set up the page layout and title with custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #ADD8E6;
    }
    h1 {
        font-size: 24px;
    }
    .dataframe-container {
        overflow-x: auto;
        margin-left: 0px;  /* Sol kenar boşluğunu azalt */
    }
    .dataframe-container .dataframe {
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)
st.title("Steam Recommendation Engine")

# Create two columns with the second column being wider
#col1, col2 = st.columns([1, 2])


# Initialize session_state for random_user if it does not exist
if 'random_user' not in st.session_state:
    st.session_state.random_user = None
    st.session_state.username = ""
    st.session_state.played_games_df = None

# Button to take a random user
if st.sidebar.button('Take a Random User'):
    st.session_state.random_user = on_button_click()

# Display username and DataFrame in the sidebar if they are available
if st.session_state.username and st.session_state.played_games_df is not None:
    st.sidebar.write(f"Username: {st.session_state.username}")
    st.sidebar.dataframe(st.session_state.played_games_df[["app_name"]], width=1500, height=800)

# Button to recommend games
if st.button("Recommend Games"):
    if st.session_state.random_user is not None:
        user_based_recommendations, general_df, played_games_df, products = user_based_suggestions(st.session_state.random_user, general_df)
        item_based_recommendations = item_based_suggestions(general_df, st.session_state.random_user, played_games_df, products)
        final_recommendations = final_suggestions(user_based_recommendations, item_based_recommendations)
        st.dataframe(final_recommendations, width=1500, height=800)
    else:
        st.write("Please take a random user first.")
