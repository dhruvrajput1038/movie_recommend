import streamlit as st
import pickle
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import requests
# model=pickle.load(open('model.pkl','rb'))


# def modelmaking():
#     movies_data = pd.read_csv('/movies.csv')
#     selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
#     for feature in selected_features:
#         movies_data[feature] = movies_data[feature].fillna('')
#     combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + \
#                         movies_data['cast'] + ' ' + movies_data['director']
#     vectorizer = TfidfVectorizer()
#     feature_vectors = vectorizer.fit_transform(combined_features)
#     similarity = cosine_similarity(feature_vectors)
#

# def fetch_poster(movieid):
#     # apikeys - 8265bd1679663a7ea12ac168da84d2e8
#     response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&Ianguage=en-US'.format(movieid))
#     data = response.json()
#     return "https://image.tmdb.org/t/p/w500/"+data['poster_path']




def predict_by_title(moviename,movies_data):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(moviename, list_of_all_titles)
    return find_close_match

def predict_other(moviename, movies_data, similarity, close_match):
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    i = 1
    other_movies = []
    movie_poster = []
    for movie in sorted_similar_movies:
        index = movie[0]

        # Fetch movie IDs directly using iloc
        movie_index = movies_data.iloc[index]['id']

        # Handle single movie ID case
        if isinstance(movie_index, np.int64):
            movie_index = [movie_index]

        # Fetch the movie poster for each ID
        # for i in movie_index:
        #     movie_poster.append(fetch_poster(i))

        title_from_index = movies_data.iloc[index]['title']
        if i < 11:
            other_movies.append(title_from_index)
            i += 1

    return other_movies


def main():

    # Model Making-------------------------------------
    movies_data = pd.read_csv('movies.csv')
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + \
                        movies_data['cast'] + ' ' + movies_data['director']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)


    # Main---------------------------------------------
    st.title("Movie Recommended System ")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Movie Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    # modelmaking()
    df = pd.read_csv("movies.csv")
    movie_list = df["title"].to_list()
    movie_name_selected = st.selectbox("Movie Name: ",movie_list)

    if st.button("Predict"):
        output1 = predict_by_title(movie_name_selected,movies_data)
        # st.subheader('Similar title movie are')
        #
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #         st.header(output1[0])
        #         # st.image("https://static.streamlit.io/examples/cat.jpg")
        # with col2:
        #         st.header(output1[1])
        #         # st.image("https://static.streamlit.io/examples/dog.jpg")
        # with col3:
        #         st.header(output1[2])
        #         # st.image("https://static.streamlit.io/examples/owl.jpg")

        movie_name = predict_other(movie_name_selected,movies_data,similarity,movie_name_selected)
        st.subheader('10 Similar movie are')

        s = ''
        for i in movie_name: s += "- " + i + "\n"
        st.markdown(s)

if __name__=='__main__':
    main()