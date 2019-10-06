from flask import Flask, render_template,url_for,request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as numpy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
Bootstrap(app)

@app.route('/')

def index():
    return render_template('index.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5])

@app.route('/predict',methods=['POST'])
def predict():

    ###### helper functions. Use them when needed #######
    def get_title_from_index(index):
        return movie_data[movie_data.index == index]["title"].values[0]
    def get_index_from_title(title):
        return movie_data[movie_data.title == title]["index"].values[0]
    ##################################################
    # ##Step 1: Read CSV File
    movie_data = pd.read_csv("movie_dataset.csv")
    ##Step 2: Select Features
    features = ['keywords','cast', 'genres','director']
    ##Step 3: Create a column in DF which combines all selected features
    for feature in features:
        movie_data[feature] = movie_data[feature].fillna('')
    def combined_features(row):
        try:
            return row['keywords']+"." +row["cast"]+"."+row["genres"]+"."+row["director"]
        except:
            print ("error")

    movie_data["combined_features"] = movie_data.apply(combined_features,axis=1)
    print ("combined features"), movie_data["combined_features"].head()
    ##Step 4: Create count matrix from this new combined column
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(movie_data["combined_features"])
    ##Step 5: Compute the Cosine Similarity based on the count_matrix
    similarity_scores = cosine_similarity(X)

    movie_user_likes = "Avatar"

    ## Step 6: Get index of this movie from its title
    movie_index = get_index_from_title(movie_user_likes)
    ## Step 7: Get a list of similar movies in descending order of similarity score
    similar_movies = list(enumerate(similarity_scores[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key = lambda x:x[1],reverse = True)

    if request.method == 'POST':
        namequery = request.form['namequery']
        data = [namequery]
        ## Step 8: Print titles of first 50 movies
        i=0
        ans = []
        for movie in sorted_similar_movies:
            #print(get_title_from_index(movie[0]))
            ans.append(get_title_from_index(movie[0]))
            i=i+1
            if i>50:
                break
    return render_template('results.html',prediction=ans,name=namequery.upper())
if __name__ == '__main__':
    app.run(debug=True)