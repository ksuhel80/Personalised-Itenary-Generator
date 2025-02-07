import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pickle
import streamlit as st
from PIL import Image

image = Image.open("india_blue.jpg")


st.title("Welcome To Personalised Itenaraies Generator")

st.image(image, caption="Displayed using PIL", use_container_width=True)

source = st.selectbox("Source", ['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi'])

destination = st.selectbox("Destination", ['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi'])

preferred_time= st.selectbox("Preffered Time", ['Evening', 'Afternoon', 'Morning'])

user_interest = st.selectbox("User Interest", ['Historical', 'Religious', 'Environmental', 'Scientific', 'Market',
       'Botanical', 'Artistic', 'Scenic', 'Wildlife', 'Recreational',
       'Nature', 'Architectural', 'Entertainment', 'Sports',
       'Educational', 'Cultural', 'Food', 'Spiritual', 'Archaeological',
       'Adventure', 'Agricultural', 'Engineering Marvel',
       'Natural Wonder', 'Trekking', 'Shopping'])

budget = st.text_input('Budget',"5000")

class1 = st.selectbox('Class', ['Economy', 'Business'])


test_sample= {'airline': 'SpiceJet', 'source_city': source, 'departure_time': 'Evening', 'stops': 'zero', 'arrival_time': 'Night', 'destination_city': destination, 'class': class1}

# Loading label encoder for encoding user details

with open('model/label.pkl', 'rb') as f:
    encoders = pickle.load(f)


encoded_sample = {'duration' : 2.17, 'days_left': 1}
for col, le in encoders.items():
    if test_sample[col] in le.classes_:
        
        encoded_sample[col] = le.transform([test_sample[col]])[0]
    else:
        encoded_sample[col] = -1  # Handle unseen values
        

df = pd.DataFrame([encoded_sample])

# st.write("Encoded Test Sample:", encoded_sample)

# st.write(df)

# Loading model

# with open('airline_model.pkl', 'rb') as f:
#     airline_model = pickle.load(f)

airline_model = joblib.load('model/airline_model.pkl')

print(encoded_sample)

new_order = ['airline', 'source_city', 'departure_time', 'stops',
       'arrival_time', 'destination_city', 'class', 'duration', 'days_left',
       ]

df = df[new_order]

#  Predict flight prices and show with persistence


def predict_price():
    result = airline_model.predict(df)
    st.session_state['price'] = result[0]
    
if "price" not in st.session_state:
    st.session_state["price"] = ""



# Display the message after button click

st.title('Flight Price: '+str(st.session_state["price"]))

st.button("Predict Flight Price",key=1, on_click=predict_price)


# if st.button("Predict Flight Price"):
    
#     result = airline_model.predict(df)
#     st.title(f"Price: {result[0]}")


# st.title(result[0])



    
places_df = pd.read_csv("dataset/Top Indian Places to Visit.csv")
    

places_df['combined_features'] = places_df['Type'] + " " + places_df['Significance'] + " " + places_df['Best Time to visit'] + " " + places_df['City']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(places_df['combined_features'])

# with open('place.pkl', 'rb') as f:
#     tfidf_matrix = pickle.load(f)
    
def recommend_places(user_interest, preferred_time, city, budget=5000, top_n=5 ):
    user_input = user_interest + " " + preferred_time + " " + city
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

    recommended_places = places_df.iloc[top_indices]
    recommended_places = recommended_places[recommended_places['Entrance Fee in INR'] <= budget]
    return recommended_places[['Name', 'City', 'State', 'Type', 'Google review rating']]


st.title("Recommended Places:")

# if st.button("Suggest Some Places"):
  
#     recommended_places1 = recommend_places(user_interest, preferred_time, destination, int(budget))
#     st.write(recommended_places1)

#  Predict suggested places and show with persistence


def predict_places():
    recommended_places1 = recommend_places(user_interest, preferred_time, destination, int(budget))
    st.session_state["places"] = recommended_places1

# Check if the key exists in session state
if "places" not in st.session_state:
    st.session_state["places"] = ""

st.button("Suggest Some Places",key=2, on_click=predict_places)

# Display the message after button click
st.write(st.session_state["places"])

