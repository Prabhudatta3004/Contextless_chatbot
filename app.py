from flask import Flask, request
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, supports_credentials=True)


# Read in CSV file and skip the first row
wifi_df = pd.read_csv('wifi_issues.csv', skiprows=1)

# Assign column names manually
wifi_df.columns = ['Question', 'Answer']

email_df=pd.read_csv('Email_Issues.csv',skiprows=1)
email_df.columns=['Question','Answer']

# hardware_df=pd.read_csv('Hardware_Issues.csv',skiprows=1)
# hardware_df.columns=['Question','Answer']

laptop_df=pd.read_csv("Laptop_Issues.csv",skiprows=1)
laptop_df.columns=['Question','Answer']

sap_df=pd.read_csv("Sap_Issues.csv",skiprows=1)
sap_df.columns=['Question','Answer']


# Load the pre-trained DistilBERT model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Define functions to generate embeddings and set threshold scores for each dataset
def generate_embeddings(df):
    question_embeddings = model.encode(df['Question'].values.tolist())
    return question_embeddings

def set_threshold_score(dataset):
    if dataset == 'wifi_df':
        threshold_score = 0.75
    elif dataset == 'email_df':
        threshold_score = 0.75
    # elif dataset == 'Hardware_df':
    #     threshold_score = 0.5
    # elif dataset == 'laptop_df':
    #     threshold_score = 0.5
    elif dataset == 'sap_df':
        threshold_score = 0.8
    else:
        threshold_score = 0.5
    return threshold_score

# Set the default dataset to 
current_dataset = 'wifi_df'
question_embeddings = generate_embeddings(wifi_df)
threshold_score = set_threshold_score(current_dataset)

def answer_question(user_query, question_embeddings, df, threshold_score):
    # Convert the user query into an embedding using the pre-trained model
    query_embedding = model.encode(user_query)

    # Compute the cosine similarity between the query embedding and the question embeddings
    cosine_similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]

    # Find the index of the question with the highest cosine similarity
    max_index = cosine_similarities.argmax().item()

    # Get the highest cosine similarity score
    max_score = cosine_similarities[max_index].item()

    # Print the values for debugging purposes
    print("User query: ", user_query)
    print("Max index: ", max_index)
    print("Max score: ", max_score)

    # If the similarity score is below the threshold, return a default message
    if max_score < threshold_score:
        return "Kindly connect with laptop.service@kiit.ac.in for any laptop related issues or connect with helpdesk@kiit.ac.in for Email or Wifi related Issues, if further assistance is required."
    else:
        # Return the answer associated with the most similar question
        return df.loc[max_index, 'Answer']

@app.route('/answer/wifi', methods=['POST'])
def answer_easy():
    global current_dataset, question_embeddings, threshold_score
    current_dataset = 'wifi_df'
    question_embeddings = generate_embeddings(wifi_df)
    threshold_score = set_threshold_score(current_dataset)

    # Get the user query from the request body
    user_query = request.json['query']

    # Call the answer_question function to get the answer
    answer = answer_question(user_query, question_embeddings, wifi_df, threshold_score)

    # Return the answer as a JSON response
    return {'answer': answer}

@app.route('/answer/email', methods=['POST'])
def answer_medium():
    global current_dataset, question_embeddings, threshold_score
    current_dataset = 'email_df'
    question_embeddings = generate_embeddings(email_df)
    threshold_score = set_threshold_score(current_dataset)

    # Get the user query from the request body
    user_query = request.json['query']
    print(user_query)

    # Call the answer_question function to get the answer
    answer = answer_question(user_query, question_embeddings, email_df, threshold_score)

    # Return the answer as a JSON response
    return {'answer': answer}
# @app.route('/answer/hardware', methods=['POST'])
# def answer():
#     global current_dataset, question_embeddings, threshold_score
#     current_dataset = 'Hardware_df'
#     question_embeddings = generate_embeddings(hardware_df)
#     threshold_score = set_threshold_score(current_dataset)

#     # Get the user query from the request body
#     user_query = request.json['query']

#     # Call the answer_question function to get the answer
#     answer = answer_question(user_query, question_embeddings, hardware_df, threshold_score)

#     # Return the answer as a JSON respons
#     return {'answer': answer}

@app.route('/answer/laptop', methods=['POST'])
def answer_software():
    global current_dataset, question_embeddings, threshold_score
    current_dataset = 'laptop_df'
    question_embeddings = generate_embeddings(laptop_df)
    threshold_score = set_threshold_score(current_dataset)

    # Get the user query from the request body
    user_query = request.json['query']

    # Call the answer_question function to get the answer
    answer = answer_question(user_query, question_embeddings, laptop_df, threshold_score)

    # Return the answer as a JSON respons
    return {'answer': answer}
@app.route('/answer/sap', methods=['POST'])
def answer_sap():
    global current_dataset, question_embeddings, threshold_score
    current_dataset = 'sap_df'
    question_embeddings = generate_embeddings(sap_df)
    threshold_score = set_threshold_score(current_dataset)

    # Get the user query from the request body
    user_query = request.json['query']

    # Call the answer_question function to get the answer
    answer = answer_question(user_query, question_embeddings, sap_df, threshold_score)

    # Return the answer as a JSON respons
    return {'answer': answer}

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=8080)
