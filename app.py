# Importing the libraries
import pickle
import pandas as pd
import webbrowser
# !pip install dash
import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# Declaring Global variables
# any variable declarled is Global by default
project_name = None
app = dash.Dash(suppress_callback_exceptions = True)

# Defining My Functions
def load_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
  
    global pickle_model
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)

    global vocab
    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)

def check_review(reviewText):

    #reviewText has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model preidctor
    #load it later

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))


    # Add code to test the sentiment of using both the model
    # 0 == negative   1 == positive
    
    return pickle_model.predict(vectorised_review)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

def create_app_ui():
    main_layout = html.Div(
        [
        html.H1(id = "main_title", children = "Sentiment Analysis with Insights") ,
        
        dcc.Textarea(
            id = "textarea_review",
            placeholder = "Enter the review here .....",
            style = {'width':'100%', 'height' : 100 }
            ),
        
        html.H1(id='result', children = None)
        
        ]
        )
    
    return main_layout


@app.callback(
    Output( 'result'   ,  "children"   ),
    [
    Input( "textarea_review"    ,  'value'    ) 
    ]
    )
def update_app_ui (  textarea_value  ):
    print("Data Type of ", str(type(textarea_value)) )
    print("Value = ", str(textarea_value))
    
    # we need to write the logic for checking the review
    
    response = check_review(textarea_value)
    print("response = ",response)
    
    if (response[0] == 0):
        result1 = 'Negative'
    elif (response[0] == 1):
        result1 = "Positive"
    else:
        result1 = 'Unknown'
      
    return result1


# Main Function to control the Flow of your Project
def main():
    print("Start of your project")
    
    load_model()
    open_browser()
    #update_app_ui()
    
    
    global project_name 
    global scrappedReviews 
    global app
    project_name = "Sentiment Analysis with Insights"
    
    
    #print("My project name = ", project_name)
    #print("my scrapped data = ",scrappedReviews.sample(5) )
    
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()
    
    #update_app_ui()
    
    print("End of your project")
    project_name = None
    scrappedReviews = None
    app = None
            
# Calling the main function 
if __name__ == '__main__':
    main()



















# https://pastebin.com/UN8hTQ8y

"""
1. Project Structuring
2. Global and Local Variables
3. Introduction to DASH Library
4. Setting up the favicon ( if time permits )

"""    
    