import time
import re
import flask 
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from flask_cors import CORS
import pyrebase

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = flask.Flask(__name__)
CORS(app)

config = {
  "apiKey": "AIzaSyBPuOXYlDP9nDUu_8UD3Gp1IwWxaq23H8k",
  "authDomain": "sentiment-17a98.firebaseapp.com",
  "databaseURL": "https://sentiment-17a98-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "sentiment-17a98.appspot.com",
  "storageBucket": "sentiments-6a0bd.appspot.com",
  "messagingSenderId" : "780644182175",
  "appId": "1:780644182175:web:84f34f7fac1368c60b9aaa",
  "measurementId": "G-PVJHLNGF6S"
};

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
database = firebase.database()
data1={}


# Initialize the YouTube API client
API_KEY = "AIzaSyCIUB_PX_0qtvZnqF39WLsnUnazGx3vkws"
youtube = build('youtube', 'v3', developerKey=API_KEY)

pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
wordnet_lemmatizer = WordNetLemmatizer()

def clean(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

def token_stop_pos(text):
    tags = nltk.pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

def getSubjectivity(review):
    return TextBlob(review).sentiment.subjectivity

def getPolarity(review):
    return TextBlob(review).sentiment.polarity

def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

def get_comments(youtube, video_id, time_limit=60):
    comments = []
    next_page_token = None
    retries = 0
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        try:
            video_response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            ).execute()
            
            for item in video_response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
                
            next_page_token = video_response.get('nextPageToken')
            if not next_page_token:
                break

            time.sleep(1)  # Add a small delay to avoid hitting rate limits

        except HttpError as e:
            if e.resp.status in [403, 429]:  # Handle rate limit errors
                print("Rate limit exceeded, waiting before retrying...")
                time.sleep(2 ** retries)  # Exponential backoff
                retries += 1
                if retries > 5:
                    raise Exception("Exceeded maximum number of retries")
            else:
                raise

    return comments

def function_analyzer(url):
    video_id = re.search(r'v=([a-zA-Z0-9_-]+)', url).group(1)
    data1 = {"video_id": video_id}
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    video_response = youtube.videos().list(part='snippet,statistics', id=video_id).execute()
    video_info = video_response['items'][0]

    title = video_info['snippet']['title']
    author = video_info['snippet']['channelTitle']
    duration = video_info['snippet']['publishedAt']
    likes = video_info['statistics'].get('likeCount', 0)
    dislikes = video_info['statistics'].get('dislikeCount', 0)
    viewcount = video_info['statistics']['viewCount']
    rating = video_info['statistics'].get('averageRating', 'N/A')

    comments = get_comments(youtube, video_id, time_limit=60)

    df = pd.DataFrame(data={"comments": comments})
    df['Cleaned Reviews'] = df['comments'].apply(clean)

    print("Cleaned Reviews:\n", df['Cleaned Reviews'].head())  # Debugging: Print cleaned comments

    df['POS tagged'] = df['Cleaned Reviews'].apply(token_stop_pos)

    print("POS Tagged Reviews:\n", df['POS tagged'].head())  # Debugging: Print POS tagged comments

    df['Lemma'] = df['POS tagged'].apply(lemmatize)

    print("Lemmatized Reviews:\n", df['Lemma'].head())  # Debugging: Print lemmatized comments

    fin_data = pd.DataFrame(df[['comments', 'Lemma']])
    fin_data['Polarity'] = fin_data['Lemma'].apply(getPolarity)

    print("Polarity Scores:\n", fin_data['Polarity'].head())  # Debugging: Print polarity scores

    fin_data['Analysis'] = fin_data['Polarity'].apply(analysis)

    print("Sentiment Analysis:\n", fin_data['Analysis'].head())  # Debugging: Print sentiment analysis

    tb_counts = fin_data.Analysis.value_counts()
    sentiment = fin_data['Analysis'].value_counts().idxmax()
    x = fin_data[fin_data['Analysis'] == 'Neutral']

    data = [title, author, duration, likes, dislikes, viewcount, rating, sentiment, (tb_counts[0] / float(fin_data.shape[0])) * 100]
    lst = x['comments'].iloc[0:5]
    data.append(lst)

    return data
@app.route('/analyze', methods=['POST'])
def analyzer():
    url = flask.request.json['url']
    print("URL: ", url)
    lst = function_analyzer(url)
    colname = ["Title", "Author", "Duration", "Likes", "Dislikes", "ViewCount", "Rating", "Sentiment_of_Comments", "Top_Comments_Percentage"]
    data1 = {}
    i = 0

    for item in lst:
        if i < len(colname):
            print(colname[i], item, sep=" : ")
            data1[colname[i]] = item
        else:
            print(item)
        i += 1

    # Adding top comments to data1 dictionary
    top_comments_key = "Top_Comments"
    data1[top_comments_key] = lst[-1].tolist() if isinstance(lst[-1], pd.Series) else lst[-1]

    # Assuming you have configured Firebase as shown before
    database.child("Video Sentiments").set(data1)
    return flask.jsonify(data1)



@app.route('/' , methods=['GET'])
def home():
    return "Hello World"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)