import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, render_template
import pickle

# Load datasets
real_news = pd.read_csv("D:/Real_fake_detection/True.csv")
fake_news = pd.read_csv("D:/Real_fake_detection/Fake.csv")

# Add labels to the data
real_news['label'] = 1
fake_news['label'] = 0

# Combine real and fake news
news_data = pd.concat([real_news, fake_news])
news_data = news_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert text to lowercase
news_data['text'] = news_data['text'].str.lower()

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the text data
tfidf_features = tfidf.fit_transform(news_data['text'])

# Apply TruncatedSVD for dimensionality reduction
svd = TruncatedSVD(n_components=100, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_features)

# Define labels
labels = news_data['label'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(tfidf_reduced, labels, test_size=0.2, random_state=42)

# Initialize and train the SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Save the model and preprocessing objects
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('svd.pkl', 'wb') as f:
    pickle.dump(svd, f)

# Initialize Flask application
app = Flask(__name__)

# Load the model and preprocessing objects
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('svd.pkl', 'rb') as f:
    svd = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        news_text = news_text.lower()
        tfidf_features = tfidf.transform([news_text])
        svd_features = svd.transform(tfidf_features)
        prediction = model.predict(svd_features)
        
        if prediction == 1:
            result = "Real"
        else:
            result = "Fake"
        
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
