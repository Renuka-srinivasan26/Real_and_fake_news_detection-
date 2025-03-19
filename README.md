INTRODUCTION:
    The prevalence of fake news has become a significant issue in today's digital age. Misinformation can have widespread and damaging effects, influencing public opinion, political decisions, and societal trust. This project addresses this problem by developing a machine learning-based solution to detect fake news. By analyzing textual features of news articles, the model aims to distinguish between real and fake news with high accuracy. The project integrates various machine learning techniques and is deployed through a user-friendly web application.

ADVANTAGES IN THIS PROJECT:
  Detection of Fake News: Helps users identify whether a news article is real or fake.
  Educational Tool: Provides an example of applying machine learning to a real-world problem, useful for learning and teaching.
  Public Awareness: Increases awareness about the problem of fake news and the importance of verifying information.
  Support for Fact-Checkers: Assists journalists and fact-checkers in verifying news articles efficiently.

TECHNOLOGIES AND LIBRARIES:
Programming Language: Python
Data Handling: Pandas for data loading and manipulation.
Feature Extraction: TfidfVectorizer from sklearn for converting text to numerical features.
Dimensionality Reduction: TruncatedSVD from sklearn for reducing feature space.
Machine Learning Model: Support Vector Machine (SVM) with a linear kernel from sklearn.
Model Evaluation: Accuracy score and classification report from sklearn.
Model Serialization: Pickle for saving and loading the trained model and preprocessing objects.
Web Framework: Flask for developing the web application.
Frontend: HTML for creating the user interface.

EXISTING METHODOLOGY:
1. Manual Verification:
Process: Rely on human fact-checkers to manually verify the authenticity of news articles.
Drawbacks: Time-consuming, labour-intensive, prone to human error.


2. Rule-based Systems:
Process: Use predefined rules and keyword matching to detect fake news.
Drawbacks: Limited flexibility, can't handle complex language nuances, high false positive/negative rates.
3. Basic Machine Learning Models:
Process: Use basic machine learning models without advanced preprocessing and feature extraction.
Drawbacks: Lower accuracy, limited scalability, inability to handle large datasets effectively.

PROPOSED METHODOLOGY:
1. Data Collection and Preprocessing:
Loading Data: Import datasets of real and fake news articles.
Labeling: Assign labels (1 for real, 0 for fake).
Combining Data: Merge real and fake news datasets.
Text Preprocessing: Convert text to lowercase, remove stop words.
2. Feature Extraction:
Technique: TF-IDF Vectorization
Purpose: Convert text data into numerical features.
3. Dimensionality Reduction:
Technique: Truncated Singular Value Decomposition (TruncatedSVD)
Parameter: n_components=100
Purpose: Reduce the dimensionality of TF-IDF features to improve computational efficiency.

4. Model Training:
Algorithm: Support Vector Machine (SVM) with a linear kernel
Dataset Split: 80% training data, 20% testing data
Training: Train the SVM model using the training dataset.
5. Model Evaluation:
Metrics: Accuracy, precision, recall, F1-score
Evaluation Process: Predict test set labels, calculate evaluation metrics.
6. Model Deployment:
Tool: Flask web framework
Purpose: Create a web application for users to input news articles and get predictions.
Model Serialization: Save trained model using PICKLE.
