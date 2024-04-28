import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Read the data
df = pd.read_csv('news.csv')

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)

# Lists to store accuracy at each iteration
accuracy_list = []

# Train the model incrementally and calculate accuracy at each step
for i in range(len(x_train)):
    pac.partial_fit(tfidf_train[i:i+1], y_train[i:i+1], classes=np.unique(y_train))
    y_pred = pac.predict(tfidf_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

# Plot accuracy rate
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(x_train) + 1), accuracy_list, marker='o', color='skyblue')
plt.title('Accuracy Rate Over Training Iterations')
plt.xlabel('Training Iteration')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Plot confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues", xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Justification for Accuracy
print("\nJustification for Accuracy:")
print("Accuracy is a measure of the overall correctness of the model's predictions.")
print("The high accuracy score indicates that the model is able to correctly classify news articles into 'FAKE' and 'REAL' categories with a high degree of accuracy.")
print(f"Accuracy: {accuracy*100:.2f}%")

# Plot Word Clouds
fake_text = " ".join(df[df['label'] == 'FAKE']['text'])
real_text = " ".join(df[df['label'] == 'REAL']['text'])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.title('Word Cloud for FAKE News')
plt.axis('off')

plt.subplot(1, 2, 2)
wordcloud_real = WordCloud(width=800, height=400, background_color='white').generate(real_text)
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.title('Word Cloud for REAL News')
plt.axis('off')

plt.show()

# Plot Bar Plot of Feature Importance
feature_names = tfidf_vectorizer.get_feature_names_out()
coefs = pac.coef_[0]
top_features = sorted(zip(coefs, feature_names), reverse=True)[:20]

plt.figure(figsize=(10, 6))
sns.barplot(x=[feat[0] for feat in top_features], y=[feat[1] for feat in top_features])
plt.title('Top 20 Features Contributing to Classification')
plt.xlabel('Coefficient (Importance)')
plt.ylabel('Feature')
plt.show()
