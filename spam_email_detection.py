
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("sms.tsv", sep="\t", names=["label", "message"])

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== LABEL COUNT ==========")
print(df["label"].value_counts())

# ----------------------------
# Feature Engineering
# ----------------------------
df["length"] = df["message"].apply(len)

# ----------------------------
# Graph 1 - Count Plot
# ----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="label")
plt.title("Spam vs Ham Count")
plt.show()

# ----------------------------
# Graph 2 - Pie Chart
# ----------------------------
plt.figure(figsize=(5,5))
df["label"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.title("Spam vs Ham Distribution")
plt.show()

# ----------------------------
# Graph 3 - Message Length
# ----------------------------
plt.figure(figsize=(7,4))
sns.histplot(df["length"], bins=30)
plt.title("Message Length Distribution")
plt.xlabel("Message Length")
plt.show()

# ----------------------------
# Graph 4 - Box Plot
# ----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="label", y="length", data=df)
plt.title("Message Length by Label")
plt.show()

# ----------------------------
# Convert Labels
# ham = 0, spam = 1
# ----------------------------
df["label"] = df["label"].map({"ham":0,"spam":1})

# ----------------------------
# Text Vectorization
# ----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# ----------------------------
# Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Model
# ----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------
y_pred = model.predict(X_test)

print("\n========== ACCURACY ==========")
print(accuracy_score(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# User Prediction
# ----------------------------
print("\n========== SPAM EMAIL DETECTOR ==========")
msg = input("Enter a message: ")

msg_vector = vectorizer.transform([msg])
prediction = model.predict(msg_vector)

if prediction[0] == 1:
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: NOT SPAM")
