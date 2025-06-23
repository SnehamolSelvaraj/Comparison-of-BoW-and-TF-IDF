import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix
)

# Load Dataset
df = pd.read_csv("dataset.csv", encoding="ISO-8859-1")
df = df[["Hotel Name", "Review Score", "Label"]].dropna()

# Features and labels
X = df["Hotel Name"]
y = df["Label"]

# 🌥️ Word Cloud from Hotel Names
text_corpus = " ".join(X.astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Hotel Names")
plt.tight_layout()
plt.show()

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorizers to compare
vectorizers = {
    "Bag-of-Words": CountVectorizer(),
    "TF-IDF": TfidfVectorizer()
}

results = {}

# 🔍 Evaluate each vectorizer
for name, vectorizer in vectorizers.items():
    X_train_vec = vectorizer.fit_transform(X_train.astype(str))
    X_test_vec = vectorizer.transform(X_test.astype(str))

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    print(f"\n🔎 {name} Results")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    results[name] = {
        "accuracy": acc,
        "conf_matrix": cm,
        "classes": model.classes_
    }

# 📊 Accuracy Comparison
accuracy_df = pd.DataFrame({
    "Vectorizer": list(results.keys()),
    "Accuracy": [res["accuracy"] for res in results.values()]
})

plt.figure(figsize=(6, 4))
sns.barplot(x="Vectorizer", y="Accuracy", data=accuracy_df, palette="Set2")
plt.title("🔍 Accuracy Comparison: BoW vs TF-IDF")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

# 🔥 Confusion Matrix Heatmaps
for name, res in results.items():
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        res["conf_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=res["classes"],
        yticklabels=res["classes"]
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()
