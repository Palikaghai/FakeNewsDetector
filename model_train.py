import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from nltk.corpus import stopwords
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

# Step 1: Load Data
def load_data():
    fake = pd.read_csv(r"C:\Users\DELL\Videos\ml\Fake.csv")
    true = pd.read_csv(r"C:\Users\DELL\Videos\ml\True.csv")
    fake['label'] = 0  # 0 = Fake
    true['label'] = 1  # 1 = Real
    df = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)
    df = df[['title', 'label']].rename(columns={'title': 'text'})
    return df

# Step 2: Clean Text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Step 3: Visualizations
def plot_class_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label')
    plt.xticks([0, 1], ['Fake', 'Real'])
    plt.title("Class Distribution: Fake vs Real")
    plt.savefig("class_distribution.png")
    plt.show()

def plot_wordclouds(df):
    fake_words = " ".join(df[df['label'] == 0]['text'])
    real_words = " ".join(df[df['label'] == 1]['text'])

    wordcloud_fake = WordCloud(background_color='black').generate(fake_words)
    wordcloud_real = WordCloud(background_color='white').generate(real_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_fake, interpolation='bilinear')
    plt.axis("off")
    plt.title("Fake News WordCloud")
    plt.savefig("wordcloud_fake.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_real, interpolation='bilinear')
    plt.axis("off")
    plt.title("Real News WordCloud")
    plt.savefig("wordcloud_real.png")
    plt.show()

# Step 4: Train & Compare Models
def train_and_compare(df):
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.naive_bayes import MultinomialNB

    df['cleaned'] = df['text'].apply(clean_text)
    df['polarity'] = df['cleaned'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Vectorize only the text
    tfidf = TfidfVectorizer(max_features=5000)
    X_text = tfidf.fit_transform(df['cleaned'])

    # Split data BEFORE combining polarity (to avoid issues with Naive Bayes)
    X_train_text, X_test_text, y_train, y_test, train_polarity, test_polarity = train_test_split(
        X_text, df['label'], df['polarity'], test_size=0.2, random_state=42
    )

    # For other models, combine polarity
    from scipy.sparse import hstack
    X_train = hstack([X_train_text, np.array(train_polarity).reshape(-1, 1)])
    X_test = hstack([X_test_text, np.array(test_polarity).reshape(-1, 1)])

    # === Model 1: Logistic Regression
    print("🔸 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")

    # === Model 2: Random Forest
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {acc_rf:.4f}")

    # === Model 3: Tuned Random Forest
    print("🔍 Tuning Random Forest with GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }
    grid = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred_best = best_model.predict(X_test)
    acc_best = accuracy_score(y_test, y_pred_best)
    print(f"✅ Tuned RF Accuracy: {acc_best:.4f}")
    print(classification_report(y_test, y_pred_best))

    # === Model 4: Naive Bayes (uses only TF-IDF)
    print("🧪 Training Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train_text, y_train)
    y_pred_nb = nb.predict(X_test_text)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes Accuracy: {acc_nb:.4f}")

    # === Confusion Matrix (for best RF)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title("Confusion Matrix (Best RF)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # === Feature Importance (Best RF)
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:][::-1]
    feature_names = list(tfidf.get_feature_names_out())
    feature_names.append("polarity")
    top_features = [feature_names[i] for i in indices]

    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances[indices], y=top_features)
    plt.title("Top 10 Important Features (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

    # === Save best model & vectorizer
    joblib.dump(best_model, "fake_news_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("✅ Model and vectorizer saved successfully!")

    # === Model Comparison Table
    metrics = {
        "Model": ["Logistic Regression", "Random Forest", "Tuned Random Forest", "Naive Bayes"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_lr),
            accuracy_score(y_test, y_pred_rf),
            accuracy_score(y_test, y_pred_best),
            accuracy_score(y_test, y_pred_nb)
        ],
        "Precision": [
            precision_score(y_test, y_pred_lr),
            precision_score(y_test, y_pred_rf),
            precision_score(y_test, y_pred_best),
            precision_score(y_test, y_pred_nb)
        ],
        "Recall": [
            recall_score(y_test, y_pred_lr),
            recall_score(y_test, y_pred_rf),
            recall_score(y_test, y_pred_best),
            recall_score(y_test, y_pred_nb)
        ],
        "F1 Score": [
            f1_score(y_test, y_pred_lr),
            f1_score(y_test, y_pred_rf),
            f1_score(y_test, y_pred_best),
            f1_score(y_test, y_pred_nb)
        ]
    }

    comparison_df = pd.DataFrame(metrics)
    print("\n📊 Model Comparison Table:")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv("model_comparison.csv", index=False)

    # === Plot
    comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
        kind='bar', colormap='Set2', figsize=(10, 6), ylim=(0, 1.05)
    )
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("model_performance_comparison.png")
    plt.show()


# Main Pipeline
def main():
    print("🚀 Starting Fake News Detection Training...")
    df = load_data()
    plot_class_distribution(df)
    plot_wordclouds(df)
    train_and_compare(df)
    print("🎉 Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
