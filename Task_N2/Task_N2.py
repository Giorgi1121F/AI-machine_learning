import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# -----------------------------
# 1) Feature extraction from raw email text
# -----------------------------
SPAM_KEYWORDS = {
    "free", "winner", "urgent", "money", "offer", "click", "prize", "bonus",
    "limited", "now", "cash", "reward", "deal", "exclusive", "congratulations"
}

def extract_features_from_text(text: str) -> np.ndarray:
    # Words: basic tokenization (letters/numbers/underscore)
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    # Links: count common URL patterns
    link_count = len(re.findall(r"http[s]?://|www\.", text, flags=re.IGNORECASE))

    # Capital words: words that are fully uppercase (length >= 2 to avoid "I")
    capital_words = sum(1 for w in words if len(w) >= 2 and w.isupper())

    # Spam keyword count
    spam_word_count = sum(1 for w in words if w.lower() in SPAM_KEYWORDS)

    return np.array([word_count, link_count, capital_words, spam_word_count], dtype=float)


# -----------------------------
# 2) Load CSV and validate columns
# -----------------------------
REQUIRED_COLS = ["words", "links", "capital_words", "spam_word_count", "is_spam"]

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}\nFound: {list(df.columns)}")

    # Ensure numeric
    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    return df


# -----------------------------
# 3) Visualizations
#   A) Class distribution bar chart
#   B) Confusion matrix heatmap (matplotlib only)
#   C) Coefficient bar chart (optional but useful)
# -----------------------------
def plot_class_distribution(y: np.ndarray):
    counts = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))}
    plt.figure()
    plt.bar(["legitimate (0)", "spam (1)"], [counts[0], counts[1]])
    plt.title("Class Distribution (Spam vs Legitimate)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

def plot_confusion_matrix_heatmap(cm: np.ndarray):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.xticks([0, 1], ["legitimate (0)", "spam (1)"])
    plt.yticks([0, 1], ["legitimate (0)", "spam (1)"])

    # annotate cells
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.colorbar()
    plt.show()

def plot_coefficients_bar(model: LogisticRegression):
    # Binary logistic regression => coef_ shape (1, n_features)
    feature_names = ["words", "links", "capital_words", "spam_word_count"]
    coefs = model.coef_.ravel()

    plt.figure()
    plt.bar(feature_names, coefs)
    plt.title("Logistic Regression Coefficients (Feature Impact)")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient value")
    plt.xticks(rotation=15)
    plt.show()


# -----------------------------
# 4) Main training + evaluation + interactive email classification
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python spam_classifier.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = load_data(csv_path)

    X = df[["words", "links", "capital_words", "spam_word_count"]].to_numpy(dtype=float)
    y = df["is_spam"].to_numpy(dtype=int)

    # 70% train / 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Train logistic regression
    model = LogisticRegression(max_iter=5000, solver="lbfgs")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {acc:.4f}\n")

    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm, "\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["legitimate(0)", "spam(1)"]))

    # Coefficients
    print("\n=== Model Coefficients (impact of each feature) ===")
    feature_names = ["words", "links", "capital_words", "spam_word_count"]
    for name, coef in zip(feature_names, model.coef_.ravel()):
        print(f"{name:15s}: {coef:.6f}")
    print(f"{'intercept':15s}: {model.intercept_[0]:.6f}")

    # Visualizations (2 required; here are 3, you can keep 2 if you want)
    plot_class_distribution(y)
    plot_confusion_matrix_heatmap(cm)
    plot_coefficients_bar(model)  # optional (can remove if you only want 2)

    # Interactive email text classification
    print("\n=== Email Text Classification ===")
    print("Paste email text below. Type 'q' to quit.\n")

    while True:
        text = input("Email text > ").strip()
        if text.lower() == "q":
            break

        feats = extract_features_from_text(text).reshape(1, -1)
        proba_spam = float(model.predict_proba(feats)[0][1])
        pred = int(model.predict(feats)[0])

        label = "SPAM" if pred == 1 else "LEGITIMATE"
        print(f"Prediction: {label} | P(spam) = {proba_spam:.4f}\n")


if __name__ == "__main__":
    main()
