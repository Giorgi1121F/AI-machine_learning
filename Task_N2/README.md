# Email Spam Classification using Logistic Regression

This repository contains a Python console application that classifies emails as **spam** or **legitimate** using a **logistic regression** model.  
The dataset consists of extracted email features and their corresponding classes.

---

## Dataset

The provided CSV file contains the following columns:

- `words` – total number of words in the email  
- `links` – number of links found in the email  
- `capital_words` – number of fully capitalized words  
- `spam_word_count` – count of predefined spam-related keywords  
- `is_spam` – class label (1 = spam, 0 = legitimate)

The dataset file is uploaded to this repository and is used directly by the application.

---

## How to Run

### 1. Install requirements
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2. Run the program
```bash
python spam_classifier.py k_sepherteladze25_72634.csv
```

---

## Model Description

A **logistic regression** classifier is trained on **70% of the dataset**, while the remaining **30%** is used for validation.  
The model learns the relationship between the extracted features and the email class.

After training, the model outputs:
- accuracy score,
- confusion matrix,
- classification report,
- model coefficients showing the impact of each feature.

---

## Evaluation Results

The trained model achieved high accuracy on unseen test data.  
Validation includes:
- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, and F1-score**

A screenshot of the terminal output containing these results is included in the repository.

---

## Email Text Classification

The application supports **interactive email text classification**.  
The user can paste raw email text into the console. The program then:
1. Parses the text,
2. Extracts the same features as in the dataset,
3. Classifies the email as spam or legitimate.

### Example – Spam Email
```
URGENT!!! You are a WINNER!
Click now to get your FREE bonus prize.
Limited time offer. Visit http://free-bonus-now.com
```

This email is classified as spam due to spam keywords, capitalized words, and the presence of a link.

### Example – Legitimate Email
```
Hello Kate,
Please find attached the meeting agenda for tomorrow.
Let me know if you have any questions.
Best regards, David
```

This email is classified as legitimate as it lacks spam-related patterns.

---

## Visualizations

The application generates the following visualizations:

1. **Class Distribution Chart**  
   Shows the ratio of spam vs. legitimate emails in the dataset, helping to identify class imbalance.

2. **Confusion Matrix Heatmap**  
   Provides a graphical representation of prediction performance, showing true vs. predicted classes.

(Visualization screenshots are included in the repository.)

---

## Reproducibility

To reproduce the results:
1. Install the required Python libraries.
2. Run the script with the provided CSV file.
3. Review the terminal output and generated plots.

All steps are fully reproducible using the files in this repository.
