# Multiclass Logistic Regression – Point Classification (Task_N1.py)

This project trains a **multiclass logistic regression** model in Python to classify 2D points into three classes based on their position relative to a reference line and their color:

- **Class 1:** all points **above** the line (red + blue + yellow points)
- **Class 2:** **purple** points **below** the line
- **Class 3:** **green** points **below** the line

The program prints the learned **model coefficients** and generates a **visualization** showing the decision regions, the points, and the reference line.

---

## Files

- `Task_N1.py` — main Python script (data, training, coefficients, visualization)

---

## How to Run

### 1) Install requirements
```bash
pip install numpy matplotlib scikit-learn
```

### 2) Run the script
```bash
python Task_N1.py
```

---

## Data

The point coordinates `(x, y)` were **manually extracted from the online graph** by hovering over each colored dot and recording its coordinates. These values are hard-coded in `Task_N1.py`.

### Class definitions (as required)
- **Class 1:** all points above the reference line  
- **Class 2:** purple points below the line  
- **Class 3:** green points below the line  

---

## Model

A **multinomial logistic regression** classifier is trained using:
- Features: `X = [x, y]`
- Labels: `y ∈ {1, 2, 3}`

The learned parameters are:
- `coef_` → coefficients for each class (for x and y)
- `intercept_` → bias term for each class

Each class score is computed in the form:

`score_class = coef_x * x + coef_y * y + intercept`

---

## Results (Model Coefficients)

The script produced the following coefficients (from the terminal output):

**Class 1**
- coef_x = **-0.093256**
- coef_y = ** 1.125782**
- intercept = **-2.233611**

**Class 2**
- coef_x = **-0.295478**
- coef_y = **-0.716972**
- intercept = ** 6.009220**

**Class 3**
- coef_x = ** 0.388733**
- coef_y = **-0.408810**
- intercept = **-3.775609**

---

## Visualization

`Task_N1.py` generates a plot that includes:
- colored data points,
- model **decision regions**,
- and the **reference line**.

The reference line is drawn as: **y = 0.07x + 3.70** (approximated from the provided graph).

---
<img width="1197" height="769" alt="image" src="https://github.com/user-attachments/assets/10c92538-ac98-4885-8bb0-06adfec3e8be" />
<img width="1476" height="466" alt="image" src="https://github.com/user-attachments/assets/1894356e-2a06-407c-85e5-2221c9a8ec30" />


---

## Reproducibility Notes

To reproduce the results:
1. Install the required libraries.
2. Run `python Task_N1.py`.
3. The terminal will print the coefficients, and a figure window will open showing the visualization.


# Email Spam Classification using Logistic Regression

This repository contains a Python console application that classifies emails as **spam** or **legitimate** using a **logistic regression** model.  
The dataset consists of extracted email features and their corresponding classes. (Task_N2.py)

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
python Task_N2.py g_gzirishvili25_44721924.csv
```
<img width="698" height="531" alt="image" src="https://github.com/user-attachments/assets/ac0a8044-c43e-4b3a-a6ef-b59b71c5c9db" />
<img width="637" height="550" alt="image" src="https://github.com/user-attachments/assets/74caea28-77bf-4b0f-b7e8-9ee11964ef85" />
<img width="637" height="553" alt="image" src="https://github.com/user-attachments/assets/9f9bdcf4-9073-4e5b-b06e-7567d7630845" />
<img width="636" height="547" alt="image" src="https://github.com/user-attachments/assets/1e2e37e4-6e5a-4561-a618-69e0ba22d516" />


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
<img width="1071" height="483" alt="image" src="https://github.com/user-attachments/assets/8f59f12b-efa9-42c3-a71e-ab9f7be4c869" />

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
