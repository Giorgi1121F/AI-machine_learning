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

## Screenshots

Upload the screenshots to the repository (recommended folder: `screenshots/`) and ensure the file names match below.

Example structure:
```
screenshots/
  terminal_output.png
  plot.png
```

Then the images will appear here:

![Terminal Output](screenshots/terminal_output.png)
![Visualization](screenshots/plot.png)

---

## Reproducibility Notes

To reproduce the results:
1. Install the required libraries.
2. Run `python Task_N1.py`.
3. The terminal will print the coefficients, and a figure window will open showing the visualization.
