import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# =========================
# 1) Data (x, y) by color
# =========================
red = np.array([
    [2.52, 5.02],
    [3.05, 6.12],
    [4.02, 6.62],
    [4.41, 5.56],
    [4.95, 6.48],
    [6.40, 7.44],
])

blue = np.array([
    [12.34, 6.98],
    [12.52, 7.62],
    [13.34, 6.48],
    [14.41, 6.42],
    [14.66, 7.20],
    [15.52, 7.00],
])

yellow = np.array([
    [18.45, 7.80],
    [19.27, 7.42],
    [20.05, 8.68],
    [21.30, 8.64],
])

purple = np.array([
    [2.45, 2.72],
    [2.48, 1.50],
    [3.27, 1.22],
    [4.48, 0.90],
    [5.13, 3.12],
    [5.34, 2.06],
    [5.98, 1.52],
])

green = np.array([
    [18.16, 2.44],
    [19.80, 1.70],
    [21.73, 3.30],
])

# =========================
# 2) Create dataset
# Class 1: all above the line (red + blue + yellow)
# Class 2: purple below the line
# Class 3: green below the line
# =========================
X = np.vstack([red, blue, yellow, purple, green])

y = np.array(
    [1] * len(red)
    + [1] * len(blue)
    + [1] * len(yellow)
    + [2] * len(purple)
    + [3] * len(green)
)

# =========================
# 3) Train multinomial Logistic Regression
# =========================
model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)
model.fit(X, y)

# =========================
# 4) Print model coefficients
# =========================
print("=== Logistic Regression Coefficients ===")
print("Classes:", model.classes_)
print("coef_ shape:", model.coef_.shape, "(n_classes, n_features)")
print("intercept_ shape:", model.intercept_.shape, "(n_classes,)")

for cls, coefs, b in zip(model.classes_, model.coef_, model.intercept_):
    print(f"\nClass {cls}:")
    print(f"  coef_x = {coefs[0]:.6f}")
    print(f"  coef_y = {coefs[1]:.6f}")
    print(f"  intercept = {b:.6f}")

# =========================
# 5) Visualization: points + decision regions + given reference line
# =========================
# Grid for decision regions
x_min, x_max = 0.5, 25.0
y_min, y_max = 0.0, 10.0
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 600),
    np.linspace(y_min, y_max, 600)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(12, 7))

# Decision regions (no custom colors specified, default colormap)
plt.contourf(xx, yy, Z, alpha=0.18)

# Scatter points (colors roughly matching your plot)
plt.scatter(red[:, 0], red[:, 1], label="Red (Class 1)", edgecolors="k")
plt.scatter(blue[:, 0], blue[:, 1], label="Blue (Class 1)", edgecolors="k")
plt.scatter(yellow[:, 0], yellow[:, 1], label="Yellow (Class 1)", edgecolors="k")
plt.scatter(purple[:, 0], purple[:, 1], label="Purple (Class 2)", edgecolors="k")
plt.scatter(green[:, 0], green[:, 1], label="Green (Class 3)", edgecolors="k")

# Reference line (from the shown graph: approximately y = 0.07x + 3.7)
# If your teacher expects a different exact line, change m and c below.
m = 0.07
c = 3.70
x_line = np.array([x_min, x_max])
y_line = m * x_line + c
plt.plot(x_line, y_line, linewidth=2, label=f"Reference line: y = {m:.2f}x + {c:.2f}")

plt.title("Multiclass Logistic Regression: Class 1 (Above Line), Class 2 (Purple), Class 3 (Green)")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()