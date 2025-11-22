import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(20)
n_samples = 100
X = np.random.uniform(-1, 1, size=(n_samples, 1))
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(scale=1, size=n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

degrees = np.arange(1, 130)
train_errors = []
test_errors = []

for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_poly)
    X_test_pca = pca.transform(X_test_poly)

    model = Ridge(alpha=0)
    model.fit(X_train_pca, y_train)

    y_train_pred = model.predict(X_train_pca)
    y_test_pred = model.predict(X_test_pca)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

plt.figure(figsize=(10,6))
plt.plot(degrees, train_errors, label='Train MSE', marker='o')
plt.plot(degrees, test_errors, label='Test MSE', marker='o')
plt.yscale('log')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (log scale)')
plt.title('Polynomial Regression with PCA (95% Variance Retained)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figures/pca_95%_variance.png", dpi=300)
plt.close()
