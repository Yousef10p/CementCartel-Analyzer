import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# 1. Load the dataset
df = pd.read_excel('Assignment Data.xlsx')

# 2. Define Independent (X) and Dependent (y) variables
features = ['FOIL', 'COAL', 'ELECTRICITY', 'GAS', 'PBAG', 'LIMESTONE', 'CLAY', 'GYPSUM', 'GDPGR']
X = df[features]
y = df['CPRICE']

# 3. Initialize and train the scikit-learn Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Get the R-squared value
r_squared = model.score(X, y)

# 4. Calculate Econometric Statistics (Standard Errors, t-stats, p-values)
# Combine intercept and coefficients into one array
params = np.append(model.intercept_, model.coef_)
predictions = model.predict(X)

# Add a column of 1s to X for the intercept matrix math
X_matrix = np.append(np.ones((len(X), 1)), X, axis=1)

# Calculate Degrees of Freedom and Mean Squared Error
n = len(X)
k = len(features)
df_resid = n - (k + 1)
mse = (sum((y - predictions)**2)) / df_resid

# Calculate Standard Errors, t-statistics, and p-values
var_b = mse * np.linalg.inv(np.dot(X_matrix.T, X_matrix)).diagonal()
std_errors = np.sqrt(var_b)
t_stats = params / std_errors
p_values = [2 * (1 - stats.t.cdf(np.abs(i), df_resid)) for i in t_stats]

# 5. Print the formatted output (Mimicking your image)
print("==============================================================================")
print("==============================================================================")
print(f"N = {n}   |   R-squared = {r_squared:.4f}")
print("------------------------------------------------------------------------------")
print(f"{'Variable':<12} {'Coefficient':>12} {'Std Error':>12} {'t-stat':>10} {'P-value':>10} {'Sig'}")
print("------------------------------------------------------------------------------")

# Create a list of variable names including the Constant (Intercept)
var_names = ['const'] + features

for i in range(len(var_names)):
    # Add asterisks for significance levels
    sig = ""
    if p_values[i] < 0.01:
        sig = "***"
    elif p_values[i] < 0.05:
        sig = "**"
    elif p_values[i] < 0.10:
        sig = "*"
        
    print(f"{var_names[i]:<12} {params[i]:>12.6f} {std_errors[i]:>12.6f} {t_stats[i]:>10.4f} {p_values[i]:>10.4f}  {sig}")

print("==============================================================================")

# 6. Generate and print the Correlation Matrix
print("\n==============================================================================")
print("CORRELATION MATRIX - Continuous Regressors")
print("==============================================================================")
corr_matrix = X.corr()
print(corr_matrix.round(3).to_string())