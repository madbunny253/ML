import pandas as pd
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('temperatures.csv')

# Split the data into features (Month) and target variable (Temperature_Celsius)
X = data[['YEAR']]
y = data['ANNUAL']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit a linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict temperatures for the test set
y_pred = model.predict(X_test)

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-Squared (R2): {r2}')

# Visualize the simple linear regression model
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Temperatures vs Month')
plt.show()

# Predict the temperature for the curriculum month (e.g., Month 13)
curriculum_month = 13
predicted_temperature = model.predict([[curriculum_month]])
print(f'Predicted temperature for Month {curriculum_month}: {predicted_temperature[0]:.2f} Celsius')
