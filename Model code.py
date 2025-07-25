# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Step 2: Load Dataset
df = pd.read_csv("teen_phone_data.csv")  # Replace with your CSV file
print(df.head())
print(df.info())

# Step 3: Target Column Check
print(df['Addiction_Level'].value_counts())

# Visualize target distribution
sns.histplot(df['Addiction_Level'], bins=30, kde=True)
plt.title("Addiction Level Distribution")
plt.xlabel("Addiction Level")
plt.ylabel("Frequency")
plt.show()

# Optional: Correlation Matrix
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Step 4: Data Preprocessing

# Drop non-informative columns
df = df.drop(['ID', 'Name'], axis=1)  # Drop ID and Name (not useful for prediction)

# Convert categorical columns using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Step 5: Feature-Target Split
X = df.drop("Addiction_Level", axis=1)
y = df["Addiction_Level"]

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Model Evaluation
y_pred = model.predict(X_test)

print("✅ Evaluation Metrics:")
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))

# Optional: Scatter plot to compare actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Addiction Level")
plt.ylabel("Predicted Addiction Level")
plt.title("Actual vs Predicted Addiction Level")
plt.show()

# Step 9: Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.show()
