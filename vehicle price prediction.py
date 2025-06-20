import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("dataset.csv")
print("Columns in dataset:", df.columns.tolist())  # Show column names
print(df.head())
print(df.info())

df = df.drop(['name', 'description', 'engine'], axis=1)
df = df.dropna(subset=['price'])
df = df.dropna()

df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

categorical_cols = ['make', 'model', 'fuel', 'transmission', 'trim',
                    'body', 'exterior_color', 'interior_color', 'drivetrain']

le = LabelEncoder()
for col in categorical_cols:
    df[col] = df[col].astype(str)  # Ensure all values are strings
    df[col] = le.fit_transform(df[col])

X = df.drop('price', axis=1)
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=importance, y=features)
plt.title("Feature Importance in Vehicle Price Prediction")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()