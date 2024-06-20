# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/Kareem/Downloads/breast-cancer random forest/breast-cancer.csv")  # Update the path

# Checking the structure of the dataset
print(data.head())

# Encode the 'diagnosis' column if it's categorical
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Dropping the 'id' column as it is not needed for prediction
data = data.drop('id', axis=1)

# Split the dataset into features and the target variable
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test data
predictions = rf_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
# استخراج أهمية الميزات
feature_importances = rf_model.feature_importances_

# تحويل الميزات إلى DataFrame لتسهيل الرسم
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort
features_df = features_df.sort_values(by='Importance', ascending=False)

# إنشاء الرسم البياني
plt.figure(figsize=(10, 8))
plt.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()  
plt.show()