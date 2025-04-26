

import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning

# # Ignore specific warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# warnings.filterwarnings("ignore", category=DataConversionWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle


dataset_path = './final_dataset_BFP .csv'
df = pd.read_csv(dataset_path)


label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['BFPcase'] = label_encoder.fit_transform(df['BFPcase'])
df['BMIcase'] = label_encoder.fit_transform(df['BMIcase'])

scaler = StandardScaler()
numerical_features = ['Weight', 'Height', 'BMI', 'Body Fat Percentage', 'Age']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

df = df.drop(['BFPcase', 'BMIcase'], axis=1)

X = df.drop('Exercise Recommendation Plan', axis=1)

y = df['Exercise Recommendation Plan'].astype('float')-1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'XGBoost Classifier': XGBClassifier(),
    'XGBoost Regressor': XGBRegressor()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Decision Tree': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    'Naive Bayes': {},
    'XGBoost Classifier': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'XGBoost Regressor': {'max_depth': [3, 4, 5, 6], 'learning_rate': [0.01, 0.1, 0.2, 0.3], 
                            'colsample_bytree': [0.6, 0.8, 1.0], 'gamma': [0, 0.1, 0.2, 0.3]}
}
for model_name, model in models.items():
    if model_name == 'XGBoost Regressor':
        scoring_metric = ['neg_mean_squared_error', 'r2']
    else:
        scoring_metric = ['accuracy', 'r2']

    grid_search = GridSearchCV(model, param_grids[model_name], cv=10, scoring=scoring_metric, refit='r2')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

y = df['Exercise Recommendation Plan'] - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(C=10, solver='saga'),
    'Random Forest': RandomForestClassifier(max_depth=10, n_estimators=50),
    'Support Vector Machine': SVC(C=10, kernel='linear'),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=5),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'Naive Bayes': GaussianNB(),
    'XGBoost Classifier': XGBClassifier(learning_rate=0.01, n_estimators=100),
    'XGBoost Regressor': XGBRegressor(colsample_bytree=1.0, gamma=0.2, learning_rate=0.2, max_depth=3)
}

print("Accuracies After Optimization")
print("-" * 35)
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print(f"{model_name} Accuracy: {accuracy:.2f}")

test_case = X_test.iloc[200].values.reshape(1, -1)
for model_name, model in models.items():
    if model_name == 'XGBoost Regressor':
        print(model.predict(test_case).round())
        continue
    print(model.predict(test_case))

# Save all trained models using pickle
for model_name, model in models.items():
    filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Saved {model_name} to {filename}")
    
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder_gender.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
