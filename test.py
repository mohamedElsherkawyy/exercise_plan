import pickle
from pymongo import MongoClient

# --- STEP 1: Load your model ---
with open('xgboost_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- STEP 2: Prepare your input ---
# Example input: Replace this with your real input data
X_new = [[-0.24582361589772553, 0.8352847858659286, -0.5362980713084466, -1.1084514427897492, 1, -0.925599266051832]]  # Example format

# --- STEP 3: Predict the class ---
predicted_class = model.predict(X_new)[0]  # Assuming it outputs one number
predicted_class = int(predicted_class)  # Simpler and safer  # Ensure it's a native Python int
print(f"Predicted class: {predicted_class}")

# --- STEP 4: Connect to MongoDB ---
client = MongoClient('mongodb+srv://mohamedmotaz:oVxmDmRXPhIwyyaW@cluster0.iia6ivy.mongodb.net/')
db = client['liftology']
collection = db['exercise_plans']

# --- STEP 5: Query the content ---
result = collection.find_one({'plan': predicted_class})

# --- STEP 5: Query all matching documents ---
results = collection.find({'plan': predicted_class})

found_any = False
for doc in results:
    print(f"Document: {doc}")
    found_any = True

if not found_any:
    print(f"No documents found for class {predicted_class}.")


# --- STEP 6: Close connection ---
client.close()
