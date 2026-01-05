import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the data
print("Loading data... ")
df = pd.read_csv("exercise_data.csv")

X = df.drop("label",axis = 1)
y = df['label']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.3, random_state = 1234)

#training

pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 500))
print("Training the model...")
model = pipeline.fit(X_train,y_train)

#Evaluation

y_prediction = model.predict(X_test)
score = accuracy_score(y_test, y_prediction)
print(f"Model Accuracy : {score * 100:.2f}%")

#Saving
if score > 0.9:
    with open ('body_language.pkl','wb') as f :
        pickle.dump(model,f)
    print("Success!! Model saved as 'body_language.pkl'")
else :
    print("Accuracy is too low. Try collecting more data. ")


