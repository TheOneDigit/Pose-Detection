import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_XGBoost_classify(df: pd.DataFrame):
  X = df.drop(['label', 'image_name'], axis=1)  # Assuming 'label' is the column containing the target variable
  y = df['label'].map({'cutting': 0, 'non_cutting': 1})  # Convert labels to 0 and 1
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
  model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy}")

  # Save the trained model
  model.save_model("model_weights.xgb")

  return "Trained Model Saved Successfully"