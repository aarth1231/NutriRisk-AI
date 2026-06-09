import pandas as pd
df=pd.read_csv("diabetes1.csv")
df.head()
y = df['Outcome']  
x = df.drop(columns=['Outcome'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42,stratify=y)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
numerical_cols = [
    'Pregnancies', 'Glucose', 'BloodPressure', 
    'SkinThickness', 'Insulin', 'BMI', 
    'DiabetesPedigreeFunction', 'Age'
]
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, numerical_cols)
])
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
scale = neg / pos
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
 
xgb_model = XGBClassifier(
    n_estimators=200,
    scale_pos_weight=scale,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    subsample=0.8,               
    colsample_bytree=0.8,
)
 
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    random_state=42,
)
from sklearn.ensemble import VotingClassifier
ensemble_voter = VotingClassifier(
    estimators=[("rf", rf_model), ("xgb", xgb_model), ("lr", lr_model),("gb", gb_model),],
    voting="soft",weights=[2, 3, 1, 2],)
 
main_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),("ensemble", ensemble_voter),
])
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
param_distributions = {
    "ensemble__rf__n_estimators":[100, 200, 300],
    "ensemble__rf__max_depth": [10, 20, None],
    "ensemble__xgb__n_estimators":[100, 200, 300],
    "ensemble__xgb__max_depth":[3, 4, 5, 6],
    "ensemble__xgb__learning_rate": [0.05, 0.1],
    "ensemble__lr__C": [0.1, 1.0, 10.0],
    "ensemble__gb__n_estimators":[100, 200],
    "ensemble__gb__learning_rate": [0.05, 0.1],
    "ensemble__gb__max_depth": [3, 4, 5],
}
 
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
search = RandomizedSearchCV(
    estimator=main_pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    scoring="roc_auc",
    cv=cv_strategy,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
 
search.fit(X_train, y_train)
best_pipeline = search.best_estimator_
 
from sklearn.metrics import classification_report, roc_auc_score

y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print(f"ROC-AUC: {roc_auc_score(y_test, y_proba) * 100:.2f}%")
print(classification_report(y_test, y_pred))
import joblib
joblib.dump(best_pipeline, "NutriRisk_AI.pkl" )

