import os
import joblib
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import hamming_loss

project_dir = os.getcwd()
data_dir = os.path.join(project_dir, "data")
model_dir = os.path.join(project_dir, "model")

tqdm.pandas()

X_train = joblib.load(f"{data_dir}/x_train.pkl")
X_val = joblib.load(f"{data_dir}/x_val.pkl")
y_train = joblib.load(f"{data_dir}/y_train.pkl")
y_val = joblib.load(f"{data_dir}/y_val.pkl")
y_classes = joblib.load(f"{data_dir}/y_classes.pkl")


xgb_classifier = XGBClassifier(max_depth=5,
                               eta=0.2,
                               gamma=4,
                               min_child_weight=6,
                               subsample=0.8,
                               early_stopping_rounds=10,
                               num_round=200,
                               n_jobs=-1)

clf = OneVsRestClassifier(xgb_classifier)
clf.fit(X_train, y_train)

joblib.dump(clf, f"{model_dir}/one_vs_rest_classifier.pkl")

y_pred = clf.predict(X_val)

joblib.dump(y_pred, f"{model_dir}/y_pred.pkl")

precision, recall, fscore, support = score(y_val, y_pred)

print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"fscore: {fscore}")


hamming = []
for i, (test, pred) in enumerate(zip(y_val.T, y_pred.T)):
    hamming.append(hamming_loss(test, pred))

metric_df = pd.DataFrame(data=[precision, recall, fscore, hamming],
                         index=["Precision", "Recall", "F-1 score", "Hamming loss"],
                         columns=y_classes)
