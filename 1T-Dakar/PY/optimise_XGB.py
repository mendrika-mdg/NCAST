import os
import sys
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(1, "/home/users/mendrika/NCAST/1T-Dakar/modules")
from utils import prepare_data


lead_time = int(sys.argv[1])


(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    scalers
) = prepare_data(lead_time=lead_time)


grid = {

    "n_estimators": [
        100,
        300,
        500
    ],

    "max_depth": [
        3,
        5,
        7
    ],

    "learning_rate": [
        0.01,
        0.05,
        0.1
    ],

    "subsample": [
        0.8,
        1.0
    ],

    "colsample_bytree": [
        0.8,
        1.0
    ]
}


best_val_auc = -999
best_model = None
best_params = None


for n_estimators in grid["n_estimators"]:

    for max_depth in grid["max_depth"]:

        for learning_rate in grid["learning_rate"]:

            for subsample in grid["subsample"]:

                for colsample_bytree in grid["colsample_bytree"]:

                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1
                    )

                    model.fit(X_train, y_train)

                    y_val_prob = model.predict_proba(X_val)[:, 1]

                    val_auc = roc_auc_score(y_val, y_val_prob)

                    print(
                        f"n_estimators={n_estimators}, "
                        f"max_depth={max_depth}, "
                        f"learning_rate={learning_rate}, "
                        f"subsample={subsample}, "
                        f"colsample_bytree={colsample_bytree}, "
                        f"val_auc={val_auc:.4f}"
                    )

                    if val_auc > best_val_auc:

                        best_val_auc = val_auc
                        best_model = model

                        best_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                            "colsample_bytree": colsample_bytree
                        }


y_train_prob = best_model.predict_proba(X_train)[:, 1]
y_val_prob = best_model.predict_proba(X_val)[:, 1]
y_test_prob = best_model.predict_proba(X_test)[:, 1]


train_auc = roc_auc_score(y_train, y_train_prob)
val_auc = roc_auc_score(y_val, y_val_prob)
test_auc = roc_auc_score(y_test, y_test_prob)

train_brier = brier_score_loss(y_train, y_train_prob)
val_brier = brier_score_loss(y_val, y_val_prob)
test_brier = brier_score_loss(y_test, y_test_prob)


results = pd.DataFrame([{
    "model": "XGB",
    "lead_time": lead_time,
    "n_estimators": best_params["n_estimators"],
    "max_depth": best_params["max_depth"],
    "learning_rate": best_params["learning_rate"],
    "subsample": best_params["subsample"],
    "colsample_bytree": best_params["colsample_bytree"],
    "train_auc": train_auc,
    "val_auc": val_auc,
    "test_auc": test_auc,
    "train_brier": train_brier,
    "val_brier": val_brier,
    "test_brier": test_brier
}])


output_dir = "/home/users/mendrika/NCAST/Output/Optimisation/Dakar/XGB"

os.makedirs(output_dir, exist_ok=True)

results.to_csv(
    f"{output_dir}/xgb_leadtime_{lead_time}h.csv",
    index=False
)
