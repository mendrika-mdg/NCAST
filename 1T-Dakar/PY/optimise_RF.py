import os
import sys
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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
        4,
        6,
        8,
        12,
        None
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "class_weight": [
        None,
        "balanced"
    ]
}


best_val_auc = -999
best_model = None
best_params = None


for n_estimators in grid["n_estimators"]:

    for max_depth in grid["max_depth"]:

        for min_samples_leaf in grid["min_samples_leaf"]:

            for class_weight in grid["class_weight"]:

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)

                y_val_prob = model.predict_proba(X_val)[:, 1]

                val_auc = roc_auc_score(y_val, y_val_prob)

                print(
                    f"n_estimators={n_estimators}, "
                    f"max_depth={max_depth}, "
                    f"min_samples_leaf={min_samples_leaf}, "
                    f"class_weight={class_weight}, "
                    f"val_auc={val_auc:.4f}"
                )

                if val_auc > best_val_auc:

                    best_val_auc = val_auc
                    best_model = model

                    best_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "class_weight": class_weight
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
    "model": "RF",
    "lead_time": lead_time,
    "n_estimators": best_params["n_estimators"],
    "max_depth": best_params["max_depth"],
    "min_samples_leaf": best_params["min_samples_leaf"],
    "class_weight": best_params["class_weight"],
    "train_auc": train_auc,
    "val_auc": val_auc,
    "test_auc": test_auc,
    "train_brier": train_brier,
    "val_brier": val_brier,
    "test_brier": test_brier
}])


output_dir = "/home/users/mendrika/NCAST/Output/Optimisation/Dakar/RF"

os.makedirs(output_dir, exist_ok=True)

results.to_csv(
    f"{output_dir}/rf_leadtime_{lead_time}h.csv",
    index=False
)
