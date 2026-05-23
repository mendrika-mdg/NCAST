import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(1, "/home/users/mendrika/NCAST/1T-Dakar/modules")
from utils import prepare_data


seed = 42

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


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

    "dropout": [
        0.0,
        0.1,
        0.2,
        0.3
    ],

    "learning_rate": [
        1e-3,
        5e-4,
        1e-4
    ],

    "batch_size": [
        128,
        256
    ],

    "architecture": [
        [64, 16, 8],
        [128, 64, 32, 16]
    ]
}


best_val_auc = -999
best_model = None
best_params = None


for dropout in grid["dropout"]:

    for learning_rate in grid["learning_rate"]:

        for batch_size in grid["batch_size"]:

            for architecture in grid["architecture"]:

                tf.keras.backend.clear_session()

                model = tf.keras.models.Sequential()

                model.add(
                    tf.keras.layers.Input(
                        shape=(X_train.shape[1],)
                    )
                )

                for units in architecture[:-1]:

                    model.add(
                        tf.keras.layers.Dense(
                            units,
                            activation="relu"
                        )
                    )

                    model.add(
                        tf.keras.layers.Dropout(dropout)
                    )

                model.add(
                    tf.keras.layers.Dense(
                        architecture[-1],
                        activation="relu"
                    )
                )

                model.add(
                    tf.keras.layers.Dense(
                        1,
                        activation="sigmoid"
                    )
                )


                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=learning_rate
                    ),

                    loss=tf.keras.losses.BinaryCrossentropy(),

                    metrics=[
                        tf.keras.metrics.AUC(name="auc")
                    ]
                )


                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc",
                    mode="max",
                    patience=10,
                    restore_best_weights=True
                )


                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[early_stopping]
                )


                y_val_prob = model.predict(
                    X_val,
                    verbose=0
                ).ravel()


                val_auc = roc_auc_score(
                    y_val,
                    y_val_prob
                )


                print(
                    f"dropout={dropout}, "
                    f"learning_rate={learning_rate}, "
                    f"batch_size={batch_size}, "
                    f"architecture={architecture}, "
                    f"val_auc={val_auc:.4f}"
                )


                if val_auc > best_val_auc:

                    best_val_auc = val_auc
                    best_model = model

                    best_params = {
                        "dropout": dropout,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "architecture": architecture
                    }


y_train_prob = best_model.predict(
    X_train,
    verbose=0
).ravel()

y_val_prob = best_model.predict(
    X_val,
    verbose=0
).ravel()

y_test_prob = best_model.predict(
    X_test,
    verbose=0
).ravel()


train_auc = roc_auc_score(y_train, y_train_prob)
val_auc = roc_auc_score(y_val, y_val_prob)
test_auc = roc_auc_score(y_test, y_test_prob)

train_brier = brier_score_loss(y_train, y_train_prob)
val_brier = brier_score_loss(y_val, y_val_prob)
test_brier = brier_score_loss(y_test, y_test_prob)


results = pd.DataFrame([{
    "model": "MLP",
    "lead_time": lead_time,
    "dropout": best_params["dropout"],
    "learning_rate": best_params["learning_rate"],
    "batch_size": best_params["batch_size"],
    "architecture": str(best_params["architecture"]),
    "train_auc": train_auc,
    "val_auc": val_auc,
    "test_auc": test_auc,
    "train_brier": train_brier,
    "val_brier": val_brier,
    "test_brier": test_brier
}])


output_dir = "/home/users/mendrika/NCAST/Output/Optimisation/Dakar/MLP"

os.makedirs(output_dir, exist_ok=True)

results.to_csv(
    f"{output_dir}/mlp_leadtime_{lead_time}h.csv",
    index=False
)
