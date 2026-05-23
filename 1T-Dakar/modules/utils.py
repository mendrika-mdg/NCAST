import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def choose_header(n, location, lead_time):

    input_headers = ['year', 'month', 'day', 'hour', 'minute']

    for prefix in ['lat', 'lon', 'wp', 'size', 'd', 'mask']:
        input_headers.extend([f'{prefix}{i}' for i in range(1, n + 1)])

    target_header = f'Cb_{location}_t{lead_time}'

    return input_headers, target_header


def prepare_data(lead_time, location="dakar", n_cores=3):

    test = pd.read_csv(
        f"/home/users/mendrika/NCAST/Data/{location.capitalize()}/data-test-{location}.csv"
    )

    train_full = pd.read_csv(
        f"/home/users/mendrika/NCAST/Data/{location.capitalize()}/data-train-{location}.csv"
    )

    val = train_full[train_full["year"] == 2019].copy()
    train = train_full[train_full["year"] != 2019].copy()

    input_headers, target_header = choose_header(
        n=n_cores,
        location=location,
        lead_time=lead_time
    )

    X_train = train[input_headers].copy()
    X_val   = val[input_headers].copy()
    X_test  = test[input_headers].copy()

    y_train = train[target_header].copy()
    y_val   = val[target_header].copy()
    y_test  = test[target_header].copy()

    cols_to_log = []

    for prefix in ['size', 'wp', 'd']:
        cols_to_log.extend([f"{prefix}{i}" for i in range(1, n_cores + 1)])

    for col in cols_to_log:

        X_train[col] = np.log1p(X_train[col])
        X_val[col] = np.log1p(X_val[col])
        X_test[col] = np.log1p(X_test[col])

    core_groups = {}

    for i in range(1, n_cores + 1):

        core_groups[i] = [
            f'lat{i}',
            f'lon{i}',
            f'wp{i}',
            f'size{i}',
            f'd{i}'
        ]

    X_train_scaled = X_train.copy()
    X_val_scaled   = X_val.copy()
    X_test_scaled  = X_test.copy()

    scalers = {}

    for i, cols in core_groups.items():

        mask = X_train[f'mask{i}'] == 1

        scaler = StandardScaler()

        scaler.fit(X_train.loc[mask, cols])

        scalers[i] = scaler

        X_train_scaled.loc[:, cols] = scaler.transform(X_train[cols])
        X_val_scaled.loc[:, cols] = scaler.transform(X_val[cols])
        X_test_scaled.loc[:, cols] = scaler.transform(X_test[cols])

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scalers
    )