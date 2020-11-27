import kfp.dsl as dsl
import kfp.components as comp
from collections import OrderedDict
from kubernetes import client as k8s_client


def load_data(rok_workspace_example_utf7s6i2o_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/.Forecastielectricity_consumption_of_italy.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("forecasting-electricity-consumption-it-4a9j3",
                                     "load_data",
                                     "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/Forecastielectricity_consumption_of_italy.ipynb")

    import pandas as pd
    import numpy as np
    import math
    import holidays
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import (
        StandardScaler, OneHotEncoder, FunctionTransformer
    )
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import (
        train_test_split, KFold, GridSearchCV, ParameterGrid,
    )
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor, DMatrix, plot_importance
    from xgboost import cv as xgb_cv
    from urllib.parse import urlparse
    import sys
    import os

    def split_train_test(df, split_time):
        df_train = df.loc[df.index < split_time]
        df_test = df.loc[df.index >= split_time]
        return df_train, df_test

    def add_time_features(df):
        cet_index = df.index.tz_convert("CET")
        df["month"] = cet_index.month
        df["weekday"] = cet_index.weekday
        df["hour"] = cet_index.hour
        return df

    def add_holiday_features(df):
        de_holidays = holidays.Germany()
        cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
        df["holiday"] = cet_dates.apply(lambda d: d in de_holidays)
        df["holiday"] = df["holiday"].astype(int)
        return df

    def add_lag_features(df, col="load"):
        for n_hours in range(24, 49):
            shifted_col = df[col].shift(n_hours, "h")
            shifted_col = shifted_col.loc[df.index.min(): df.index.max()]
            label = f"{col}_lag_{n_hours}"
            df[label] = np.nan
            df.loc[shifted_col.index, label] = shifted_col
        return df

    def add_all_features(df, target_col="load"):
        df = df.copy()
        df = add_time_features(df)
        df = add_holiday_features(df)
        df = add_lag_features(df, col=target_col)
        return df

    def fit_prep_pipeline(df):
        cat_features = ["month", "weekday", "hour"]  # categorical features
        bool_features = ["holiday"]  # boolean features
        num_features = [c for c in df.columns
                        if c.startswith("load_lag")]  # numerical features
        prep_pipeline = ColumnTransformer([
            ("cat", OneHotEncoder(), cat_features),
            ("bool", FunctionTransformer(), bool_features),  # identity
            ("num", StandardScaler(), num_features),
        ])
        prep_pipeline = prep_pipeline.fit(df)

        feature_names = []
        one_hot_tf = prep_pipeline.transformers_[0][1]
        for i, cat_feature in enumerate(cat_features):
            categories = one_hot_tf.categories_[i]
            cat_names = [f"{cat_feature}_{c}" for c in categories]
            feature_names += cat_names
        feature_names += (bool_features + num_features)

        return feature_names, prep_pipeline

    def compute_learning_curves(model, X, y, curve_step, verbose=False):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        n_train_obs = X_train.shape[0]
        n_iter = math.ceil(n_train_obs / curve_step)
        train_errors, val_errors, steps = [], [], []
        for i in range(n_iter):
            n_obs = (i+1) * curve_step
            n_obs = min(n_obs, n_train_obs)
            model.fit(X_train[:n_obs], y_train[:n_obs])
            y_train_predict = model.predict(X_train[:n_obs])
            y_val_predict = model.predict(X_val)
            train_mse = mean_squared_error(y_train[:n_obs], y_train_predict)
            val_mse = mean_squared_error(y_val, y_val_predict)
            train_errors.append(train_mse)
            val_errors.append(val_mse)
            steps.append(n_obs)
            if verbose:
                msg = "Iteration {0}/{1}: train_rmse={2:.2f}, val_rmse={3:.2f}".format(
                    i+1, n_iter, np.sqrt(train_mse), np.sqrt(val_mse)
                )
                print(msg)
        return steps, train_errors, val_errors

    def plot_learning_curves(steps, train_errors, val_errors, ax=None, title=""):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))
        train_rmse = np.sqrt(train_errors)
        val_rmse = np.sqrt(val_errors)
        ax.plot(steps, train_rmse, color="tab:blue",
                marker=".", label="training")
        ax.plot(steps, val_rmse, color="tab:orange",
                marker=".", label="validation")
        ylim = (0.8*np.median(train_rmse),
                1.5*np.median(val_rmse))
        ax.set_ylim(ylim)
        ax.set_xlabel("Number of observations")
        ax.set_ylabel("RMSE (MW)")
        ax.set_title(title)
        ax.legend()
        ax.grid()

    def compute_predictions_df(model, X, y):
        y_pred = model.predict(X)
        df = pd.DataFrame(dict(actual=y, prediction=y_pred), index=X.index)
        df["squared_error"] = (df["actual"] - df["prediction"])**2
        return df

    def plot_predictions(pred_df, start=None, end=None):
        _, ax = plt.subplots(1, 1, figsize=(12, 5))
        start = start or pred_df.index.min()
        end = end or pred_df.index.max()
        pred_df.loc[
            (pred_df.index >= start) & (pred_df.index <= end),
            ["actual", "prediction"]
        ].plot.line(ax=ax)
        ax.set_title("Predictions on test set")
        ax.set_ylabel("MW")
        ax.grid()

    STUDY_START_DATE = pd.Timestamp("2015-01-01 00:00", tz="utc")
    STUDY_END_DATE = pd.Timestamp("2020-01-31 23:00", tz="utc")

    dataset_url = os.path.join(os.getcwd(), "../datasets/it.csv")
    it_load = pd.read_csv(dataset_url)
    it_load = it_load.drop(columns="end").set_index("start")
    it_load.index = pd.to_datetime(it_load.index)
    it_load.index.name = "time"
    it_load = it_load.groupby(pd.Grouper(freq="h")).mean()
    it_load = it_load.loc[
        (it_load.index >= STUDY_START_DATE) & (it_load.index <= STUDY_END_DATE), :
    ]
    it_load.info()

    # -----------------------DATA SAVING START---------------------------------
    if "it_load" in locals():
        _kale_resource_save(it_load, os.path.join(
            _kale_data_directory, "it_load"))
    else:
        print("_kale_resource_save: `it_load` not found.")
    if "y_train" in locals():
        _kale_resource_save(y_train, os.path.join(
            _kale_data_directory, "y_train"))
    else:
        print("_kale_resource_save: `y_train` not found.")
    if "df_test" in locals():
        _kale_resource_save(df_test, os.path.join(
            _kale_data_directory, "df_test"))
    else:
        print("_kale_resource_save: `df_test` not found.")
    if "df_train" in locals():
        _kale_resource_save(df_train, os.path.join(
            _kale_data_directory, "df_train"))
    else:
        print("_kale_resource_save: `df_train` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def split_train_test_set(rok_workspace_example_utf7s6i2o_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/.Forecastielectricity_consumption_of_italy.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("forecasting-electricity-consumption-it-4a9j3",
                                     "split_train_test_set",
                                     "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/Forecastielectricity_consumption_of_italy.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "it_load" not in _kale_directory_file_names:
        raise ValueError("it_load" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "it_load"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "it_load" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    it_load = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import pandas as pd
    import numpy as np
    import math
    import holidays
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import (
        StandardScaler, OneHotEncoder, FunctionTransformer
    )
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import (
        train_test_split, KFold, GridSearchCV, ParameterGrid,
    )
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor, DMatrix, plot_importance
    from xgboost import cv as xgb_cv
    from urllib.parse import urlparse
    import sys
    import os

    def split_train_test(df, split_time):
        df_train = df.loc[df.index < split_time]
        df_test = df.loc[df.index >= split_time]
        return df_train, df_test

    def add_time_features(df):
        cet_index = df.index.tz_convert("CET")
        df["month"] = cet_index.month
        df["weekday"] = cet_index.weekday
        df["hour"] = cet_index.hour
        return df

    def add_holiday_features(df):
        de_holidays = holidays.Germany()
        cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
        df["holiday"] = cet_dates.apply(lambda d: d in de_holidays)
        df["holiday"] = df["holiday"].astype(int)
        return df

    def add_lag_features(df, col="load"):
        for n_hours in range(24, 49):
            shifted_col = df[col].shift(n_hours, "h")
            shifted_col = shifted_col.loc[df.index.min(): df.index.max()]
            label = f"{col}_lag_{n_hours}"
            df[label] = np.nan
            df.loc[shifted_col.index, label] = shifted_col
        return df

    def add_all_features(df, target_col="load"):
        df = df.copy()
        df = add_time_features(df)
        df = add_holiday_features(df)
        df = add_lag_features(df, col=target_col)
        return df

    def fit_prep_pipeline(df):
        cat_features = ["month", "weekday", "hour"]  # categorical features
        bool_features = ["holiday"]  # boolean features
        num_features = [c for c in df.columns
                        if c.startswith("load_lag")]  # numerical features
        prep_pipeline = ColumnTransformer([
            ("cat", OneHotEncoder(), cat_features),
            ("bool", FunctionTransformer(), bool_features),  # identity
            ("num", StandardScaler(), num_features),
        ])
        prep_pipeline = prep_pipeline.fit(df)

        feature_names = []
        one_hot_tf = prep_pipeline.transformers_[0][1]
        for i, cat_feature in enumerate(cat_features):
            categories = one_hot_tf.categories_[i]
            cat_names = [f"{cat_feature}_{c}" for c in categories]
            feature_names += cat_names
        feature_names += (bool_features + num_features)

        return feature_names, prep_pipeline

    def compute_learning_curves(model, X, y, curve_step, verbose=False):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        n_train_obs = X_train.shape[0]
        n_iter = math.ceil(n_train_obs / curve_step)
        train_errors, val_errors, steps = [], [], []
        for i in range(n_iter):
            n_obs = (i+1) * curve_step
            n_obs = min(n_obs, n_train_obs)
            model.fit(X_train[:n_obs], y_train[:n_obs])
            y_train_predict = model.predict(X_train[:n_obs])
            y_val_predict = model.predict(X_val)
            train_mse = mean_squared_error(y_train[:n_obs], y_train_predict)
            val_mse = mean_squared_error(y_val, y_val_predict)
            train_errors.append(train_mse)
            val_errors.append(val_mse)
            steps.append(n_obs)
            if verbose:
                msg = "Iteration {0}/{1}: train_rmse={2:.2f}, val_rmse={3:.2f}".format(
                    i+1, n_iter, np.sqrt(train_mse), np.sqrt(val_mse)
                )
                print(msg)
        return steps, train_errors, val_errors

    def plot_learning_curves(steps, train_errors, val_errors, ax=None, title=""):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))
        train_rmse = np.sqrt(train_errors)
        val_rmse = np.sqrt(val_errors)
        ax.plot(steps, train_rmse, color="tab:blue",
                marker=".", label="training")
        ax.plot(steps, val_rmse, color="tab:orange",
                marker=".", label="validation")
        ylim = (0.8*np.median(train_rmse),
                1.5*np.median(val_rmse))
        ax.set_ylim(ylim)
        ax.set_xlabel("Number of observations")
        ax.set_ylabel("RMSE (MW)")
        ax.set_title(title)
        ax.legend()
        ax.grid()

    def compute_predictions_df(model, X, y):
        y_pred = model.predict(X)
        df = pd.DataFrame(dict(actual=y, prediction=y_pred), index=X.index)
        df["squared_error"] = (df["actual"] - df["prediction"])**2
        return df

    def plot_predictions(pred_df, start=None, end=None):
        _, ax = plt.subplots(1, 1, figsize=(12, 5))
        start = start or pred_df.index.min()
        end = end or pred_df.index.max()
        pred_df.loc[
            (pred_df.index >= start) & (pred_df.index <= end),
            ["actual", "prediction"]
        ].plot.line(ax=ax)
        ax.set_title("Predictions on test set")
        ax.set_ylabel("MW")
        ax.grid()

    df_train, df_test = split_train_test(
        it_load, pd.Timestamp("2019-02-01", tz="utc")
    )

    # -----------------------DATA SAVING START---------------------------------
    if "y_train" in locals():
        _kale_resource_save(y_train, os.path.join(
            _kale_data_directory, "y_train"))
    else:
        print("_kale_resource_save: `y_train` not found.")
    if "df_test" in locals():
        _kale_resource_save(df_test, os.path.join(
            _kale_data_directory, "df_test"))
    else:
        print("_kale_resource_save: `df_test` not found.")
    if "df_train" in locals():
        _kale_resource_save(df_train, os.path.join(
            _kale_data_directory, "df_train"))
    else:
        print("_kale_resource_save: `df_train` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def data_preparation(rok_workspace_example_utf7s6i2o_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/.Forecastielectricity_consumption_of_italy.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("forecasting-electricity-consumption-it-4a9j3",
                                     "data_preparation",
                                     "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/Forecastielectricity_consumption_of_italy.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "df_test" not in _kale_directory_file_names:
        raise ValueError("df_test" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "df_test"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "df_test" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    df_test = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "df_train" not in _kale_directory_file_names:
        raise ValueError("df_train" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "df_train"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "df_train" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    df_train = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import pandas as pd
    import numpy as np
    import math
    import holidays
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import (
        StandardScaler, OneHotEncoder, FunctionTransformer
    )
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import (
        train_test_split, KFold, GridSearchCV, ParameterGrid,
    )
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor, DMatrix, plot_importance
    from xgboost import cv as xgb_cv
    from urllib.parse import urlparse
    import sys
    import os

    def split_train_test(df, split_time):
        df_train = df.loc[df.index < split_time]
        df_test = df.loc[df.index >= split_time]
        return df_train, df_test

    def add_time_features(df):
        cet_index = df.index.tz_convert("CET")
        df["month"] = cet_index.month
        df["weekday"] = cet_index.weekday
        df["hour"] = cet_index.hour
        return df

    def add_holiday_features(df):
        de_holidays = holidays.Germany()
        cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
        df["holiday"] = cet_dates.apply(lambda d: d in de_holidays)
        df["holiday"] = df["holiday"].astype(int)
        return df

    def add_lag_features(df, col="load"):
        for n_hours in range(24, 49):
            shifted_col = df[col].shift(n_hours, "h")
            shifted_col = shifted_col.loc[df.index.min(): df.index.max()]
            label = f"{col}_lag_{n_hours}"
            df[label] = np.nan
            df.loc[shifted_col.index, label] = shifted_col
        return df

    def add_all_features(df, target_col="load"):
        df = df.copy()
        df = add_time_features(df)
        df = add_holiday_features(df)
        df = add_lag_features(df, col=target_col)
        return df

    def fit_prep_pipeline(df):
        cat_features = ["month", "weekday", "hour"]  # categorical features
        bool_features = ["holiday"]  # boolean features
        num_features = [c for c in df.columns
                        if c.startswith("load_lag")]  # numerical features
        prep_pipeline = ColumnTransformer([
            ("cat", OneHotEncoder(), cat_features),
            ("bool", FunctionTransformer(), bool_features),  # identity
            ("num", StandardScaler(), num_features),
        ])
        prep_pipeline = prep_pipeline.fit(df)

        feature_names = []
        one_hot_tf = prep_pipeline.transformers_[0][1]
        for i, cat_feature in enumerate(cat_features):
            categories = one_hot_tf.categories_[i]
            cat_names = [f"{cat_feature}_{c}" for c in categories]
            feature_names += cat_names
        feature_names += (bool_features + num_features)

        return feature_names, prep_pipeline

    def compute_learning_curves(model, X, y, curve_step, verbose=False):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        n_train_obs = X_train.shape[0]
        n_iter = math.ceil(n_train_obs / curve_step)
        train_errors, val_errors, steps = [], [], []
        for i in range(n_iter):
            n_obs = (i+1) * curve_step
            n_obs = min(n_obs, n_train_obs)
            model.fit(X_train[:n_obs], y_train[:n_obs])
            y_train_predict = model.predict(X_train[:n_obs])
            y_val_predict = model.predict(X_val)
            train_mse = mean_squared_error(y_train[:n_obs], y_train_predict)
            val_mse = mean_squared_error(y_val, y_val_predict)
            train_errors.append(train_mse)
            val_errors.append(val_mse)
            steps.append(n_obs)
            if verbose:
                msg = "Iteration {0}/{1}: train_rmse={2:.2f}, val_rmse={3:.2f}".format(
                    i+1, n_iter, np.sqrt(train_mse), np.sqrt(val_mse)
                )
                print(msg)
        return steps, train_errors, val_errors

    def plot_learning_curves(steps, train_errors, val_errors, ax=None, title=""):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))
        train_rmse = np.sqrt(train_errors)
        val_rmse = np.sqrt(val_errors)
        ax.plot(steps, train_rmse, color="tab:blue",
                marker=".", label="training")
        ax.plot(steps, val_rmse, color="tab:orange",
                marker=".", label="validation")
        ylim = (0.8*np.median(train_rmse),
                1.5*np.median(val_rmse))
        ax.set_ylim(ylim)
        ax.set_xlabel("Number of observations")
        ax.set_ylabel("RMSE (MW)")
        ax.set_title(title)
        ax.legend()
        ax.grid()

    def compute_predictions_df(model, X, y):
        y_pred = model.predict(X)
        df = pd.DataFrame(dict(actual=y, prediction=y_pred), index=X.index)
        df["squared_error"] = (df["actual"] - df["prediction"])**2
        return df

    def plot_predictions(pred_df, start=None, end=None):
        _, ax = plt.subplots(1, 1, figsize=(12, 5))
        start = start or pred_df.index.min()
        end = end or pred_df.index.max()
        pred_df.loc[
            (pred_df.index >= start) & (pred_df.index <= end),
            ["actual", "prediction"]
        ].plot.line(ax=ax)
        ax.set_title("Predictions on test set")
        ax.set_ylabel("MW")
        ax.grid()

    df_train.loc[df_train["load"].isna(), :].index
    df_train = add_all_features(df_train).dropna()
    df_test = add_all_features(df_test).dropna()
    df_train.info()

    target_col = "load"
    X_train = df_train.drop(columns=target_col)
    y_train = df_train.loc[:, target_col]
    X_test = df_test.drop(columns=target_col)
    y_test = df_test.loc[:, target_col]
    feature_names, prep_pipeline = fit_prep_pipeline(X_train)

    X_train_prep = prep_pipeline.transform(X_train)
    X_train_prep = pd.DataFrame(
        X_train_prep, columns=feature_names, index=df_train.index)
    X_test_prep = prep_pipeline.transform(X_test)
    X_test_prep = pd.DataFrame(
        X_test_prep, columns=feature_names, index=df_test.index)

    X_train_prep.info()

    # -----------------------DATA SAVING START---------------------------------
    if "y_train" in locals():
        _kale_resource_save(y_train, os.path.join(
            _kale_data_directory, "y_train"))
    else:
        print("_kale_resource_save: `y_train` not found.")
    if "X_train_prep" in locals():
        _kale_resource_save(X_train_prep, os.path.join(
            _kale_data_directory, "X_train_prep"))
    else:
        print("_kale_resource_save: `X_train_prep` not found.")
    if "X_test_prep" in locals():
        _kale_resource_save(X_test_prep, os.path.join(
            _kale_data_directory, "X_test_prep"))
    else:
        print("_kale_resource_save: `X_test_prep` not found.")
    if "y_test" in locals():
        _kale_resource_save(y_test, os.path.join(
            _kale_data_directory, "y_test"))
    else:
        print("_kale_resource_save: `y_test` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def train_model(rok_workspace_example_utf7s6i2o_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/.Forecastielectricity_consumption_of_italy.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("forecasting-electricity-consumption-it-4a9j3",
                                     "train_model",
                                     "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/Forecastielectricity_consumption_of_italy.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "y_train" not in _kale_directory_file_names:
        raise ValueError("y_train" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "y_train"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "y_train" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    y_train = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "X_train_prep" not in _kale_directory_file_names:
        raise ValueError("X_train_prep" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "X_train_prep"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "X_train_prep" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    X_train_prep = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import pandas as pd
    import numpy as np
    import math
    import holidays
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import (
        StandardScaler, OneHotEncoder, FunctionTransformer
    )
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import (
        train_test_split, KFold, GridSearchCV, ParameterGrid,
    )
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor, DMatrix, plot_importance
    from xgboost import cv as xgb_cv
    from urllib.parse import urlparse
    import sys
    import os

    def split_train_test(df, split_time):
        df_train = df.loc[df.index < split_time]
        df_test = df.loc[df.index >= split_time]
        return df_train, df_test

    def add_time_features(df):
        cet_index = df.index.tz_convert("CET")
        df["month"] = cet_index.month
        df["weekday"] = cet_index.weekday
        df["hour"] = cet_index.hour
        return df

    def add_holiday_features(df):
        de_holidays = holidays.Germany()
        cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
        df["holiday"] = cet_dates.apply(lambda d: d in de_holidays)
        df["holiday"] = df["holiday"].astype(int)
        return df

    def add_lag_features(df, col="load"):
        for n_hours in range(24, 49):
            shifted_col = df[col].shift(n_hours, "h")
            shifted_col = shifted_col.loc[df.index.min(): df.index.max()]
            label = f"{col}_lag_{n_hours}"
            df[label] = np.nan
            df.loc[shifted_col.index, label] = shifted_col
        return df

    def add_all_features(df, target_col="load"):
        df = df.copy()
        df = add_time_features(df)
        df = add_holiday_features(df)
        df = add_lag_features(df, col=target_col)
        return df

    def fit_prep_pipeline(df):
        cat_features = ["month", "weekday", "hour"]  # categorical features
        bool_features = ["holiday"]  # boolean features
        num_features = [c for c in df.columns
                        if c.startswith("load_lag")]  # numerical features
        prep_pipeline = ColumnTransformer([
            ("cat", OneHotEncoder(), cat_features),
            ("bool", FunctionTransformer(), bool_features),  # identity
            ("num", StandardScaler(), num_features),
        ])
        prep_pipeline = prep_pipeline.fit(df)

        feature_names = []
        one_hot_tf = prep_pipeline.transformers_[0][1]
        for i, cat_feature in enumerate(cat_features):
            categories = one_hot_tf.categories_[i]
            cat_names = [f"{cat_feature}_{c}" for c in categories]
            feature_names += cat_names
        feature_names += (bool_features + num_features)

        return feature_names, prep_pipeline

    def compute_learning_curves(model, X, y, curve_step, verbose=False):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        n_train_obs = X_train.shape[0]
        n_iter = math.ceil(n_train_obs / curve_step)
        train_errors, val_errors, steps = [], [], []
        for i in range(n_iter):
            n_obs = (i+1) * curve_step
            n_obs = min(n_obs, n_train_obs)
            model.fit(X_train[:n_obs], y_train[:n_obs])
            y_train_predict = model.predict(X_train[:n_obs])
            y_val_predict = model.predict(X_val)
            train_mse = mean_squared_error(y_train[:n_obs], y_train_predict)
            val_mse = mean_squared_error(y_val, y_val_predict)
            train_errors.append(train_mse)
            val_errors.append(val_mse)
            steps.append(n_obs)
            if verbose:
                msg = "Iteration {0}/{1}: train_rmse={2:.2f}, val_rmse={3:.2f}".format(
                    i+1, n_iter, np.sqrt(train_mse), np.sqrt(val_mse)
                )
                print(msg)
        return steps, train_errors, val_errors

    def plot_learning_curves(steps, train_errors, val_errors, ax=None, title=""):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))
        train_rmse = np.sqrt(train_errors)
        val_rmse = np.sqrt(val_errors)
        ax.plot(steps, train_rmse, color="tab:blue",
                marker=".", label="training")
        ax.plot(steps, val_rmse, color="tab:orange",
                marker=".", label="validation")
        ylim = (0.8*np.median(train_rmse),
                1.5*np.median(val_rmse))
        ax.set_ylim(ylim)
        ax.set_xlabel("Number of observations")
        ax.set_ylabel("RMSE (MW)")
        ax.set_title(title)
        ax.legend()
        ax.grid()

    def compute_predictions_df(model, X, y):
        y_pred = model.predict(X)
        df = pd.DataFrame(dict(actual=y, prediction=y_pred), index=X.index)
        df["squared_error"] = (df["actual"] - df["prediction"])**2
        return df

    def plot_predictions(pred_df, start=None, end=None):
        _, ax = plt.subplots(1, 1, figsize=(12, 5))
        start = start or pred_df.index.min()
        end = end or pred_df.index.max()
        pred_df.loc[
            (pred_df.index >= start) & (pred_df.index <= end),
            ["actual", "prediction"]
        ].plot.line(ax=ax)
        ax.set_title("Predictions on test set")
        ax.set_ylabel("MW")
        ax.grid()

    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    xgb_steps, xgb_train_mse, xgb_val_mse = compute_learning_curves(
        xgb_model, X_train_prep, y_train, 1000, verbose=True
    )
    xgb_model.fit(X_train_prep, y=y_train)

    # -----------------------DATA SAVING START---------------------------------
    if "xgb_model" in locals():
        _kale_resource_save(xgb_model, os.path.join(
            _kale_data_directory, "xgb_model"))
    else:
        print("_kale_resource_save: `xgb_model` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def predict(rok_workspace_example_utf7s6i2o_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/.Forecastielectricity_consumption_of_italy.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("forecasting-electricity-consumption-it-4a9j3",
                                     "predict",
                                     "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/Forecastielectricity_consumption_of_italy.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "xgb_model" not in _kale_directory_file_names:
        raise ValueError("xgb_model" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "xgb_model"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "xgb_model" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    xgb_model = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "X_test_prep" not in _kale_directory_file_names:
        raise ValueError("X_test_prep" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "X_test_prep"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "X_test_prep" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    X_test_prep = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "y_test" not in _kale_directory_file_names:
        raise ValueError("y_test" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "y_test"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "y_test" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    y_test = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import pandas as pd
    import numpy as np
    import math
    import holidays
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import (
        StandardScaler, OneHotEncoder, FunctionTransformer
    )
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import (
        train_test_split, KFold, GridSearchCV, ParameterGrid,
    )
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor, DMatrix, plot_importance
    from xgboost import cv as xgb_cv
    from urllib.parse import urlparse
    import sys
    import os

    def split_train_test(df, split_time):
        df_train = df.loc[df.index < split_time]
        df_test = df.loc[df.index >= split_time]
        return df_train, df_test

    def add_time_features(df):
        cet_index = df.index.tz_convert("CET")
        df["month"] = cet_index.month
        df["weekday"] = cet_index.weekday
        df["hour"] = cet_index.hour
        return df

    def add_holiday_features(df):
        de_holidays = holidays.Germany()
        cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
        df["holiday"] = cet_dates.apply(lambda d: d in de_holidays)
        df["holiday"] = df["holiday"].astype(int)
        return df

    def add_lag_features(df, col="load"):
        for n_hours in range(24, 49):
            shifted_col = df[col].shift(n_hours, "h")
            shifted_col = shifted_col.loc[df.index.min(): df.index.max()]
            label = f"{col}_lag_{n_hours}"
            df[label] = np.nan
            df.loc[shifted_col.index, label] = shifted_col
        return df

    def add_all_features(df, target_col="load"):
        df = df.copy()
        df = add_time_features(df)
        df = add_holiday_features(df)
        df = add_lag_features(df, col=target_col)
        return df

    def fit_prep_pipeline(df):
        cat_features = ["month", "weekday", "hour"]  # categorical features
        bool_features = ["holiday"]  # boolean features
        num_features = [c for c in df.columns
                        if c.startswith("load_lag")]  # numerical features
        prep_pipeline = ColumnTransformer([
            ("cat", OneHotEncoder(), cat_features),
            ("bool", FunctionTransformer(), bool_features),  # identity
            ("num", StandardScaler(), num_features),
        ])
        prep_pipeline = prep_pipeline.fit(df)

        feature_names = []
        one_hot_tf = prep_pipeline.transformers_[0][1]
        for i, cat_feature in enumerate(cat_features):
            categories = one_hot_tf.categories_[i]
            cat_names = [f"{cat_feature}_{c}" for c in categories]
            feature_names += cat_names
        feature_names += (bool_features + num_features)

        return feature_names, prep_pipeline

    def compute_learning_curves(model, X, y, curve_step, verbose=False):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        n_train_obs = X_train.shape[0]
        n_iter = math.ceil(n_train_obs / curve_step)
        train_errors, val_errors, steps = [], [], []
        for i in range(n_iter):
            n_obs = (i+1) * curve_step
            n_obs = min(n_obs, n_train_obs)
            model.fit(X_train[:n_obs], y_train[:n_obs])
            y_train_predict = model.predict(X_train[:n_obs])
            y_val_predict = model.predict(X_val)
            train_mse = mean_squared_error(y_train[:n_obs], y_train_predict)
            val_mse = mean_squared_error(y_val, y_val_predict)
            train_errors.append(train_mse)
            val_errors.append(val_mse)
            steps.append(n_obs)
            if verbose:
                msg = "Iteration {0}/{1}: train_rmse={2:.2f}, val_rmse={3:.2f}".format(
                    i+1, n_iter, np.sqrt(train_mse), np.sqrt(val_mse)
                )
                print(msg)
        return steps, train_errors, val_errors

    def plot_learning_curves(steps, train_errors, val_errors, ax=None, title=""):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 4))
        train_rmse = np.sqrt(train_errors)
        val_rmse = np.sqrt(val_errors)
        ax.plot(steps, train_rmse, color="tab:blue",
                marker=".", label="training")
        ax.plot(steps, val_rmse, color="tab:orange",
                marker=".", label="validation")
        ylim = (0.8*np.median(train_rmse),
                1.5*np.median(val_rmse))
        ax.set_ylim(ylim)
        ax.set_xlabel("Number of observations")
        ax.set_ylabel("RMSE (MW)")
        ax.set_title(title)
        ax.legend()
        ax.grid()

    def compute_predictions_df(model, X, y):
        y_pred = model.predict(X)
        df = pd.DataFrame(dict(actual=y, prediction=y_pred), index=X.index)
        df["squared_error"] = (df["actual"] - df["prediction"])**2
        return df

    def plot_predictions(pred_df, start=None, end=None):
        _, ax = plt.subplots(1, 1, figsize=(12, 5))
        start = start or pred_df.index.min()
        end = end or pred_df.index.max()
        pred_df.loc[
            (pred_df.index >= start) & (pred_df.index <= end),
            ["actual", "prediction"]
        ].plot.line(ax=ax)
        ax.set_title("Predictions on test set")
        ax.set_ylabel("MW")
        ax.grid()

    pred_df = compute_predictions_df(
        xgb_model, X_test_prep, y_test
    )
    pred_df.head()
    plot_predictions(pred_df)


def final_auto_snapshot(rok_workspace_example_utf7s6i2o_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/.Forecastielectricity_consumption_of_italy.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("forecasting-electricity-consumption-it-4a9j3",
                                     "final_auto_snapshot",
                                     "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale/Forecastielectricity_consumption_of_italy.ipynb")


load_data_op = comp.func_to_container_op(
    load_data, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


split_train_test_set_op = comp.func_to_container_op(
    split_train_test_set, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


data_preparation_op = comp.func_to_container_op(
    data_preparation, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


train_model_op = comp.func_to_container_op(
    train_model, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


predict_op = comp.func_to_container_op(
    predict, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


final_auto_snapshot_op = comp.func_to_container_op(
    final_auto_snapshot, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


@dsl.pipeline(
    name='forecasting-electricity-consumption-it-4a9j3',
    description='Forecasting hourly electricity consumption in Italy'
)
def auto_generated_pipeline(rok_workspace_example_utf7s6i2o_url='http://rok.rok.svc.cluster.local/swift/v1/d88db09f-bf61-41d1-b3d6-515c2bd2f681/notebooks/example-0_workspace-example-utf7s6i2o?version=fd2b1906-3329-475b-99da-a79f877e3ee1'):
    pvolumes_dict = OrderedDict()

    annotations = {'rok/origin': 'http://rok.rok.svc.cluster.local/swift/v1/d88db09f-bf61-41d1-b3d6-515c2bd2f681/notebooks/example-0_workspace-example-utf7s6i2o?version=fd2b1906-3329-475b-99da-a79f877e3ee1'}

    annotations['rok/origin'] = rok_workspace_example_utf7s6i2o_url

    vop1 = dsl.VolumeOp(
        name='create-volume-1',
        resource_name='workspace-example-utf7s6i2o',
        annotations=annotations,
        size='5Gi'
    )
    volume = vop1.volume

    pvolumes_dict['/home/jovyan'] = volume

    load_data_task = load_data_op(rok_workspace_example_utf7s6i2o_url)\
        .add_pvolumes(pvolumes_dict)\
        .after()
    load_data_task.container.working_dir = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale"
    load_data_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    load_data_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    split_train_test_set_task = split_train_test_set_op(rok_workspace_example_utf7s6i2o_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_data_task)
    split_train_test_set_task.container.working_dir = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale"
    split_train_test_set_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    split_train_test_set_task.output_artifact_paths.update(
        mlpipeline_ui_metadata)

    data_preparation_task = data_preparation_op(rok_workspace_example_utf7s6i2o_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(split_train_test_set_task)
    data_preparation_task.container.working_dir = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale"
    data_preparation_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    data_preparation_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    train_model_task = train_model_op(rok_workspace_example_utf7s6i2o_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(data_preparation_task)
    train_model_task.container.working_dir = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale"
    train_model_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    train_model_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    predict_task = predict_op(rok_workspace_example_utf7s6i2o_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(train_model_task)
    predict_task.container.working_dir = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale"
    predict_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    predict_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    final_auto_snapshot_task = final_auto_snapshot_op(rok_workspace_example_utf7s6i2o_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(predict_task)
    final_auto_snapshot_task.container.working_dir = "/home/jovyan/kale/mlflow_demo/forecasting-electricity-consumption-of-italy-with-kale"
    final_auto_snapshot_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    final_auto_snapshot_task.output_artifact_paths.update(
        mlpipeline_ui_metadata)


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('First experiment')

    # Submit a pipeline run
    run_name = 'forecasting-electricity-consumption-it-4a9j3_run'
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
