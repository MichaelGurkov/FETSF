import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import root_mean_squared_error

# Import data
passengers_df = pd.read_csv("C:\\Users\\Home\\Documents\\FETSF\\data\\example_air_passengers.csv",
                            index_col=["ds"], parse_dates=["ds"])

retail_df = pd.read_csv("C:\\Users\\Home\\Documents\\FETSF\\data\\example_retail_sales.csv", index_col=False)

# Split data

holdout_size = 12 * 6

train_set = passengers_df.iloc[:-holdout_size].copy()

test_set = passengers_df.iloc[-holdout_size:].copy()

def preprocess_data(temp_df):
    temp_df["trend"] = np.round((temp_df.index - pd.to_datetime("1949-01-01")) / np.timedelta64(4, "W"))
    
    temp_df["trend2"] = temp_df["trend"] ** 2
    
    temp_df["trend3"] = temp_df["trend"] ** 3 
      
    X_mat = temp_df.copy().drop("y", axis = 1)
    
    y_vec = temp_df.copy()["y"]
    
    return([X_mat, y_vec])
  
min_max_scale = MinMaxScaler()

X_train, y_train = preprocess_data(train_set.copy())

X_train = min_max_scale.fit_transform(X_train)

X_test, y_test= preprocess_data(test_set.copy())

X_test = min_max_scale.transform(X_test)


# Fit model

def get_predictions(model, X_train, y_train, X_test):

    model.fit(X_train.copy(), y_train.copy())

    y_train_pred = model.predict(X_train.copy())

    y_test_pred = model.predict(X_test.copy())
    
    return([y_train_pred, y_test_pred])


lin_reg_train_pred, lin_reg_test_pred = get_predictions(LinearRegression(), X_train, y_train, X_test)


# Visualization

def make_plot_df(train_set, y_train_pred, test_set, y_test_pred):
    
    train_df = pd.concat([train_set.copy(), pd.DataFrame(y_train_pred.copy(),
                                                         columns=["pred"], index=train_set.index)], axis=1)
    
    train_df["split"] = "train"
    
    test_df = pd.concat([test_set.copy(), pd.DataFrame(y_test_pred.copy(),
                                                         columns=["pred"], index=test_set.index)], axis=1)
    
    test_df["split"] = "test"
    
    final_df = pd.concat([train_df.copy(), test_df.copy()])
    
    return final_df


lin_reg_plot_df = make_plot_df(train_set, lin_reg_train_pred, test_set, lin_reg_test_pred)

sns.lineplot(data=lin_reg_plot_df.reset_index(), x="ds", y="y", hue="split", markers=True)
sns.lineplot(data=lin_reg_plot_df.reset_index(), x="ds", y="pred", style="split", dashes=True, color = "black")


# Evaluation

train_rmse = root_mean_squared_error(y_train, lin_reg_train_pred)

test_rmse = root_mean_squared_error(y_test, lin_reg_test_pred)