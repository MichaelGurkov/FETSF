#%% Import libraries


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures
from feature_engine.timeseries.forecasting import (
    LagFeatures,
    WindowFeatures,
)

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

import seaborn as sns

#%%





# Import air_quality

air_quality_folder_path = os.path.dirname(os.path.abspath('__file__')) + "\\Documents\\FETSF\\data"

def load_air_quality(file_path):

    # Load air_quality: only the time variable and CO.
    air_quality = pd.read_csv(
        file_path,
        usecols=["Date_Time", "CO_sensor", "RH"],
        parse_dates=["Date_Time"],
        index_col=["Date_Time"],
    )

    # Sanity: sort index.
    air_quality.sort_index(inplace=True)

    # Reduce air_quality span.
    air_quality = air_quality["2004-04-04":"2005-04-30"]

    # Remove outliers
    air_quality = air_quality.loc[(air_quality["CO_sensor"] > 0)]
    
    ######## NEW #########
    # Add missing timestamps (easier for the demo)
    air_quality = air_quality.asfreq("1H")

    # Fill in missing air_quality.
    air_quality = air_quality.fillna(method="ffill")
    ######################
    
    return air_quality

air_quality = load_air_quality(air_quality_folder_path + "\\AirQualityUCI_ready.csv")

# Split air_quality

train_set = air_quality[air_quality.index < "2005-03-04"]
test_set = air_quality[air_quality.index >= pd.Timestamp("2005-03-04") - pd.offsets.Hour(24)]

del air_quality, air_quality_folder_path

# Preprocess air_quality

def get_X_mat(temp_df):

    # Datetime features
    dtf = DatetimeFeatures(
        # the datetime variable
        variables="index",
        
        # the features we want to create
        features_to_extract=[
            "month",
            "week",
            "day_of_week",
            "day_of_month",
            "hour",
            "weekend",
        ],
    )
    
    lagf = LagFeatures(
        variables=["CO_sensor"],  # the input variables
        freq=["1H", "24H"],  # move 1 hr and 24 hrs forward
        missing_values="ignore",
    )
    
    winf = WindowFeatures(
        variables=["CO_sensor"],  # the input variables
        window="3H",  # average of 3 previous hours
        freq="1H",  # move 1 hr forward
        missing_values="ignore",
    )
    
    cyclicf = CyclicalFeatures(
        # The features we want to transform.
        variables=["month", "hour"],
        # Whether to drop the original features.
        drop_original=False,
    )
    
    imputer = DropMissingData()
    
    drop_ts = DropFeatures(features_to_drop=["CO_sensor", "RH"])
    
    pipe = Pipeline(
        [
            ("datetime_features", dtf),
            ("lagf", lagf),
            ("winf", winf),
            ("Periodic", cyclicf),
            ("drop_target",drop_ts),
            ("dropna", imputer)
        ]
    )
    
    
    X_mat = pipe.fit_transform(temp_df.copy())
    
        
    return X_mat

def get_next_perid_features(data_df):
    
    temp_data = data_df.copy()

    forecast_date = temp_data.index[len(temp_data.index)- 1] + pd.offsets.Hour(1)
        
    temp_data.loc[forecast_date] = np.nan

    next_pred_X_mat = get_X_mat(temp_data.copy())
    
    return next_pred_X_mat

def forecast_next_step(temp_data, fitted_model):
    
    next_pred_X_mat = get_next_perid_features(temp_data)

    next_pred = fitted_model.predict(next_pred_X_mat)

    next_pred = pd.DataFrame(data = next_pred,
                             index = next_pred_X_mat.index, columns = ['CO_sensor', 'RH'])

    temp_data = pd.concat([temp_data.iloc[1:].copy(), next_pred])
    
    return([next_pred, temp_data])

def forecast_entire_horizon(data_df, model, horizon):
    
    ## Make data for model

    X_mat = get_X_mat(data_df)

    Y_mat = data_df.loc[X_mat.index]

    ## Fit model

    model.fit(X_mat, Y_mat)

    predictions_df = []


    ## Make next step
    
    temp_data = data_df.iloc[-24:].copy()

    for i in range(horizon):      
        next_pred, temp_data = forecast_next_step(temp_data, fitted_model = model)
        
        predictions_df.append(next_pred)


    predictions_df = pd.concat(predictions_df)
    
    return predictions_df

    
prediction_df = forecast_entire_horizon(data_df = train_set.copy(),
                                        model = MultiOutputRegressor(Lasso(random_state=0)),
                                        horizon = len(test_set))

# Plot 

plot_df = pd.melt(pd.merge(test_set["CO_sensor"], prediction_df["CO_sensor"],
         left_index=True,
         right_index=True,
         suffixes=["_true","_predict"]).reset_index(), id_vars="Date_Time")

sns.lineplot(plot_df, x = "Date_Time", y = "value", hue = "variable")

mean_squared_error(test_set["CO_sensor"], prediction_df["CO_sensor"], squared=False)