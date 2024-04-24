
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

# Forecast step by step

def forecast_next_step(data_df, model):
    
    X_mat = get_X_mat(data_df)
    
    Y_mat = data_df.loc[X_mat.index]
    
    model.fit(X_mat, Y_mat)

    # Make forecast input data    

    predict_data = data_df.iloc[-24:].copy()
    
    forecast_date = predict_data.index[len(predict_data.index)- 1] + pd.offsets.Hour(1)
        
    predict_data.loc[forecast_date] = np.nan
    
    next_pred_X_mat = get_X_mat(predict_data.copy())
    
    next_pred = model.predict(next_pred_X_mat)
    
    next_pred = pd.DataFrame(data = next_pred,
                             columns = data_df.columns.values,
                             index = [forecast_date])
    
    return next_pred


# Forecast 24 hours

temp_data = train_set.copy()

for i in range(24):
    print(i)
    
    next_pred = forecast_next_step(data_df = temp_data,
                               model = MultiOutputRegressor(Lasso(random_state=0)))
    
    temp_data = pd.concat([temp_data.copy(), next_pred])
    

# Plot 

temp_data.loc[test_set.index]

pred_data = temp_data.iloc[-24:].copy()

pd.melt(pred_data,)

plot_df = test_set[["CO_sensor"]].loc[pred_data.index].join(pred_data[["CO_sensor"]],
                                   lsuffix = "_actual", rsuffix = "_pred")

mean_squared_error(test_set["CO_sensor"].loc[pred_data.index], pred_data["CO_sensor"],squared=False)

mean_squared_error(test_set["RH"].loc[pred_data.index], pred_data["RH"],squared=False)

sns.lineplot(pd.melt(plot_df.reset_index(), id_vars="index"),
             x = "index", y = "value", hue = "variable")