# mlfs-book
O'Reilly book - Building Machine Learning Systems with a feature store: batch, real-time, and LLMs


## ML System Examples


[Dashboards for Example ML Systems](https://featurestorebook.github.io/mlfs-book/)


# Run Air Quality Tutorial

See [tutorial instructions here](https://docs.google.com/document/d/1YXfM1_rpo1-jM-lYyb1HpbV9EJPN6i1u6h2rhdPduNE/edit?usp=sharing)
    
# Create a conda or virtual environment for your project before you install the requirements
    pip install -r requirements.txt


##  Run pipelines with make commands

    make aq-backfill
    make aq-features
    make aq-train
    make aq-inference
    make aq-clean

or 
    make aq-all



## Feldera


mkdir -p /tmp/c.app.hopsworks.ai
ln -s  /tmp/c.app.hopsworks.ai ~/hopsworks
docker run -p 8080:8080 \
  -v ~/hopsworks:/tmp/c.app.hopsworks.ai \
  --tty --rm -it ghcr.io/feldera/pipeline-manager:latest


  We only implemented for C-level. 

Branch in git is "final", exist as default though. 

Link for the dashboard: https://astega1.github.io/mlfs-book/air-quality/ 


## Introduction to ML
I wrote a brief introduction to machine learning [here](./introduction_to_supervised_ml.pdf)

# Changes we made in notebook 1:

### Adding lag values
```python
df_aq["pm25_lag1"] = df_aq["pm25"].shift(-1)
df_aq["pm25_lag2"] = df_aq["pm25"].shift(-2)
df_aq["pm25_lag3"] = df_aq["pm25"].shift(-3)
```

# Changes we made in notebook 2: 

```python
aq_today_df = util.get_pm25(aqicn_url, country, city, street, today, AQICN_API_KEY)

#Step 1: get the last 3 real pm25 values
last_3 = (
    air_quality_fg
    .select(['date', 'pm25'])
    .filter(air_quality_fg.city == city)
    .read()
    .sort_values("date")['pm25']
    .astype('float32')   # <-- convert here
    .tail(3)
    .tolist()
)

aq_today_df["pm25_lag1"] = last_3[-1]
aq_today_df["pm25_lag2"] = last_3[-2]
aq_today_df["pm25_lag3"] = last_3[-3]

aq_today_df = aq_today_df.astype({
    "pm25_lag1": "float32",
    "pm25_lag2": "float32",
    "pm25_lag3": "float32"
})

aq_today_df
```

# Changes we made in notebook 3:
```python
#Select features for training data.
selected_features_lag = air_quality_fg.select(['pm25', 'date', 'pm25_lag1', 'pm25_lag2', 'pm25_lag3']).join(weather_fg.select_features(), on=['city'])

feature_view = fs.get_or_create_feature_view(
    name='air_quality_final_fv',
    description="weather features and laged three days",
    version=1,
    labels=['pm25'],
    query=selected_features_lag,
)
```

# Changes in notebook 4: 
```python
air_quality_fg = fs.get_feature_group(
    name='air_quality_final',
    version=1,
)

feature_order = [
    "pm25_lag1",
    "pm25_lag2",
    "pm25_lag3",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant"
]

#Step 1: get the last 3 real pm25 values
last_3 = (
    air_quality_fg
    .select(['date', 'pm25'])
    .filter(air_quality_fg.city == city)
    .read()
    .sort_values("date")['pm25']
    .tail(3)
    .tolist()
)

lags = last_3.copy()
preds = []

#Step 2: recursive loop for each future forecast date
for i in range(len(batch_data)):
    row = batch_data.iloc[i].copy()

    # Add lag features (generated manually)
    row["pm25_lag1"] = lags[-1]
    row["pm25_lag2"] = lags[-2]
    row["pm25_lag3"] = lags[-3]

    # Build input row in correct order
    X_df = pd.DataFrame([row])[feature_order]

    # Predict
    y_pred = np.float32(retrieved_xgboost_model.predict(X_df)[0])
    preds.append(float(y_pred))

    # Update lag list with new prediction
    lags.append(float(y_pred))

batch_data["predicted_pm25"] = pd.Series(preds, dtype="float32")
batch_data
```

