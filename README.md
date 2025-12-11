# Forecasting Sea Surface Temperature and Marine Weather Variables with ICOADS Time Series Data
-----

## Why?

Global maritime freight transport underpins international trade and is valued at hundreds of billions of USD annually. Shipping is often the only viable way to transport goods such as energy resources, grains, manufactured products, etc. However, ocean routes are highly exposed to adverse marine weather; weather-affected voyages can incur hundreds of thousands of dollars of extra cost due to added fuel and time at sea.

Accurately forecasting key ocean–atmosphere variables such as **sea surface temperature (SST)** (and other related atmosphere variables) can help with the reduction of disruptions and operating costs.

As such, this project explores statistical and deep learning methods to forecast SST and other related variables from the **ICOADS** dataset.

-----

## The Problem

Concretely, throught this project, we seek to:

- Forecast **SST** and other auxiliary marine weather variables
- Use **time series modeling** (statistical and deep learning) to model:
    - Seasonality and long-term trends
    - Short-term temporal dependencies
    - Relationships between SST and covariates

-----

## The ICOADS Dataset

### Sourcing the Data

We used the **NOAA ICOADS** dataset:

- One of the most extensive historical collections of marine environmental observations
- Contains measurements from:
    - Ships
    - Drifting buoys
    - Moored buoys
    - Other ocean-based platforms
- ~800 million records spanning from the **mid-1600s to 2017**

### Accessing the Data

Our data access workflow was as follows:

1. Use **Kaggle’s public BigQuery** interface, as the ICOADS tables are exposed as part of `bigquery-public-data`
2. Filter and store a subset in custom **BigQuery** tables
3. Query, aggregate, and export data into analysis environments (BigQuery client)

For computational efficacy/efficiency, we set the following limits:

- Time window restricted to the year range **2005–2017**
- Data is filtered and partitioned by **year** for faster BigQuery queries

### Variables Used

Key attributes used in the analysis:

- **Main (Target) Variable**
    - `SST`, sea surface temperature
- **Atmospheric covariates**
    - `air_temperature`
    - `sea_level_pressure (SLP)`
    - `wind_speed`
    - `dpt_indicator`
- **Spatiotemporal attributes**
    - `latitude`, `longitude`
    - `year`, `month`, `day`, `hour`

The final subset of data we chose to focus on was centered on **37.81° N, 122.47° W**. Its granularity was as follows:
    - Native data: hourly
    - Modeling data: resampled to **daily** SST for stability and comparability

-----

## Methods

This project explores three main classes of models:

1. **Statistical time series models** (autoregressive)
2. **Recurrent neural networks** (LSTM)
3. **Open-source decomposable models** (Prophet)

All analysis and preprocessing operations were carried out using the following:

- **Google BigQuery** (SQL for data filtering and aggregation)
- **Python** (Pandas, NumPy)
- **Statsmodels** (statistical time series)
- **PyTorch** (LSTM)
- **Prophet** (decomposable time-series model)
- **Matplotlib** (visualization)

### Exploratory Data Analysis (EDA)

The main EDA steps were as follows:

- **Schema inspection and cleaning**:
    - Removal of audit and non-analytical fields
- **Correlation analysis**:
    - Identify variables highly correlated with SST
    - Drop all other temperature variables highly collinear with SST, retaining `dpt_indicator`
- **Missingness analysis**:
    - Pairwise missingness dependence analysis confirmed that our data was not missing completely at random (MCAR)
    - Missingness heatmaps showed that there were structured missingness patterns across specific groups ov variables
- **Temporal structure**:
    - Time series plots of SST (levels and decomposed into trend/seasonal/remainder)
    - Evidence of strong **annual seasonality** and smooth long-term trend
- **Geospatial structure**:
    - Maps and density plots showing spatial concentration of observations
- **Distributional properties**:
    - Wind speed exhibits skew; log transformation used to approximate normality
    - Correlation matrices examined before and after transformation

These analyses informed:

- Variable selection (dropping collinear temperatures),
- Choice to treat later models as primarily **univariate in SST** due to structured missingness in covariates,
- Need for differencing to achieve approximate stationarity for autoregressive modeling.

-----

### Statistical Modeling (Autoregressive)

Because exogenous variables have substantial missingness, and because SST shows strong temporal dependence, we focused on **univariate autoregressive models** as baselines:

Key steps:

1. **Location selection**:
    - Choose top 5 locations by SST record count
    - Focus on the most data-rich location (**37.81° N, 122.47° W**)
2. **Resampling**:
    - Convert hourly SST series to **daily** mean SST
3. **Autocorrelation analysis**:
    - ACF and PACF plots for SST showed:
        - Persistent autocorrelation
        - Need for differencing
    - First-difference SST ACF/PACF indicated:
        - Autoregressive structure with lags up to 2 is appropriate
4. **Model selection**:
    - Fit AR models with differencing; candidate orders include AR(1) and AR(2) with \( d = 1 \)
    - Use information criteria (Log-likelihood, AIC, BIC, HQIC) and residual ACF to evaluate models
    - Select **AR(2, 1, 0)** as the best-performing statistical baseline:
        - Residuals show no clear remaining autocorrelation
        - Indicates good capture of temporal structure

-----

### Recurrent Neural Network: LSTM

To capture **nonlinear temporal dynamics** and incorporate multiple predictors, we implemented a **Long Short-Term Memory (LSTM)** network in PyTorch.

#### Features and Preprocessing

- **Input features**:
    - Latitude, longitude
    - Sea level pressure (SLP)
    - Wind speed
    - Cyclic encodings of **month** and **hour** (sine/cosine)
- **Target**:
    - SST
- **Preprocessing**:
    - Forward–backward fill for missing values
    - Cast all features to numeric types
    - Standardized inputs and target using `StandardScaler`

#### Sequence Construction

- Construct overlapping sliding windows:
    - Window length: **48 time steps**
    - Each window uses 48-step history to predict SST at the next time step (single-step forecast)
- Train/validation/test split:
    - Chronological split:
        - ~70% train
        - ~20% validation
        - ~10% test
- Data loaders:
    - Wrap sequences into PyTorch `Dataset`/`DataLoader` for mini-batch training

#### Model Architecture and Training

- **Architecture**:
    - 2-layer LSTM with 64 hidden units
    - Dropout: 0.2
    - Fully connected output layer mapping final hidden state → scalar SST forecast
- **Training setup**:
    - Loss: Mean Squared Error (MSE)
    - Optimizer: Adam, learning rate 0.001
    - Batch size: 64
    - Epochs: 10 (with validation monitoring)

Training and validation loss curves showed:

- Gradual reduction in training error
- Stable validation loss
- No severe overfitting at the chosen configuration

#### LSTM Performance

On the held-out test set, after inverse-transforming predictions back to physical SST units:

- **MSE** ~= 4.19  
- **RMSE** ~= 2.05  
- **MAE** ~= 1.71  

Qualitatively, the LSTM:

- Tracks main SST trends and seasonal oscillations
- Captures many local fluctuations
- Under- or over-estimates some high-variance spikes, reflecting intrinsic noise and unresolved dynamics

-----

### Prophet

To complement AR and LSTM models, we use **Facebook Prophet** as an interpretable baseline for the same SST series.

#### Setup

- Use daily resampled SST at **(37.81° N, 122.47° W)**.
- Prepare data in Prophet format:
    - `ds`: timestamp
    - `y`: SST
- No external regressors:
    - Aligns with univariate AR setup and avoids missingness in atmospheric covariates
- Model components:
    - Trend: piecewise linear
    - Seasonality:
        - **Yearly seasonality enabled** (dominant pattern)
        - Weekly seasonality disabled (no meaningful weekly cycle in SST)
        - Additional Fourier terms allowed for flexible annual cycle

#### Training and Evaluation

- Train on **2005–2014**
- Test on **2015–2017**, matching LSTM time split strategy

Performance on test period:

- **RMSE** ~= 4.51  
- **MAE** ~= 3.85  
- **MAPE** ~= 28.68  

Prophet:

- Accurately captures **annual rise and fall** in SST
- Smooths out some rapid short-term oscillations
- Produces sensible uncertainty intervals that widen during higher-variance periods

Prophet’s strength lies in:

- Interpretability (trend + seasonal components)
- Robustness to missing data and outliers
- Serving as a diagnostic and baseline model against more complex deep learning methods

-----

## Summary of Results

Across models:

| Model          | Inputs                    | Granularity   | RMSE (SST) | MAE (SST) | Notes                                                   |
|----------------|---------------------------|---------------|------------|-----------|---------------------------------------------------------|
| AR(2, 1, 0)    | Univariate SST            | Daily         | Baseline   | Baseline  | Captures trend/seasonality after differencing           |
| Prophet        | Univariate SST            | Daily         | ~= 4.51    | ~= 3.85   | Strong on yearly cycle; smooths high-frequency noise    |
| LSTM           | Multivariate (SST + exog) | Daily windows | ~= 2.05    | ~= 1.71   | Best short-term accuracy; captures nonlinear dynamics   |

Key observations:

- SST is **strongly seasonal** with a clear annual cycle and smooth long-term trend
- Autocorrelation structure motivates differencing and supports autoregressive modeling
- Univariate AR(2,1,0) provides a well-behaved statistical baseline with uncorrelated residuals
- Prophet improves interpretability and long-horizon seasonal forecasting but is less reactive to rapid changes
- LSTM, leveraging multivariate inputs and nonlinear dynamics, yields the **lowest forecast error** on short horizons, at the expense of interpretability and complexity

Overall, SST in the selected region is **moderately predictable**:

- Simple models are sufficient for capturing large-scale trend and seasonality
- Deep learning models improve accuracy for short-term fluctuations important for operational maritime decision-making



<!-- For more info, please see: https://github.com/alexyzha/ICOADS-Time-Series-Analysis/pull/13 -->
