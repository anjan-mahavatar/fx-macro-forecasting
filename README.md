# Quantitative FX Macro Forecasting, Scenario Shock and Option Pricing 
Business Value Add:
The core business value of this system is the integration of predictive analytics and real-time derivative pricing into a unified, interactive quantitative studio. By leveraging AI models (such as LSTMs and LightGBM) alongside established mathematical frameworks, the application enables financial analysts to forecast foreign exchange (FX) prices and volatility with significantly higher precision. The inclusion of a Macro Scenario Simulator allows for dynamic stress-testing against market shocks (e.g., Brent Crude shifts or VIX spikes), which is critical for robust capital risk profiling. Furthermore, the dedicated Explainable AI (XAI) stage ensures that model outputs remain transparent, satisfying the strict regulatory and compliance standards required within the financial sector.

Without ML (Baseline Implementation):
If Machine Learning were stripped from this pipeline, the system would rely entirely on traditional statistical and econometric methods to achieve a similar, though less adaptive, result:

Volatility & Price Forecasting: Instead of utilizing neural networks, you would implement an AutoRegressive Integrated Moving Average (ARIMA) model for directional price trends and a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model to capture volatility clustering.

Sentiment Analysis: Instead of utilizing the FinBERT LLM pipeline to analyze market text, a non-ML approach would involve a deterministic, rule-based lexicon (like the Loughran-McDonald financial dictionary) to mechanically count positive and negative terms in headlines.

Risk Modeling: The analytical pricing modules would be fed strictly by historical rolling standard deviations and simple moving averages, rather than dynamically predicted targets derived from complex feature interactions.

2. Module & Library Documentation
data_import.py

Functionality: Acts as the multi-source data ingestion engine. It connects to external APIs to pull structural market data and macroeconomic indicators, such as the 2-Year Treasury Yield and the VIX. It also structures historical anchor points for news headlines.

Libraries: yfinance (for FX, Oil, Gold), pandas_datareader (for FRED API economic data), pandas, numpy, datetime.

data_cleansing.py

Functionality: Executes the structural data cleansing pipeline. It handles missing data via forward and backward filling to prevent time-series data leakage, engineers new temporal features, and establishes the dual targets (next-day price and 30-day rolling volatility). A critical engineering decision in this module is maintaining object data types for specific time-series columns within the dataframes; this strictly complies with the automated testing expectations defined in the CI pipeline.

Libraries: pandas, numpy, sklearn.preprocessing (specifically StandardScaler to prevent gradient issues).

modules.py & nn_lstm.py

Functionality: These files define the structural architecture for both the deep sequence layers and the pricing mathematics. modules.py transforms flat historical 2D vectors into 3D sequential array matrices to feed into Keras networks (Simple RNN, LSTM, CNN-LSTM). It also houses the analytical derivative pricing logic, including Garman-Kohlhagen and Binomial Trees. nn_lstm.py implements a focused Multi-Layer Perceptron (MLP) architecture specifically tuned for volatility regression.

Libraries: tensorflow.keras (Sequential, layers), numpy, scipy.stats, sklearn.neural_network.

training.py & plotting.py

Functionality: training.py orchestrates the chronological, time-series-aware split of the data (80% train / 20% test) and triggers the training of both the price predictors and volatility neural networks. plotting.py provides the visual output generation for actual versus predicted targets. During baseline testing phases for these modeling pipelines, it was identified that reverting to default parameters in a Logistic Regression model (max_iter=1000) was necessary to resolve validation errors when evaluating custom unregularized objective functions against these more advanced techniques.

Libraries: matplotlib.pyplot, sys, os.

explainability.py

Functionality: Drives the Explainable AI (XAI) pipeline. It leverages game-theoretic approaches to produce global and local interpretability charts, ensuring the ensemble tree models and volatility networks can be understood via feature importance summaries and "if-then" anchor boundary rules.

Libraries: shap, lime, alibi.explainers (AnchorTabular), matplotlib, pandas.

llm_scenario.py

Functionality: Introduces a natural language processing layer by utilizing a pre-trained financial LLM to extract macro risk scores from textual headlines. This effectively converts qualitative market sentiment into quantitative risk metrics.

Libraries: transformers (Hugging Face pipeline).

app.py

Functionality: The unified frontend application. It provides sidebar controls for data ingestion, dynamic parameter tuning (adjusting lookback windows, boosting rounds, epochs), and interactive slider controls for macro scenario shocks. It seamlessly bridges the Python backend logic with a user-friendly browser experience.

Libraries: streamlit, lightgbm, tensorflow, yfinance, scipy, sklearn.

gitlab-ci.yml

Functionality: Defines the Continuous Integration (CI) parameters, orchestrating the automated steps for setting up Python 3.10 environments, running pytest suites, and conditionally executing the training pipeline on the main branch.
