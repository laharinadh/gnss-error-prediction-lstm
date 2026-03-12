# gnss-error-prediction-lstm
LSTM-based deep learning framework for predicting GNSS satellite position and clock errors using time-series data from GEO and MEO satellites.

Overview:-
Global Navigation Satellite Systems (GNSS) provide critical positioning, navigation, and timing services used in aerospace, autonomous systems, and satellite operations. However, GNSS positioning accuracy is affected by multiple sources of error including orbital perturbations, atmospheric delays, and satellite clock drift.
This project develops a time-series deep learning framework based on Long Short-Term Memory (LSTM) networks to predict GNSS satellite position errors. The model learns temporal patterns from historical satellite error data and forecasts future deviations, enabling improved positioning reliability and error mitigation.

Problem Statement:-
Satellite navigation systems experience dynamic positioning errors caused by environmental and orbital factors. Traditional models often rely on physical modeling or statistical estimation, which may not capture complex temporal dependencies.
The objective of this project is to leverage deep learning techniques to model GNSS error dynamics and predict future position errors using sequential data.

Model Architecture:-
The system uses a Long Short-Term Memory (LSTM) neural network designed for sequential time-series prediction.

Key characteristics:-
Captures long-term temporal dependencies in GNSS data
Handles nonlinear error dynamics
Suitable for sequential satellite telemetry data
Input sequence → LSTM layers → Fully connected output layer → Predicted errors

Dataset:-
The dataset contains satellite positional and clock error measurements.

Predicted parameters:-
X-axis position error
Y-axis position error
Z-axis position error
Satellite clock error
The dataset is divided into training and testing sets for evaluating prediction performance.

Methodology:-
Data preprocessing and cleaning
Robust scaling for outlier-resistant normalization
Time-series sequence generation
LSTM model training using PyTorch
Error prediction for satellite position parameters
Model evaluation using statistical metrics

Evaluation Metrics:-
Model performance is evaluated using:
Root Mean Square Error (RMSE) for prediction accuracy
Residual analysis to study error distribution

Technologies:-
Python
PyTorch
NumPy
Pandas
Scikit-learn

Applications:-
Satellite navigation accuracy improvement
GNSS error correction systems
Space mission navigation analysis

Autonomous navigation systems
