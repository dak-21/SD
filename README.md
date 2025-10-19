🚥 Overview
This dashboard helps supervisors monitor crowd safety during large public events and instantly notify nearby hospitals for surge response if risky crowd behaviors (stampede, dispersal, abnormal density/speed) are detected by AI.
It integrates:

Live crowd analytics using LSTM neural network predictions
Resource-aware hospital selection via agent-based logic
Immediate pop-up notifications to the best-equipped hospitals nearby
Model tweaking and retraining tools
Data visualization and export/import features


⚡ Features

Upload crowd event CSVs (columns: Density, Speed, PoseVariance, optionally Latitude/Longitude)
Predict crowd behavior (Safe, Warning, Dispersal, Emergency, etc.)
Visualize trends: Density, speed, pose variance over time
Hospital surge response

Recommends top 3 hospitals based on proximity and resource availability
Pops up a warning/confirmation when alert is sent
Provides full hospital details (beds, O2, ICUs, ambulances, location/contact)


Manual alert option (send notification directly to hospitals for any crowd status & location)
Model Training tab

Set epoch/batch size/LR/validation split
Simulate (or implement) retraining your crowd behavior model
Progress bar and feedback


System settings (custom thresholds, admin options, expandability for Q-learning, RL tuning)

💡 How It Works

Live prediction: AI model analyzes crowd data, labels each frame as Safe/Warning/Dispersal/Emergency/etc.
Hospital recommendation: Top hospitals are selected by proximity and resource availability (via HospitalSurgeReadinessAgent).
Pop-up notifications: When triggering hospital alerts, a bright warning/banner appears confirming the notification.
Model tweaking: Supervisors can change hyperparameters and retrain the LSTM model directly from the UI.
