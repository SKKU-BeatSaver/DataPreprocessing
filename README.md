# BeatSaver - Panic Disorder Risk Assessment System

A Flask-based web application that analyzes ECG data and psychological assessments to predict panic disorder risk.

## Key Features

- ECG data processing and HRV (Heart Rate Variability) analysis
- Integration of psychological assessments (PDSS and APPQ)
- Machine learning model for risk prediction (Random Forest)
- Age-specific RMSSD range analysis

## Components

- ECG data processing using BioSPPy
- PDSS (Panic Disorder Severity Scale) evaluation
- APPQ (Anxiety Sensitivity Profile Questionnaire) assessment
- Combined risk calculation with weighted probabilities:
  - ECG/HRV: 40%
  - PDSS: 30%
  - APPQ: 30%

## API Endpoints

```
POST /upload - Upload and process ECG data
POST /dsm - Submit PDSS assessment
POST /appq - Submit APPQ assessment
GET /get_final_probability - Get combined risk assessment
POST /reset_scores - Reset all assessment scores
```

## Model Performance

The Random Forest model achieved:
- Accuracy: 99.27%
- Precision: 99.69%
- Recall: 98.49%
- F1 Score: 99.09%

## Setup

1. Install required Python packages:
```bash
pip install flask pandas numpy scipy biosppy scikit-learn
```

2. Start the Flask server:
```bash
python main.py
```

## Data Processing

- Supports CSV ECG data with 500Hz sampling rate
- Processes metadata and voltage data separately
- Calculates RMSSD and HRV metrics
- Age-specific risk normalization

## Note

This system is part of a larger panic disorder prediction platform integrating wearable ECG monitoring with standardized psychological assessments.
