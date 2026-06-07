# Sensor Data Anomaly Detection (IQR and Z-Score)

## Description

You are given a CSV file `sensor_data.csv` located in the workspace directory. This file contains sensor readings with the following columns:

- `timestamp`: ISO 8601 formatted timestamp string
- `sensor_id`: Identifier for the sensor (string)
- `value`: Numeric sensor reading (float)

Your task is to detect anomalies in the sensor readings for each sensor using two methods:

1. **IQR Method**:
   - Compute the first quartile (Q1) and third quartile (Q3) of the sensor's values.
   - Calculate the interquartile range (IQR = Q3 - Q1).
   - Flag any reading as an anomaly if it is below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.

2. **Z-Score Method**:
   - Compute the mean and standard deviation of the sensor's values.
   - Calculate the z-score for each reading.
   - Flag any reading as an anomaly if the absolute z-score is greater than 3.

After detecting anomalies with both methods, compare the results per sensor:

- List the timestamps and values flagged by each method.
- Calculate the anomaly rate (percentage of readings flagged) for each method.

## Output

Write a JSON file named `anomalies.json` in the workspace directory with the following structure:

```json
{
  "sensor_id_1": {
    "iqr_anomalies": [
      {"timestamp": "...", "value": ...},
      ...
    ],
    "zscore_anomalies": [
      {"timestamp": "...", "value": ...},
      ...
    ],
    "anomaly_rate": {
      "iqr": 0.05,
      "zscore": 0.03
    }
  },
  "sensor_id_2": {
    ...
  },
  ...
}
```

## Requirements

- Read `sensor_data.csv` from the workspace.
- Perform anomaly detection per sensor using both methods.
- Write the results to `anomalies.json` in the workspace.
- Use threshold 3 for Z-score method.

## Notes

- Ensure your solution handles multiple sensors independently.
- The dataset contains at least 20 readings per sensor.
- Use standard Python libraries or command line tools available in a typical data science environment.

## Files

- Input: `sensor_data.csv` (in workspace)
- Output: `anomalies.json` (in workspace)

Good luck!