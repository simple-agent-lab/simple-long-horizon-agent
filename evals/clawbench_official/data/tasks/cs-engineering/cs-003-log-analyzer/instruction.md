# Server Log Anomaly Detection and Incident Report

## Overview

You are provided with a server log file located at `workspace/server.log`. Each log entry contains the following fields:

- `timestamp` (ISO 8601 format, e.g., `2024-06-01T12:34:56`)
- `level` (e.g., `INFO`, `WARN`, `ERROR`)
- `service` (name of the service generating the log)
- `message` (text message describing the event)

Your task is to analyze this log file and detect anomalies related to server errors and service failures. Specifically, you need to:

1. **Detect error spikes:** Identify any 1-minute intervals where the number of `ERROR` level log entries exceeds 5.

2. **Detect repeated patterns:** Identify any repeated log messages that occur at least 3 times consecutively.

3. **Detect service failures:** Identify services that have continuous error logs indicating failure (e.g., 3 or more consecutive `ERROR` entries from the same service).


## Output

You must write a JSON report to `workspace/incident_report.json` containing the following keys:

- `timeline`: A chronological list of detected anomaly events with timestamps and descriptions.

- `anomalies`: A summary list of anomaly types detected (e.g., `error_spikes`, `repeated_patterns`, `service_failures`).

- `affected_services`: A list of service names involved in anomalies.

- `severity`: An overall severity level string (`low`, `medium`, `high`) based on the number and type of anomalies detected.


## Details

- The `timeline` should include entries such as:
  - Time window of error spike and count
  - The repeated message and its occurrence timestamps
  - Service failure periods and affected service names

- The `severity` is defined as:
  - `low`: Only repeated patterns detected
  - `medium`: Error spikes or service failures detected
  - `high`: Both error spikes and service failures detected


## Constraints

- Use only the provided `server.log` file.
- The output JSON must be valid and well-formatted.
- Efficiency is important; the log file may contain hundreds of entries.


## Example

If the log contains a spike of 7 errors between `2024-06-01T12:00:00` and `2024-06-01T12:01:00`, repeated messages "Connection timeout" 4 times consecutively, and a service "auth" failing with 3 consecutive errors, the report should reflect all these anomalies with appropriate severity.


## Submission

Your solution must read `workspace/server.log` and write `workspace/incident_report.json` as described.

---