# REST API Design with OpenAPI Spec and Flask Stubs

## Overview

You are provided with an API requirements file `api_requirements.txt` in the workspace. This file describes a set of REST API endpoints, including their HTTP methods, request schemas, and response schemas.

Your task is to:

1. **Read** the API requirements from `workspace/api_requirements.txt`.
2. **Generate** a valid OpenAPI 3.0 specification file `workspace/openapi.yaml` that fully describes the API endpoints, methods, request bodies, and responses.
3. **Generate** a Flask application stub file `workspace/app.py` that defines the endpoints and methods as Flask route stubs, including request parsing and response placeholders.
4. **Write** a JSON summary file `workspace/api_summary.json` containing:
   - `endpoint_count`: total number of distinct endpoints
   - `methods`: a dictionary mapping HTTP methods to the count of endpoints supporting them

## Details

### Input File: `workspace/api_requirements.txt`

- Each endpoint is described in blocks separated by blank lines.
- Each block has the following format:

```
Endpoint: /path/to/resource
Method: GET|POST|PUT|DELETE|PATCH
RequestSchema:
{
  JSON schema for request body or empty `{}` if none
}
ResponseSchema:
{
  JSON schema for response body
}
```

- Example:

```
Endpoint: /users
Method: POST
RequestSchema:
{
  "type": "object",
  "properties": {
    "username": {"type": "string"},
    "email": {"type": "string"}
  },
  "required": ["username", "email"]
}
ResponseSchema:
{
  "type": "object",
  "properties": {
    "id": {"type": "integer"},
    "username": {"type": "string"}
  }
}
```

### Output Files

- `workspace/openapi.yaml`: OpenAPI 3.0 spec describing all endpoints with methods, requestBody (if any), and responses.
- `workspace/app.py`: Flask app with route stubs for each endpoint and method. Each stub should:
  - Use Flask decorators with route and methods.
  - Parse JSON request body if applicable.
  - Return a placeholder JSON response.
- `workspace/api_summary.json`: JSON file with keys:
  - `endpoint_count`: integer
  - `methods`: object mapping HTTP method names to counts

## Requirements

- The OpenAPI spec must be valid YAML and conform to OpenAPI 3.0.
- Flask stubs must be syntactically valid Python code.
- The summary JSON must correctly count endpoints and methods.

## Evaluation

Your solution will be tested by running your generated Flask app and validating the OpenAPI spec and summary.

Good luck!