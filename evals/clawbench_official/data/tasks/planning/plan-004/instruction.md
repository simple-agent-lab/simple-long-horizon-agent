# Task: API Design from PRD

You are given a product requirements document (PRD) for an online bookstore and must design a complete RESTful API.

## Input Files

- `workspace/prd.md` — product requirements for a bookstore API

## Goal

Design a comprehensive RESTful API that covers all functionality described in the PRD.

## Requirements

1. Read the PRD carefully.
2. Create `workspace/api_design.json` with the following structure:
   ```json
   {
     "endpoints": [
       {
         "method": "GET",
         "path": "/api/books",
         "description": "List all books with pagination",
         "request_body": null,
         "response": {"type": "array", "items": "Book"},
         "auth_required": false
       }
     ]
   }
   ```
3. Design at least 8 API endpoints.
4. Each endpoint must have:
   - `method` — valid HTTP method (GET, POST, PUT, PATCH, DELETE)
   - `path` — API path starting with `/api/`
   - `description` — what the endpoint does
   - `request_body` — object describing request body (or `null` for GET/DELETE)
   - `response` — object describing response shape
   - `auth_required` — boolean indicating if authentication is needed
5. Cover CRUD operations for the main resources (books, users, orders, reviews).
6. All write operations (POST, PUT, PATCH, DELETE) must require authentication.
7. Paths must follow RESTful conventions (e.g., `/api/books/:id`).

## Notes

- Use consistent path naming conventions (lowercase, hyphens for multi-word).
- Include at least one endpoint for each CRUD operation (Create, Read, Update, Delete).
- Response objects should describe the shape (type, fields) not actual data.
- Consider pagination for list endpoints.
