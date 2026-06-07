#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

python3 << PYEOF
import json

api_design = {
    "endpoints": [
        {
            "method": "GET",
            "path": "/api/books",
            "description": "List all books with pagination, filtering, and search",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {
                    "data": {"type": "array", "items": "Book"},
                    "total": {"type": "integer"},
                    "page": {"type": "integer"},
                    "per_page": {"type": "integer"}
                }
            },
            "auth_required": False
        },
        {
            "method": "GET",
            "path": "/api/books/:id",
            "description": "Get detailed information for a specific book",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {
                    "id": "integer",
                    "title": "string",
                    "author": "string",
                    "isbn": "string",
                    "description": "string",
                    "price": "number",
                    "category": "string",
                    "format": "string",
                    "stock_quantity": "integer",
                    "cover_image_url": "string",
                    "published_date": "string",
                    "average_rating": "number"
                }
            },
            "auth_required": False
        },
        {
            "method": "POST",
            "path": "/api/books",
            "description": "Add a new book to the catalog (admin only)",
            "request_body": {
                "title": "string (required)",
                "author": "string (required)",
                "isbn": "string (required)",
                "description": "string",
                "price": "number (required)",
                "category": "string (required)",
                "format": "string (physical|digital)",
                "stock_quantity": "integer",
                "cover_image_url": "string",
                "published_date": "string (YYYY-MM-DD)"
            },
            "response": {
                "type": "object",
                "fields": {"id": "integer", "message": "string"}
            },
            "auth_required": True
        },
        {
            "method": "PUT",
            "path": "/api/books/:id",
            "description": "Update an existing book's information (admin only)",
            "request_body": {
                "title": "string",
                "author": "string",
                "price": "number",
                "description": "string",
                "category": "string",
                "stock_quantity": "integer"
            },
            "response": {
                "type": "object",
                "fields": {"id": "integer", "message": "string"}
            },
            "auth_required": True
        },
        {
            "method": "DELETE",
            "path": "/api/books/:id",
            "description": "Remove a book from the catalog (admin only)",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {"message": "string"}
            },
            "auth_required": True
        },
        {
            "method": "POST",
            "path": "/api/auth/register",
            "description": "Register a new customer account",
            "request_body": {
                "email": "string (required)",
                "password": "string (required)",
                "name": "string (required)"
            },
            "response": {
                "type": "object",
                "fields": {"id": "integer", "token": "string"}
            },
            "auth_required": False
        },
        {
            "method": "POST",
            "path": "/api/auth/login",
            "description": "Authenticate user and receive JWT token",
            "request_body": {
                "email": "string (required)",
                "password": "string (required)"
            },
            "response": {
                "type": "object",
                "fields": {"token": "string", "expires_in": "integer"}
            },
            "auth_required": False
        },
        {
            "method": "GET",
            "path": "/api/users/me",
            "description": "Get current user's profile information",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {
                    "id": "integer",
                    "name": "string",
                    "email": "string",
                    "shipping_address": "string"
                }
            },
            "auth_required": True
        },
        {
            "method": "PATCH",
            "path": "/api/users/me",
            "description": "Update current user's profile information",
            "request_body": {
                "name": "string",
                "email": "string",
                "shipping_address": "string"
            },
            "response": {
                "type": "object",
                "fields": {"message": "string"}
            },
            "auth_required": True
        },
        {
            "method": "POST",
            "path": "/api/cart/items",
            "description": "Add a book to the shopping cart",
            "request_body": {
                "book_id": "integer (required)",
                "quantity": "integer (required, min 1)"
            },
            "response": {
                "type": "object",
                "fields": {"message": "string", "cart_total": "number"}
            },
            "auth_required": True
        },
        {
            "method": "GET",
            "path": "/api/cart",
            "description": "View current cart contents and total price",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {
                    "items": {"type": "array", "items": "CartItem"},
                    "total_price": "number"
                }
            },
            "auth_required": True
        },
        {
            "method": "DELETE",
            "path": "/api/cart/items/:book_id",
            "description": "Remove an item from the shopping cart",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {"message": "string"}
            },
            "auth_required": True
        },
        {
            "method": "POST",
            "path": "/api/orders",
            "description": "Place an order from current cart contents",
            "request_body": {
                "shipping_address": "string (required)"
            },
            "response": {
                "type": "object",
                "fields": {
                    "order_id": "integer",
                    "total_amount": "number",
                    "status": "string"
                }
            },
            "auth_required": True
        },
        {
            "method": "GET",
            "path": "/api/orders",
            "description": "List current user's order history",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {
                    "data": {"type": "array", "items": "Order"},
                    "total": "integer"
                }
            },
            "auth_required": True
        },
        {
            "method": "GET",
            "path": "/api/orders/:id",
            "description": "Get details of a specific order",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {
                    "id": "integer",
                    "items": {"type": "array", "items": "OrderItem"},
                    "total_amount": "number",
                    "status": "string",
                    "shipping_address": "string",
                    "created_at": "string"
                }
            },
            "auth_required": True
        },
        {
            "method": "PATCH",
            "path": "/api/orders/:id/status",
            "description": "Update order status (admin only)",
            "request_body": {
                "status": "string (processing|shipped|delivered|cancelled)"
            },
            "response": {
                "type": "object",
                "fields": {"message": "string"}
            },
            "auth_required": True
        },
        {
            "method": "POST",
            "path": "/api/books/:id/reviews",
            "description": "Write a review for a purchased book",
            "request_body": {
                "rating": "integer (required, 1-5)",
                "comment": "string (required)"
            },
            "response": {
                "type": "object",
                "fields": {"id": "integer", "message": "string"}
            },
            "auth_required": True
        },
        {
            "method": "GET",
            "path": "/api/books/:id/reviews",
            "description": "List all reviews for a specific book",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {
                    "data": {"type": "array", "items": "Review"},
                    "average_rating": "number"
                }
            },
            "auth_required": False
        },
        {
            "method": "DELETE",
            "path": "/api/reviews/:id",
            "description": "Delete a review (owner or admin)",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {"message": "string"}
            },
            "auth_required": True
        },
        {
            "method": "GET",
            "path": "/api/categories",
            "description": "List all book categories",
            "request_body": None,
            "response": {
                "type": "object",
                "fields": {
                    "data": {"type": "array", "items": "Category"}
                }
            },
            "auth_required": False
        }
    ]
}

with open("$WORKSPACE/api_design.json", "w") as f:
    json.dump(api_design, f, indent=2)

print(f"Generated API design with {len(api_design['endpoints'])} endpoints")
PYEOF

echo "API design saved to $WORKSPACE/api_design.json"
