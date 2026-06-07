#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

python3 << PYEOF
import json

architecture = {
    "components": [
        {
            "name": "API Gateway",
            "responsibility": "Route incoming HTTP and WebSocket requests, handle rate limiting, SSL termination, and authentication token validation",
            "interfaces": ["REST API", "WebSocket", "HTTPS"]
        },
        {
            "name": "Auth Service",
            "responsibility": "Handle user authentication, SSO integration (SAML 2.0, OAuth 2.0, OIDC), JWT token issuance, and session management",
            "interfaces": ["REST API", "gRPC"]
        },
        {
            "name": "User Service",
            "responsibility": "Manage user profiles, presence status, contacts, preferences, and role-based access control",
            "interfaces": ["REST API", "gRPC"]
        },
        {
            "name": "Chat Service",
            "responsibility": "Handle real-time message delivery, typing indicators, read receipts, message editing/deletion, and thread management via persistent WebSocket connections",
            "interfaces": ["WebSocket", "gRPC", "AMQP"]
        },
        {
            "name": "Channel Service",
            "responsibility": "Manage channel creation, membership, permissions, pinned messages, and channel metadata",
            "interfaces": ["REST API", "gRPC"]
        },
        {
            "name": "Notification Service",
            "responsibility": "Process and deliver push notifications, email digests, mention alerts, and manage per-user notification preferences and quiet hours",
            "interfaces": ["gRPC", "AMQP", "Push (APNs/FCM)"]
        },
        {
            "name": "File Service",
            "responsibility": "Handle file uploads, virus scanning, thumbnail generation, storage management, and serve file downloads with CDN integration",
            "interfaces": ["REST API", "S3-compatible"]
        },
        {
            "name": "Search Service",
            "responsibility": "Index messages and files for full-text search, support filtered queries across message history with sub-500ms response times",
            "interfaces": ["REST API", "gRPC"]
        },
        {
            "name": "Message Store",
            "responsibility": "Persist all messages with ordering guarantees, support message retrieval by channel/thread/time range, and enforce retention policies",
            "interfaces": ["gRPC", "Database protocol"]
        },
        {
            "name": "Message Queue",
            "responsibility": "Decouple services with reliable async message passing, ensure at-least-once delivery for notifications and cross-service events",
            "interfaces": ["AMQP", "Pub/Sub"]
        },
        {
            "name": "Presence Service",
            "responsibility": "Track online/away/busy/offline status for all connected users, broadcast presence updates, and integrate with calendar for automatic status",
            "interfaces": ["WebSocket", "gRPC", "Redis Pub/Sub"]
        },
        {
            "name": "Audit Service",
            "responsibility": "Log all administrative actions, message exports, and access events for compliance (SOC 2, GDPR, HIPAA) reporting",
            "interfaces": ["gRPC", "AMQP"]
        }
    ],
    "data_flow": [
        {
            "from": "Client Application",
            "to": "API Gateway",
            "protocol": "WebSocket/HTTPS",
            "description": "Client establishes WebSocket for real-time messaging and HTTPS for REST API calls"
        },
        {
            "from": "API Gateway",
            "to": "Auth Service",
            "protocol": "gRPC",
            "description": "Validate JWT tokens and authenticate incoming requests"
        },
        {
            "from": "API Gateway",
            "to": "Chat Service",
            "protocol": "WebSocket",
            "description": "Forward authenticated WebSocket connections for real-time messaging"
        },
        {
            "from": "Chat Service",
            "to": "Message Store",
            "protocol": "gRPC",
            "description": "Persist sent messages and retrieve message history"
        },
        {
            "from": "Chat Service",
            "to": "Message Queue",
            "protocol": "AMQP",
            "description": "Publish message events for notification delivery and search indexing"
        },
        {
            "from": "Message Queue",
            "to": "Notification Service",
            "protocol": "AMQP",
            "description": "Consume message events to trigger push notifications and mention alerts"
        },
        {
            "from": "Message Queue",
            "to": "Search Service",
            "protocol": "AMQP",
            "description": "Consume message events to index new messages for full-text search"
        },
        {
            "from": "Chat Service",
            "to": "Presence Service",
            "protocol": "Redis Pub/Sub",
            "description": "Update and broadcast user presence and typing indicators"
        },
        {
            "from": "API Gateway",
            "to": "File Service",
            "protocol": "HTTP",
            "description": "Upload files and retrieve download URLs"
        },
        {
            "from": "File Service",
            "to": "Message Store",
            "protocol": "gRPC",
            "description": "Associate uploaded files with messages and channels"
        },
        {
            "from": "API Gateway",
            "to": "User Service",
            "protocol": "gRPC",
            "description": "User profile management and RBAC permission checks"
        },
        {
            "from": "Notification Service",
            "to": "Client Application",
            "protocol": "Push (APNs/FCM)",
            "description": "Deliver push notifications to mobile and desktop clients"
        }
    ],
    "deployment": {
        "environments": [
            {
                "name": "development",
                "purpose": "Local development and feature testing with mocked external services"
            },
            {
                "name": "staging",
                "purpose": "Pre-production environment for integration testing, load testing, and QA validation"
            },
            {
                "name": "production",
                "purpose": "Multi-region production deployment serving end users with full HA and DR capabilities"
            }
        ],
        "containerized": True,
        "orchestration": "Kubernetes",
        "ci_cd": "GitHub Actions with ArgoCD for GitOps deployment",
        "regions": ["us-east-1", "eu-west-1"],
        "monitoring": "Prometheus + Grafana with PagerDuty alerting"
    },
    "tech_stack": {
        "frontend": "React with TypeScript, Socket.IO client for WebSocket, Redux for state management",
        "backend": "Go for Chat and Presence services (performance-critical), Python (FastAPI) for User, Channel, and File services",
        "database": "PostgreSQL for relational data (users, channels, metadata); Cassandra for message storage (high-write throughput); Redis for presence, caching, and pub/sub",
        "infrastructure": "AWS (EKS, RDS, ElastiCache, S3, CloudFront, SQS/SNS), Terraform for IaC",
        "search": "Elasticsearch for full-text message search and indexing",
        "message_queue": "RabbitMQ for inter-service messaging with dead-letter queues",
        "cdn": "CloudFront for static assets and file downloads",
        "monitoring": "Prometheus, Grafana, Jaeger for distributed tracing, ELK stack for log aggregation"
    },
    "scaling_strategy": {
        "approach": "Horizontal auto-scaling with Kubernetes HPA based on CPU, memory, and custom metrics (WebSocket connections, message throughput). Multi-region deployment with global load balancing for latency optimization. Stateless services scale independently; stateful components use sharding and replication.",
        "components": [
            {
                "name": "Chat Service",
                "strategy": "Horizontal scaling with sticky sessions per WebSocket connection. Shard by channel_id to distribute load. Target: 10,000 concurrent connections per pod, scale up at 70% capacity."
            },
            {
                "name": "Message Store (Cassandra)",
                "strategy": "Partition by channel_id and time bucket for even distribution. Add nodes to scale write throughput linearly. Replication factor 3 across availability zones."
            },
            {
                "name": "Presence Service",
                "strategy": "Redis Cluster with hash-slot based sharding. Ephemeral data allows aggressive eviction. Scale Redis nodes horizontally as connection count grows."
            },
            {
                "name": "Search Service",
                "strategy": "Elasticsearch cluster with index sharding by organization. Add data nodes for more storage, replicas for read throughput. Time-based indices with automated rollover."
            },
            {
                "name": "API Gateway",
                "strategy": "Kubernetes HPA with target 60% CPU utilization. Global load balancer distributes across regions based on latency. Rate limiting at gateway level protects downstream services."
            },
            {
                "name": "Notification Service",
                "strategy": "Consumer group scaling based on queue depth. Multiple workers process notification events in parallel with at-least-once delivery guarantees."
            }
        ]
    }
}

with open("$WORKSPACE/architecture.json", "w") as f:
    json.dump(architecture, f, indent=2)

print(f"Generated architecture with {len(architecture['components'])} components, "
      f"{len(architecture['data_flow'])} data flows")
PYEOF

echo "Architecture saved to $WORKSPACE/architecture.json"
