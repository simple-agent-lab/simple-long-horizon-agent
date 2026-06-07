# Task: System Architecture Document

You are given business requirements for a real-time chat application and must produce a comprehensive system architecture document.

## Input Files

- `workspace/business_requirements.md` — business requirements for a real-time chat application

## Goal

Design a complete system architecture that addresses all requirements, covering components, data flow, deployment, technology stack, and scaling strategy.

## Requirements

1. Read the business requirements carefully.
2. Create `workspace/architecture.json` with the following top-level sections:
   - `components` — array of system components
   - `data_flow` — array of data flow connections between components
   - `deployment` — deployment configuration
   - `tech_stack` — technology choices
   - `scaling_strategy` — how the system scales

3. **Components** — each must have:
   - `name` — component name
   - `responsibility` — what the component does
   - `interfaces` — array of interfaces it exposes or consumes (e.g., "REST API", "WebSocket", "gRPC")

4. **Data Flow** — each connection must have:
   - `from` — source component name
   - `to` — destination component name
   - `protocol` — communication protocol (e.g., "HTTP", "WebSocket", "AMQP")
   - `description` — what data flows through this connection

5. **Deployment** — must have:
   - `environments` — array of environment objects with `name` and `purpose` (at least: development, staging, production)
   - `containerized` — boolean
   - `orchestration` — orchestration tool name

6. **Tech Stack** — must have:
   - `frontend` — frontend technology/framework
   - `backend` — backend technology/framework
   - `database` — primary database
   - `infrastructure` — cloud/infrastructure platform

7. **Scaling Strategy** — must have:
   - `approach` — description of overall scaling approach
   - `components` — array of component-specific scaling strategies

## Notes

- Components should cover at least: API gateway, chat service, notification service, user service, and message storage.
- Data flow should show how messages travel from sender to receiver.
- Consider real-time requirements (WebSocket connections, message delivery guarantees).
- The architecture should handle the concurrent user and message volume requirements.
