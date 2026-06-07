#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

python3 << PYEOF
import json

analysis = {
    "options": [
        {
            "name": "React + Django REST + PostgreSQL on AWS",
            "pros": [
                "Team already has React and Python skills — minimal ramp-up time",
                "Django has built-in ORM, auth, and admin panel reducing development effort",
                "PostgreSQL is HIPAA-capable with strong encryption and audit features",
                "Mature ecosystem with extensive libraries for HL7 FHIR integration",
                "AWS offers HIPAA-eligible services and BAA agreements"
            ],
            "cons": [
                "Django can be opinionated, limiting flexibility for unconventional patterns",
                "Real-time features require additional WebSocket setup (Django Channels)",
                "AWS costs can escalate without careful resource management"
            ],
            "cost_estimate": 85000,
            "recommendation_score": 9,
            "recommended": True
        },
        {
            "name": "React + Node.js (Express) + MongoDB on GCP",
            "pros": [
                "Full JavaScript stack enables code sharing between frontend and backend",
                "Node.js excels at real-time applications with native WebSocket support",
                "MongoDB flexible schema speeds initial development",
                "GCP healthcare API provides native HL7 FHIR support"
            ],
            "cons": [
                "Team has limited Node.js experience — training overhead",
                "MongoDB requires extra effort for HIPAA-compliant configuration",
                "Schema flexibility can lead to data consistency issues in healthcare context",
                "Weaker ORM ecosystem compared to Django"
            ],
            "cost_estimate": 90000,
            "recommendation_score": 6,
            "recommended": False
        },
        {
            "name": "React + .NET Core + SQL Server on Azure",
            "pros": [
                "Azure has the strongest HIPAA compliance tooling and healthcare offerings",
                "SQL Server provides enterprise-grade security and auditing out of the box",
                ".NET Core offers excellent performance and built-in dependency injection"
            ],
            "cons": [
                "Team has no C#/.NET experience — significant learning curve",
                "Higher licensing costs for SQL Server and Azure services",
                "Longer development timeline due to skill gap",
                "Less open-source ecosystem alignment"
            ],
            "cost_estimate": 115000,
            "recommendation_score": 4,
            "recommended": False
        },
        {
            "name": "Next.js + FastAPI + PostgreSQL on Fly.io",
            "pros": [
                "Next.js offers SSR for faster initial loads and better SEO",
                "FastAPI is Python-based with automatic OpenAPI documentation",
                "Fly.io offers simpler deployment with lower operational overhead",
                "Lower infrastructure costs with edge deployment"
            ],
            "cons": [
                "Fly.io has limited HIPAA compliance certifications",
                "Smaller ecosystem for healthcare-specific integrations",
                "FastAPI is newer with fewer production references in healthcare",
                "Limited managed database options"
            ],
            "cost_estimate": 65000,
            "recommendation_score": 5,
            "recommended": False
        }
    ]
}

with open("$WORKSPACE/analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)

print(f"Generated analysis with {len(analysis['options'])} technology options")
PYEOF

echo "Analysis saved to $WORKSPACE/analysis.json"
