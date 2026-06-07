#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

python3 << PYEOF
import json

stories = [
    {
        "role": "user",
        "action": "create new tasks with a title, description, and due date",
        "benefit": "I can capture and track my work items efficiently"
    },
    {
        "role": "user",
        "action": "set priority levels on tasks (low, medium, high, urgent)",
        "benefit": "I can focus on the most important work first"
    },
    {
        "role": "user",
        "action": "mark tasks as complete and archive them",
        "benefit": "I can track my progress and keep my active list clean"
    },
    {
        "role": "user",
        "action": "organize tasks into projects and apply custom labels",
        "benefit": "I can group related work and find tasks easily"
    },
    {
        "role": "user",
        "action": "search tasks by keyword, label, or date range",
        "benefit": "I can quickly find specific tasks without scrolling"
    },
    {
        "role": "user",
        "action": "share tasks with team members and add comments",
        "benefit": "I can collaborate with my team on shared work items"
    },
    {
        "role": "user",
        "action": "configure notification preferences and quiet hours",
        "benefit": "I receive timely reminders without being overwhelmed"
    },
    {
        "role": "system",
        "action": "send reminder notifications before task due dates",
        "benefit": "users are alerted before deadlines pass"
    },
    {
        "role": "system",
        "action": "automatically back up data every 24 hours and sync offline changes",
        "benefit": "user data is protected and available across devices"
    },
    {
        "role": "admin",
        "action": "manage user accounts, permissions, and view usage analytics",
        "benefit": "the organization can control access and monitor productivity"
    },
    {
        "role": "admin",
        "action": "export all task data in CSV format",
        "benefit": "organizational data can be analyzed externally or archived"
    }
]

with open("$WORKSPACE/user_stories.json", "w") as f:
    json.dump(stories, f, indent=2)

print(f"Generated {len(stories)} user stories")
PYEOF

echo "User stories saved to $WORKSPACE/user_stories.json"
