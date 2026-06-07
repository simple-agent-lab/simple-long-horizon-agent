# Task: Extract User Stories from Requirements

You are given a product requirements document for a todo application.

## Input Files

- `workspace/requirements.md` — product requirements document describing features of a todo app

## Goal

Extract user stories from the requirements and output them in a structured JSON format.

## Requirements

1. Read the requirements document carefully.
2. Identify all distinct user stories implied by the requirements.
3. Create `workspace/user_stories.json` containing an array of user story objects.
4. Each user story must have:
   - `role` — who the user is (one of: `"user"`, `"admin"`, `"system"`)
   - `action` — what they want to do (a clear, concise description)
   - `benefit` — why they want to do it (the value gained)
5. Extract at least 5 user stories from the document.
6. Stories should follow the pattern: "As a [role], I want to [action], so that [benefit]."

## Notes

- Cover the major feature areas described in the requirements.
- Each story should be specific enough to be actionable.
- Do not duplicate stories; each should address a distinct piece of functionality.
- Roles must be one of: `user`, `admin`, or `system`.
