# Task: Project Milestone Timeline

You are given a project brief describing a 3-month web application project.

## Input Files

- `workspace/project_brief.md` — a project brief describing the scope, team, and phases of a web app project

## Goal

Create a structured milestone timeline that breaks the project into phased milestones with dates, deliverables, and dependencies.

## Requirements

1. Read the project brief carefully.
2. Create `workspace/milestones.json` containing an array of milestone objects.
3. Each milestone must have:
   - `name` — descriptive name for the milestone
   - `start_date` — start date in ISO 8601 format (YYYY-MM-DD)
   - `end_date` — end date in ISO 8601 format (YYYY-MM-DD)
   - `deliverables` — array of strings describing what will be delivered
   - `dependencies` — array of milestone names this milestone depends on (empty for the first)
4. Milestones must be in chronological order.
5. Milestones must not overlap (a milestone's start_date must be >= previous milestone's end_date).
6. All 4 project phases described in the brief must be covered.
7. The timeline must fit within the 3-month project window starting 2025-01-06.

## Notes

- Dates must be valid ISO 8601 format (YYYY-MM-DD).
- Each milestone should have at least 2 deliverables.
- Dependencies should reference milestone names exactly as defined.
- The last milestone's end_date should not exceed 2025-04-04.
