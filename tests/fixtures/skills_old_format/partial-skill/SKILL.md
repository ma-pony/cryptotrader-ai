---
name: partial-skill
description: A skill that already has importance field but is missing the other new fields.
scope: shared
version: "1.0"
manually_edited: true
importance: 0.9
---

# Partial Skill

This skill already has the importance field set to 0.9 (manually edited).
The migration script should NOT overwrite this value.
Other new fields (regime_tags, triggers_keywords, access_count, last_accessed_at, confidence) should be added with defaults.
