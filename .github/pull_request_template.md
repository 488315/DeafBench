## Summary

Explain the change in two or three direct sentences.

## Problem

Describe the incorrect behavior, missing capability, or requirement.

## Root cause

Explain the underlying cause, or write `N/A` for a new feature.

## Solution

Explain the implementation, ownership boundaries, and important decisions.

## Behavior changes

List observable changes, or write `None`.

## Validation

- [ ] Focused tests
- [ ] Full test suite, or reason it was not run
- [ ] Ruff
- [ ] Bytecode compilation
- [ ] `git diff --check`
- [ ] Frozen benchmark and generated-artifact integrity checks, when applicable

## Risk and rollback

Describe likely failure modes and how to revert safely.

## Dependencies

List prerequisites, or write `None`.

## Review focus

Identify the files, behavior, and failure paths needing closest review.

## Checklist

- [ ] This pull request contains one logical change.
- [ ] I reviewed the complete diff.
- [ ] Unrelated formatting and refactoring are excluded.
- [ ] New behavior has tests where practical.
- [ ] Failure paths and invalid input are handled.
- [ ] Logs and artifacts do not expose customer data, transcripts, or secrets.
- [ ] Documentation was updated when contracts changed.
- [ ] All reviewer conversations are resolved.
