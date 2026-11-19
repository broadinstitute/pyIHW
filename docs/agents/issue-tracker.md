# Issue tracker: GitHub

Use `gh` for GitHub Issues in `broadinstitute/pyIHW`. Verify the target with `gh repo view --json nameWithOwner` before publishing.

- Publish a ticket or spec with `gh issue create`; fetch one with `gh issue view`.
- Use `gh issue list` for queues and `gh issue edit` for labels or assignment.
- Use GitHub sub-issues and native blocking dependencies. If unavailable, record `Part of #<number>` or `Blocked by: #<number>` in the issue body.
- Record the resolution with `gh issue comment` before closing with `gh issue close`.

**PRs as a request surface: no.**
