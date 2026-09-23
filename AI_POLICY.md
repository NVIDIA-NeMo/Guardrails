# AI Usage Policy

NeMo Guardrails welcomes responsible AI-assisted contributions. AI tools can be
useful for exploration, implementation, review, and documentation, but the human
submitter remains responsible for the contribution.

## Contributor Responsibilities

- Disclose AI assistance in the pull request description when AI tools create or
  substantially modify code, tests, docs, issues, or comments. Include the tool
  used and the extent of assistance.
- Issues must be opened manually by a human through the repository issue
  templates. AI tools may help draft an issue, but agents must not submit issues
  directly.
- Review, edit, and verify AI-generated content before submitting it. Do not
  paste unreviewed AI output into issues, PR descriptions, code comments, docs,
  or review comments.
- Understand every submitted change well enough to explain what it does, why it
  is needed, and how it interacts with the surrounding code.
- Keep AI-assisted pull requests cohesive, scoped, and useful. Duplicate,
  low-value, mechanical, or noisy contributions may be closed.
- Do not add AI tools as commit co-authors. Contributions should be authored by
  the human submitter and must still satisfy the DCO or GPG-signing
  requirements in `CONTRIBUTING.md`.

## Safety and Privacy

- Do not commit API keys, credentials, private endpoints, proprietary prompts,
  raw provider logs, or sensitive request/response data.
- Do not use AI tools to fabricate test results, benchmark results, citations,
  maintainer approvals, or compatibility claims.
- Generated media, large generated assets, or synthetic datasets require clear
  provenance and maintainer alignment before inclusion.

## Maintainer Expectations

AI assistance does not lower the review bar. Maintainers may ask contributors to
explain, simplify, test, rewrite, or withdraw AI-assisted work that is not ready
for review.
