# Session logs

Verbatim historical records of each implementation pass, extracted from
`CLAUDE.md` (2026-09-17) to stop that file growing linearly with every
session. Nothing was edited or summarized — each file is the original
section, cut out whole.

`CLAUDE.md` itself keeps only: what the project is, correct commands, the
architecture map, the invariants (with the reasons they exist), the current
known-gaps list, and the Graphify notes. The invariants and gaps that were
originally *introduced* inside these session logs remain in CLAUDE.md; only
the narrative "what was fixed" history moved here.

| Log | Pass |
|---|---|
| [2026-07-30-audit.md](2026-07-30-audit.md) | The original full audit: entrypoint/bridge/storage fixes, CI honesty, frontend wiring |
| [2026-09-07-elite-readiness.md](2026-09-07-elite-readiness.md) | Trained xG/PSxG/xT models, StatsBomb interop, Pro Analytics tranches 1+2 |
| [2026-09-16-commit-integrity-xg-angles.md](2026-09-16-commit-integrity-xg-angles.md) | Broken-HEAD repair, xG angle-convention fix, first real tracking benchmark |
| [2026-09-16-elite-implementation.md](2026-09-16-elite-implementation.md) | The 16-item "make it elite" pass (whiteboard, onboarding, bridge fix, release pipeline) |
| [2026-09-16-prerelease-audit.md](2026-09-16-prerelease-audit.md) | Full-codebase pre-release audit (4 real defects fixed) |
