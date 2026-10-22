## What was fixed this session (2026-09-16 — full-codebase pre-release audit)

Full debug review of ~92k lines Python / ~20k JS before market entry. Verdict:
**structured and disciplined, not spaghetti** — clean layering, typed event
dataclasses, consistent error sanitizing, zero TODO debt, 5,000+ tests. Four
real defects found and fixed (commit `a7be3eb`, pinned by
tests/unit/test_prerelease_audit_fixes.py):

1. **analyze_match discarded its own computed results** —
   `_compute_team_stats()` ran, then the returned `MatchAnalysis` used freshly
   zeroed TeamStats objects (only possession_pct was overwritten), and the
   computed away-side PPDA was dropped entirely; `ppda_breakdown` had zero
   consumers ever. Fixed by using the computed stats objects (no
   double-counting: `_compute_player_stats` fills only physical fields) and
   populating `ppda_breakdown` with both sides.
2. **EnhancedXgModel.compute_single shared a class-level lru_cache** — the
   cache keys on the features dataclass alone, so two instances with different
   coefficients could serve each other's cached xG. Replaced with per-instance
   memoization (dict on self, same 128-entry clear-when-full policy).
3. **OAuth HTTP calls had no timeout** — token exchange/userinfo/refresh
   could hang a login indefinitely; now 10s each.
4. **decrypt_dict failed silently on medical/PII fields** — a wrong or
   rotated key was indistinguishable from "no data"; now logs field name and
   error class. (Note for future tests: the project logs via loguru and the
   unit-suite stubs replace loguru wholesale — assert on the logger object
   with patch.object, caplog cannot see these records.)

Verified: 5,106 tests — 5,058 passed, 6 documented env-only failures, 0
errors; frontend 122/122; import + CLI smoke OK; ruff clean on all changed
hunks. Remaining documented hotspots (style-level, not defects): ~250 unused
imports, 87 exception swallows (sampled — nearly all intentional
skip-if-unavailable paths), `confidence_breakdown` computed-but-unconsumed,
and the B019 lru_cache-on-method pattern in epv/xa/xt models (bounded
retention only — those models are never mutated in place, unlike the xG one
that had to be fixed).

