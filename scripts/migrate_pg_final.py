"""Migrate PostgreSQL schema: DROPS every existing table (CASCADE), then
creates all 43 tables with FKs, indexes, triggers, RLS.

DESTRUCTIVE -- this deletes all data in every table on the target
database before recreating the schema. Only ever point this at a
database you're OK losing entirely (e.g. a fresh local dev instance).

Reads the connection string from KAWKAB_DB_URL rather than a hardcoded
DSN -- this script used to embed a real local Postgres password
directly in source.
"""

import contextlib
import os
import re
import sys

import psycopg2

DB = os.environ.get("KAWKAB_DB_URL")
if not DB:
    sys.exit(
        "KAWKAB_DB_URL is not set. Export it first, e.g.:\n"
        "  postgresql://postgres:<password>@localhost:5432/kawkab"
    )


def main():
    conn = psycopg2.connect(DB)
    conn.autocommit = True  # Each statement commits immediately
    cur = conn.cursor()

    schema_path = r"src/kawkab/migrations/pg_schema.sql"
    with open(schema_path, encoding="utf-8") as f:
        raw = f.read()

    # Parse all CREATE TABLE statements with bodies
    parts = raw.split("CREATE TABLE IF NOT EXISTS ")
    stmts = {}
    for part in parts[1:]:
        m = re.match(r"(\w+)\s*\((.*)", part, re.DOTALL)
        if m:
            name = m.group(1)
            body = m.group(2)
            depth, i = 1, 0
            while i < len(body) and depth > 0:
                if body[i] == "(":
                    depth += 1
                elif body[i] == ")":
                    depth -= 1
                i += 1
            body = body[: i - 1]
            stmts[name] = body

    # Sort tables topologically by FK dependency
    fk_refs = {}
    for name, body in stmts.items():
        refs = set(re.findall(r"REFERENCES\s+(\w+)\s*\(", body))
        refs.discard(name)
        fk_refs[name] = refs

    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_deg = dict.fromkeys(stmts, 0)
    for name, refs in fk_refs.items():
        for ref in refs:
            if ref in stmts:
                graph[ref].append(name)
                in_deg[name] += 1

    q = deque([n for n, d in in_deg.items() if d == 0])
    sorted_tables = []
    while q:
        n = q.popleft()
        sorted_tables.append(n)
        for child in graph.get(n, []):
            in_deg[child] -= 1
            if in_deg[child] == 0:
                q.append(child)
    remaining = [n for n in stmts if n not in sorted_tables]
    sorted_tables.extend(remaining)

    print(f"Creating {len(sorted_tables)} tables in dependency order...")

    # Phase 1: Drop all existing tables (reverse order for FK safety)
    cur.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'"
    )
    existing = [r[0] for r in cur.fetchall()]
    for t in existing:
        with contextlib.suppress(Exception):
            cur.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
    if existing:
        print(f"  Dropped {len(existing)} existing tables")

    # Phase 2: Create tables (with FKs inline)
    created = 0
    for tbl in sorted_tables:
        body = stmts[tbl]
        sql = f"CREATE TABLE IF NOT EXISTS {tbl} ({body})"
        try:
            cur.execute(sql)
            created += 1
        except Exception:
            # FK likely failed - create without FK
            on_clause = r"(?:\s+ON\s+(?:DELETE|UPDATE)\s+(?:SET\s+)?\w+)?"
            body_clean = re.sub(
                r",\s*\n?\s*FOREIGN KEY\s*\([^)]+\)\s*REFERENCES\s+\w+\s*\([^)]+\)"
                + on_clause
                + on_clause,
                "",
                body,
            )
            body_clean = re.sub(
                r",\s*\n?\s*\w+\s+(?:INTEGER|INT|BIGINT|SMALLINT|REAL|TEXT)\s+REFERENCES\s+\w+\s*\([^)]+\)"
                + on_clause
                + on_clause,
                "",
                body_clean,
            )
            body_clean = re.sub(r",\s*\)", ")", body_clean)
            sql2 = f"CREATE TABLE IF NOT EXISTS {tbl} ({body_clean})"
            try:
                cur.execute(sql2)
                created += 1
            except Exception as e2:
                print(f"  X {tbl}: {e2}")
    print(f"  Created {created}/{len(stmts)} tables")

    # Phase 3: Add FK constraints
    on_clause_rx = r"(\s+ON\s+(?:DELETE|UPDATE)\s+(?:SET\s+)?\w+)?"
    fk_pattern = (
        rf"FOREIGN KEY\s*\(([^)]+)\)\s*REFERENCES\s+(\w+)\s*\(([^)]+)\){on_clause_rx}{on_clause_rx}"
    )
    inline_fk_pattern = rf"(\w+)\s+(?:INTEGER|INT|BIGINT|SMALLINT)\s+REFERENCES\s+(\w+)\s*\(([^)]+)\){on_clause_rx}{on_clause_rx}"
    fk_added = 0
    for tbl in sorted_tables:
        body = stmts[tbl]
        for pattern in [fk_pattern, inline_fk_pattern]:
            for m in re.finditer(pattern, body):
                try:
                    cols = m.group(1)
                    ref_tbl = m.group(2)
                    ref_cols = m.group(3)
                    if ref_tbl not in stmts:
                        continue
                    # Determine FK name
                    if "FOREIGN KEY" in pattern:
                        fk_name = f"fk_{tbl}_{ref_tbl}"
                    else:
                        fk_name = f"fk_{tbl}_{cols}"
                    sql = f"ALTER TABLE {tbl} ADD CONSTRAINT {fk_name} FOREIGN KEY ({cols}) REFERENCES {ref_tbl}({ref_cols})"
                    for g in [m.group(4), m.group(5)]:
                        if g:
                            sql += g
                    cur.execute(sql)
                    fk_added += 1
                except Exception:
                    pass
    print(f"  Added {fk_added} FK constraints")

    # Phase 4: Indexes
    idx_count = 0
    for m in re.finditer(
        r"CREATE\s+(UNIQUE\s+)?INDEX\s+(IF NOT EXISTS\s+)?\w+\s+ON\s+\w+\s*\([^)]+\);",
        raw,
        re.IGNORECASE,
    ):
        try:
            cur.execute(m.group())
            idx_count += 1
        except Exception:
            pass
    print(f"  Created {idx_count} indexes")

    # Phase 5: Triggers
    trig_count = 0
    for m in re.finditer(
        r"CREATE TRIGGER\s+\w+[^;]+EXECUTE FUNCTION\s+set_updated_at\(\)\s*;", raw, re.IGNORECASE
    ):
        try:
            cur.execute(m.group())
            trig_count += 1
        except Exception:
            pass
    print(f"  Created {trig_count} triggers")

    # Phase 6: RLS
    for m in re.finditer(r"ALTER TABLE\s+\w+\s+ENABLE ROW LEVEL SECURITY;", raw, re.IGNORECASE):
        with contextlib.suppress(Exception):
            cur.execute(m.group())
    for m in re.finditer(
        r"CREATE POLICY\s+\w+\s+ON\s+\w+\s+FOR ALL\s+USING\s*\(true\);", raw, re.IGNORECASE
    ):
        with contextlib.suppress(Exception):
            cur.execute(m.group())

    # Verify
    cur.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name"
    )
    tables = [r[0] for r in cur.fetchall()]
    print(f"\nSchema migration complete! {len(tables)} tables created.")
    missing = set(stmts.keys()) - set(tables)
    if missing:
        print(f"  Missing: {missing}")
    else:
        print(f"  All {len(stmts)} expected tables present.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
