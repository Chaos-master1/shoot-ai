"""Apply PostgreSQL schema: creates all tables, FKs, indexes, triggers, RLS.

Reads the connection string from KAWKAB_DB_URL (same env var the rest of
the codebase uses, e.g. postgresql://user:pass@host:5432/dbname) rather
than a hardcoded DSN -- this script used to embed a real local
Postgres password directly in source.
"""

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
    conn.autocommit = False
    cur = conn.cursor()

    schema_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "kawkab", "migrations", "pg_schema.sql"
    )
    with open(schema_path, encoding="utf-8") as f:
        raw = f.read()

    # 1. Extensions + function (no table deps)
    for pattern in [
        r"CREATE EXTENSION IF NOT EXISTS pgcrypto;",
        r"CREATE OR REPLACE FUNCTION set_updated_at\(\)[^;]+LANGUAGE plpgsql;",
    ]:
        for m in re.finditer(pattern, raw, re.DOTALL):
            run(cur, m.group(), "extension/function")

    # 2. Parse all CREATE TABLE statements
    parts = raw.split("CREATE TABLE IF NOT EXISTS ")
    stmts = {}
    for part in parts[1:]:
        m = re.match(r"(\w+)\s*\((.*)", part, re.DOTALL)
        if m:
            name = m.group(1)
            body = m.group(2)
            # Find matching close paren
            depth, i = 1, 0
            while i < len(body) and depth > 0:
                if body[i] == "(":
                    depth += 1
                elif body[i] == ")":
                    depth -= 1
                i += 1
            body = body[: i - 1]  # exclude closing )
            stmts[name] = body

    # 3. Topological sort by FK deps
    fk_deps = {}
    for name, body in stmts.items():
        refs = re.findall(r"REFERENCES\s+(\w+)\s*\(", body)
        fk_deps[name] = [r for r in refs if r != name and r in stmts]

    sorted_tables = topological_sort(fk_deps)
    extra = [t for t in stmts if t not in sorted_tables]
    sorted_tables.extend(extra)
    print(f"Creating {len(sorted_tables)} tables in order:\n  {' -> '.join(sorted_tables)}")

    # 4. Create tables without FK constraints
    for tbl in sorted_tables:
        body = stmts[tbl]
        # Debug: show first table SQL
        if tbl == sorted_tables[0]:
            print(f"  DEBUG body for {tbl}: {repr(body[:200])}")
        # Remove FK constraints from body (both FOREIGN KEY clause and inline REFERENCES)
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
        sql = f"CREATE TABLE IF NOT EXISTS {tbl} ({body_clean})"
        run(cur, sql, tbl)

    # 5. Add FK constraints back
    on_clause_rx = r"(\s+ON\s+(?:DELETE|UPDATE)\s+(?:SET\s+)?\w+)?"
    fk_clause_rx = (
        rf"FOREIGN KEY\s*\(([^)]+)\)\s*REFERENCES\s+(\w+)\s*\(([^)]+)\){on_clause_rx}{on_clause_rx}"
    )
    inline_fk_rx = rf"(\w+)\s+(?:INTEGER|INT|BIGINT|SMALLINT)\s+REFERENCES\s+(\w+)\s*\(([^)]+)\){on_clause_rx}{on_clause_rx}"
    fk_added = 0
    for tbl in sorted_tables:
        body = stmts[tbl]
        # FOREIGN KEY (...) REFERENCES ... (...) ON DELETE/UPDATE ...
        for m in re.finditer(fk_clause_rx, body):
            cols, ref_tbl, ref_cols = m.group(1), m.group(2), m.group(3)
            if ref_tbl not in stmts:
                continue
            fk_name = f"fk_{tbl}_{ref_tbl}"
            sql = f"ALTER TABLE {tbl} ADD CONSTRAINT {fk_name} FOREIGN KEY ({cols}) REFERENCES {ref_tbl}({ref_cols})"
            for g in [m.group(4), m.group(5)]:
                if g:
                    sql += g
            run(cur, sql, f"{tbl} FK->{ref_tbl}", ignore_errors=True)
            fk_added += 1
        # Inline REFERENCES col_name type REFERENCES table(col)
        for m in re.finditer(inline_fk_rx, body):
            col, ref_tbl, ref_cols = m.group(1), m.group(2), m.group(3)
            if ref_tbl not in stmts:
                continue
            fk_name = f"fk_{tbl}_{col}"
            sql = f"ALTER TABLE {tbl} ADD CONSTRAINT {fk_name} FOREIGN KEY ({col}) REFERENCES {ref_tbl}({ref_cols})"
            for g in [m.group(4), m.group(5)]:
                if g:
                    sql += g
            run(cur, sql, f"{tbl}.{col}->{ref_tbl}", ignore_errors=True)
            fk_added += 1
    print(f"  Added {fk_added} FK constraints")

    # 6. Indexes
    idx_count = 0
    for m in re.finditer(
        r"CREATE\s+(UNIQUE\s+)?INDEX\s+IF NOT EXISTS\s+\w+\s+ON\s+\w+\s*\([^)]+\);",
        raw,
        re.IGNORECASE,
    ):
        run(cur, m.group(), "index", ignore_errors=True)
        idx_count += 1
    print(f"  Created {idx_count} indexes")

    # 7. Triggers
    for m in re.finditer(
        r"CREATE TRIGGER\s+\w+[^;]+EXECUTE FUNCTION\s+set_updated_at\(\)\s*;", raw, re.IGNORECASE
    ):
        run(cur, m.group(), "trigger", ignore_errors=True)
        print("  + trigger added")

    # 8. RLS
    for m in re.finditer(r"ALTER TABLE\s+\w+\s+ENABLE ROW LEVEL SECURITY;", raw, re.IGNORECASE):
        run(cur, m.group(), "RLS", ignore_errors=True)
    for m in re.finditer(
        r"CREATE POLICY\s+\w+\s+ON\s+\w+\s+FOR ALL\s+USING\s*\(true\);", raw, re.IGNORECASE
    ):
        run(cur, m.group(), "RLS policy", ignore_errors=True)

    conn.commit()

    # 9. Verify
    cur.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name"
    )
    tables = [r[0] for r in cur.fetchall()]
    print(f"\nSchema migration complete! {len(tables)} tables created.")
    missing = set(stmts.keys()) - set(tables)
    if missing:
        print(f"  Missing tables: {missing}")
    else:
        print(f"  All {len(stmts)} expected tables present.")

    cur.close()
    conn.close()


def run(cur, sql, label, ignore_errors=False):
    try:
        cur.execute(sql)
        return True
    except Exception as e:
        if ignore_errors:
            return False
        print(f"  FAILED {label}: {e}".encode("ascii", errors="replace").decode("ascii"))
        raise


def topological_sort(deps):
    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_deg = defaultdict(int)
    for name in deps:
        if name not in in_deg:
            in_deg[name] = 0
    for name, refs in deps.items():
        for ref in refs:
            graph[ref].append(name)
            in_deg[name] = in_deg.get(name, 0) + 1
    q = deque([n for n, d in in_deg.items() if d == 0])
    result = []
    while q:
        n = q.popleft()
        result.append(n)
        for child in graph.get(n, []):
            in_deg[child] -= 1
            if in_deg[child] == 0:
                q.append(child)
    return result


if __name__ == "__main__":
    main()
