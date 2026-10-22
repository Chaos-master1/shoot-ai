"""Apply PostgreSQL schema using psql-compatible ordered execution."""

import os
import subprocess
import sys

schema_path = os.path.join(
    os.path.dirname(__file__), "..", "src", "kawkab", "migrations", "pg_schema.sql"
)
ordered_path = os.path.join(os.path.dirname(__file__), "..", "data", "pg_ordered.sql")

# Read schema and reorder tables by FK dependency
import re

with open(schema_path, encoding="utf-8") as f:
    raw = f.read()

# Extract all CREATE TABLE statements with their full bodies
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
        stmts[name] = body[:i]  # include closing )

# Topological sort
from collections import defaultdict, deque

fk_refs = {}
for name, body in stmts.items():
    refs = set(re.findall(r"REFERENCES\s+(\w+)\s*\(", body))
    refs.discard(name)
    fk_refs[name] = refs

graph = defaultdict(list)
in_deg = {}
for name in stmts:
    in_deg[name] = 0
for name, refs in fk_refs.items():
    for ref in refs:
        if ref in stmts:
            graph[ref].append(name)
            in_deg[name] += 1

q = deque([n for n, d in in_deg.items() if d == 0])
ordered = []
while q:
    n = q.popleft()
    ordered.append(n)
    for child in graph.get(n, []):
        in_deg[child] -= 1
        if in_deg[child] == 0:
            q.append(child)
# Add any remaining (cycles)
remaining = [n for n in stmts if n not in ordered]
ordered.extend(remaining)

print(f"Reordering {len(ordered)} tables by FK dependency")

# Build ordered SQL
ordered_sql = "-- Auto-generated ordered schema\nBEGIN;\n\n"

# Add non-table content (extensions, function, indexes, triggers, RLS)
# These are split on CREATE TABLE and reassembled
non_table_parts = []
for i, part in enumerate(parts):
    if i == 0:
        non_table_parts.append(part)
    else:
        name = stmts.get(
            part.split("\n")[0].strip() if "\n" in part else part.split("(")[0].strip()
        )
        if name and name in stmts:
            pass  # skip, we'll add it in order
        else:
            non_table_parts.append(part)

# Actually, let's just extract indexes, triggers, RLS etc from the raw
# Remove all CREATE TABLE statements from raw
no_tables = raw
for name, body in stmts.items():
    pattern = f"CREATE TABLE IF NOT EXISTS {name} ({body})"
    no_tables = no_tables.replace(pattern, "", 1)

# Add tables in dependency order
for name in ordered:
    ordered_sql += f"CREATE TABLE IF NOT EXISTS {name} ({stmts[name]})\n\n"

# Add everything else (indexes, triggers, RLS)
ordered_sql += "\n-- Indexes, triggers, RLS\n"
ordered_sql += no_tables.strip()
ordered_sql += "\n\nCOMMIT;\n"

with open(ordered_path, "w", encoding="utf-8") as f:
    f.write(ordered_sql)

print(f"Written to {ordered_path}")

# Execute via psql. Reads the password from PGPASSWORD (already the
# standard libpq env var psql itself understands) rather than a
# hardcoded value -- this script used to embed a real local Postgres
# password directly in source.
pg_password = os.environ.get("PGPASSWORD")
if not pg_password:
    sys.exit("PGPASSWORD is not set. Export it first (or use a .pgpass file).")

print("Executing via psql...")
result = subprocess.run(
    ["psql", "-U", "postgres", "-d", "kawkab", "-f", ordered_path],
    capture_output=True,
    text=True,
    encoding="utf-8",
    env={"PGPASSWORD": pg_password, **os.environ},
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print(f"Return code: {result.returncode}")
