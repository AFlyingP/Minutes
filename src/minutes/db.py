import re
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool, PoolTimeout

from minutes.config import ROOT
from minutes.errors import DatabaseError, MigrationError

MIGRATIONS_DIR = ROOT / "migrations"
# down files have a second dot in the name and do not match
MIGRATION_FILE = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")

_pools: dict[str, ConnectionPool] = {}


def _register_vector(conn: psycopg.Connection[TupleRow]) -> None:
    register_vector(conn)
    # the type lookup opened a transaction, and the pool wants the connection idle
    conn.commit()


def get_pool(url: str) -> ConnectionPool:
    if url not in _pools:
        _pools[url] = ConnectionPool(
            url, min_size=1, max_size=8, timeout=10, configure=_register_vector, open=True
        )
    return _pools[url]


@contextmanager
def connect(url: str) -> Iterator[psycopg.Connection[TupleRow]]:
    """A pooled connection that commits on normal exit and rolls back on an exception."""
    try:
        with get_pool(url).connection() as conn:
            yield conn
    except PoolTimeout as err:
        raise DatabaseError("database unavailable") from err


def _open(url: str) -> psycopg.Connection[TupleRow]:
    # migrations run before the vector type exists, so they cannot use the pool
    try:
        return psycopg.connect(url, autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as err:
        raise DatabaseError(f"cannot connect to database: {err}") from err


def migrate(url: str) -> list[str]:
    """Apply the migration files not yet recorded, each in its own transaction."""
    files = sorted(p for p in MIGRATIONS_DIR.iterdir() if MIGRATION_FILE.match(p.name))
    applied = []
    with _open(url) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for path in files:
            version = path.name[:4]
            if version in done:
                continue
            try:
                with conn.transaction():
                    conn.execute(path.read_bytes())
                    conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            except psycopg.Error as err:
                raise MigrationError(f"{path.name}: {err}") from err
            applied.append(version)
    return applied


def rollback(url: str, steps: int = 1) -> list[str]:
    """Run the down files of the highest applied versions."""
    rolled_back = []
    with _open(url) as conn:
        rows = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT %s", (steps,)
        ).fetchall()
        for (version,) in rows:
            path = next(MIGRATIONS_DIR.glob(f"{version}_*.down.sql"))
            try:
                with conn.transaction():
                    conn.execute(path.read_bytes())
                    conn.execute("DELETE FROM schema_migrations WHERE version = %s", (version,))
            except psycopg.Error as err:
                raise MigrationError(f"{path.name}: {err}") from err
            rolled_back.append(version)
    return rolled_back


def reset(url: str) -> None:
    """Drop and recreate the database named in the URL, then migrate it."""
    parts = urlsplit(url)
    if parts.hostname not in ("localhost", "127.0.0.1"):
        raise DatabaseError("reset is allowed only on localhost")
    name = sql.Identifier(parts.path.lstrip("/"))
    with _open(parts._replace(path="/postgres").geturl()) as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(name))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(name))
    pool = _pools.pop(url, None)
    if pool is not None:
        pool.close()
    migrate(url)
