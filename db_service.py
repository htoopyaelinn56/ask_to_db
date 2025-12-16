import psycopg2
import psycopg2.extras
from decimal import Decimal
from typing import Optional
import math

from embedding_service import embed_text

DB_NAME = "rag_test"
DB_USER = "postgres"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = 5432

EMBEDDING_DIM = 768


def get_connection():
    """
    Create and return a new psycopg2 connection to the Postgres database.
    Defaults to local Postgres on localhost:5432 with user 'postgres' and password 'password'.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    # Let callers control transactions; we'll explicitly commit/rollback around operations
    conn.autocommit = False
    return conn


# --- Helpers -----------------------------------------------------------------

def _nz(value: Optional[str]) -> str:
    """Normalize a possibly None string: strip and collapse whitespace; return '' for None."""
    if value is None:
        return ""
    # Collapse whitespace and strip
    s = " ".join(str(value).split())
    return s


def _fmt_price(value: Optional[Decimal]) -> str:
    if value is None:
        return ""
    try:
        return f"{Decimal(value):.2f}"
    except Exception:
        # Fallback to string
        return str(value)


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except Exception:
        return str(value)


def build_serialized_text(row: dict) -> str:
    """
    Build a consistent, language-inclusive serialized text from a product row.
    Includes English and Myanmar fields and key attributes so the embedding can
    support a range of queries.
    """
    parts = [
        f"id: {row.get('id')}",
        f"name: {_nz(row.get('name'))}",
        f"name_mm: {_nz(row.get('name_mm'))}",
        f"description: {_nz(row.get('description'))}",
        f"description_mm: {_nz(row.get('description_mm'))}",
        f"category: {_nz(row.get('category'))}",
        f"brand: {_nz(row.get('brand'))}",
        f"price: {_fmt_price(row.get('price'))}",
        f"stock_quantity: {_fmt_int(row.get('stock_quantity'))}",
    ]
    return " | ".join(parts)


def _to_pgvector_literal(vec: list[float]) -> str:
    """Convert a list of floats into pgvector text representation: [v1, v2, ...]."""
    # Limit precision to keep payload small while preserving information
    return "[" + ", ".join(f"{float(x):.6f}" if not (x is None or (isinstance(x, float) and math.isnan(x))) else "0.000000" for x in vec) + "]"


# --- Main backfill -----------------------------------------------------------

def set_embeddings(batch_size: int = 50):
    print("Setting up embeddings in the database...")

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Ensure extension exists (no-op if already installed)
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, name_mm, description, description_mm,
                       category, brand, price, stock_quantity, serialized_text, embedding
                FROM products
                WHERE serialized_text IS NULL OR embedding IS NULL
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall()

        if not rows:
            print("No rows need updates. serialized_text and embedding are already populated.")
            conn.commit()
            return

        print(f"Found {len(rows)} product(s) to update.")

        updated = 0
        with conn.cursor() as cur:  # simple cursor for updates
            for idx, row in enumerate(rows, start=1):
                # DictRow behaves like dict
                row_dict = dict(row)

                text = row_dict.get("serialized_text")
                if not text or not text.strip():
                    text = build_serialized_text(row_dict)

                try:
                    vec = embed_text(text)
                except Exception as e:
                    print(f"[WARN] Skipping id={row_dict.get('id')}: embedding failed: {e}")
                    continue

                if not isinstance(vec, list) or not vec:
                    print(f"[WARN] Skipping id={row_dict.get('id')}: invalid embedding output")
                    continue

                # Enforce dimensionality: truncate or pad to EMBEDDING_DIM
                if len(vec) < EMBEDDING_DIM:
                    vec = vec + [0.0] * (EMBEDDING_DIM - len(vec))
                elif len(vec) > EMBEDDING_DIM:
                    vec = vec[:EMBEDDING_DIM]

                vec_literal = _to_pgvector_literal(vec)

                # Update both fields to keep them consistent
                cur.execute(
                    """
                    UPDATE products
                    SET serialized_text = %s,
                        embedding = %s::vector
                    WHERE id = %s
                    """,
                    (text, vec_literal, row_dict["id"]),
                )
                updated += 1

                # Commit in batches to avoid large transactions
                if updated % batch_size == 0:
                    conn.commit()
                    print(f"Committed {updated} updates so far...")

            # Final commit for remaining updates
            conn.commit()

        print(f"Done. Updated {updated} product(s).")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Backfill failed and was rolled back: {e}")
        raise
    finally:
        conn.close()


set_embeddings()