import psycopg2
import psycopg2.extras
from decimal import Decimal
from typing import Optional
import math

from embedding_service import embed_text, generate_chunks_for_about_shop

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

def set_embeddings_for_products(batch_size: int = 50):
    print("Setting up embeddings for ALL products (Forced Update)...")

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Ensure extension exists (no-op if already installed)
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # CHANGE 1: Removed "WHERE serialized_text IS NULL..."
            # We fetch all products to ensure everything is up to date.
            cur.execute(
                """
                SELECT id, name, name_mm, description, description_mm,
                       category, brand, price, stock_quantity
                FROM products
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall()

        if not rows:
            print("No products found in the database.")
            return

        print(f"Found {len(rows)} product(s) to process.")

        updated = 0
        with conn.cursor() as cur:  # simple cursor for updates
            for idx, row in enumerate(rows, start=1):
                # DictRow behaves like dict
                row_dict = dict(row)

                # CHANGE 2: Always rebuild the text.
                # Since we want to update embeddings based on potential data changes,
                # we must regenerate the text from the current column values.
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

                # Update both fields
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
        print(f"[ERROR] Update failed and was rolled back: {e}")
        raise
    finally:
        conn.close()

# Flag to ensure we only reset the document_chunks table once per run
_document_chunks_initialized = False


def _init_document_chunks_table(conn):
    """Ensure pgvector extension and document_chunks table exist, then clear it.

    This runs TRUNCATE so each run of generate_chunks_for_about_shop starts
    from a clean slate.
    """
    with conn.cursor() as cur:
        # Ensure pgvector extension exists
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create table if it does not exist (idempotent)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                contextualized_text TEXT NOT NULL,
                chunk_tokens INTEGER NOT NULL,
                contextualized_tokens INTEGER NOT NULL,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chunk_index)
            );
            """
        )

        # Clear existing data so we always have the latest representation
        cur.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")

    conn.commit()


def set_embedding_about_shop(chunk_index, chunk_text, ser_txt, txt_tokens, ser_tokens):
    """Callback used by generate_chunks_for_about_shop to persist each chunk.

    For the first invocation in a run, this will clear the document_chunks
    table, then insert each chunk with its embedding and token counts.
    """
    global _document_chunks_initialized

    conn = get_connection()
    try:
        # Initialize table and clear existing rows once per run
        if not _document_chunks_initialized:
            _init_document_chunks_table(conn)
            _document_chunks_initialized = True

        # Compute embedding from contextualized text
        try:
            vec = embed_text(ser_txt)
        except Exception as e:
            print(f"[WARN] Skipping chunk_index={chunk_index}: embedding failed: {e}")
            return

        if not isinstance(vec, list) or not vec:
            print(f"[WARN] Skipping chunk_index={chunk_index}: invalid embedding output")
            return

        # Enforce dimensionality: truncate or pad to EMBEDDING_DIM
        if len(vec) < EMBEDDING_DIM:
            vec = vec + [0.0] * (EMBEDDING_DIM - len(vec))
        elif len(vec) > EMBEDDING_DIM:
            vec = vec[:EMBEDDING_DIM]

        vec_literal = _to_pgvector_literal(vec)

        chunk_tokens_count = len(txt_tokens) if txt_tokens is not None else 0
        ser_tokens_count = len(ser_tokens) if ser_tokens is not None else 0

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks (
                    chunk_index,
                    chunk_text,
                    contextualized_text,
                    chunk_tokens,
                    contextualized_tokens,
                    embedding
                ) VALUES (%s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (chunk_index) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    contextualized_text = EXCLUDED.contextualized_text,
                    chunk_tokens = EXCLUDED.chunk_tokens,
                    contextualized_tokens = EXCLUDED.contextualized_tokens,
                    embedding = EXCLUDED.embedding
                """,
                (
                    int(chunk_index),
                    chunk_text,
                    ser_txt,
                    int(chunk_tokens_count),
                    int(ser_tokens_count),
                    vec_literal,
                ),
            )

        conn.commit()
        print(f"[INFO] Upserted chunk_index={chunk_index}")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to persist chunk_index={chunk_index}: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    set_embeddings_for_products(batch_size=50)
    generate_chunks_for_about_shop(set_embedding_about_shop)