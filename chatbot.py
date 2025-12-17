import psycopg2
import psycopg2.extras
import json
import re

from ai_service import gemini_client, DEFAULT_MODEL
from db_service import get_connection
from embedding_service import embed_text

# ---------------------------------------------------------
# 1. HELPER: DATABASE SCHEMA DEFINITION
# ---------------------------------------------------------
def get_table_schema():
    """Returns the schema representation for the LLM to understand the DB structure."""
    return """
    Table: products
    Columns:
    - id (integer)
    - name (text): Product name in English
    - name_mm (text): Product name in Myanmar
    - description (text): English description
    - description_mm (text): Myanmar description
    - category (text)
    - brand (text)
    - price (numeric): The cost of the item
    - stock_quantity (integer): How many items are available
    """

# ---------------------------------------------------------
# 2. EXISTING SEMANTIC SEARCH LOGIC
# ---------------------------------------------------------
def _to_pgvector_literal(vec: list[float]) -> str:
    return "[" + ", ".join(f"{float(x):.6f}" for x in vec) + "]"

def retrieve_similar_products(query: str, top_k: int = 5):
    query_vec = embed_text(query)
    vec_literal = _to_pgvector_literal(query_vec)

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT id, name, name_mm, description, description_mm,
                       category, brand, price, stock_quantity,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM products
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                """,
                (vec_literal, vec_literal, top_k),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    finally:
        conn.close()

def build_context(products: list[dict]) -> str:
    if not products:
        return "No relevant products found in the database."

    lines = ["Here are the relevant products from our database:\n"]
    for idx, p in enumerate(products, start=1):
        lines.append(f"{idx}. **{p.get('name')}** ({p.get('name_mm')})")
        lines.append(f"   - Category: {p.get('category')}, Brand: {p.get('brand')}")
        lines.append(f"   - Price: ${p.get('price')}, Stock: {p.get('stock_quantity')}")
        lines.append(f"   - Description: {p.get('description')}")
        lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------
# 3. NEW: SQL QUERY LOGIC (For Counts, Averages, Filters)
# ---------------------------------------------------------
def execute_sql_query(sql_query: str):
    """Executes a generated SQL query safely."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            # Fetch results (handle case where query returns nothing)
            if cur.description:
                columns = [desc[0] for desc in cur.description]
                results = cur.fetchall()
                return columns, results
            return None, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

def handle_sql_query(user_query: str, model: str):
    """Generator that:
    1. Generates SQL based on user question.
    2. Runs SQL.
    3. Streams a synthesized natural-language answer.
    Yields text chunks suitable for streaming to the client.
    """
    schema = get_table_schema()

    # A. Generate SQL
    sql_prompt = f"""
    You are a SQL expert. Convert the user question into a standard PostgreSQL query.
    
    {schema}
    
    Rules:
    1. return ONLY the SQL string. No markdown, no explanations.
    2. Use 'ilike' for text matching.
    3. If the user asks for 'products lower than 50', use 'WHERE price < 50'.
    4. If the user asks for 'count', use 'COUNT(*)'.
    5. Do not use INSERT, UPDATE, or DELETE. Read-only.
    6. Only Select name column when listing products.
    7. Make sure the SQL is valid PostgreSQL and references only existing columns.
    
    User Question: {user_query}
    SQL:
    """

    response = gemini_client.models.generate_content(
        model=model, contents=sql_prompt
    )

    # Clean up the response (remove ```sql ... ``` if present)
    generated_sql = response.text.replace("```sql", "").replace("```", "").strip()
    print(f"\n[DEBUG] Generated SQL: {generated_sql}")  # Useful for debugging

    # B. Execute SQL
    cols, results = execute_sql_query(generated_sql)

    # Error path from execute_sql_query
    if isinstance(results, str):
        yield f"I tried to calculate that, but encountered a database error: {results}"
        return

    # C. Formulate Answer
    answer_prompt = f"""
    User Question: {user_query}
    SQL Used: {generated_sql}
    Database Result: {results}
    
    Task: Answer the user's question naturally based on the database result. 
    - If the result is a number, just give the number context.
    - If it is a list of products, list them briefly.
    - If the result is empty, explain that no matching products were found.
    """

    # Stream the final answer
    stream = gemini_client.models.generate_content_stream(
        model=model, contents=answer_prompt
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


# ---------------------------------------------------------
# 4. NEW: ROUTER LOGIC
# ---------------------------------------------------------
def route_query(user_query: str, model: str) -> str:
    """
    Decides if the query needs 'semantic' search or 'sql' aggregation.
    """
    router_prompt = f"""
    You are a router. Classify the user query into one of two categories:
    
    1. "sql": For questions about:
       - Show me products by brand/category or all products
       - Counting items (how many, total number)
       - Aggregations (average price, max price, sum)
       - Strict Filtering (products under $50, price > 100)
       - Checking stock levels specifically
       - Any analytical query requiring precise data retrieval
       
    2. "semantic": For questions about:
       - Finding products by description ("comfortable shoes")
       - Features, recommendations, or general info
       - "Do you have something like X?"
    
    Return ONLY the word "sql" or "semantic".
    
    Query: {user_query}
    Category:
    """

    response = gemini_client.models.generate_content(
        model=model, contents=router_prompt
    )
    category = response.text.strip().lower()

    # Fallback to semantic if unsure
    if "sql" in category: return "sql"
    return "semantic"

# ---------------------------------------------------------
# 5. MAIN CHATBOT LOGIC
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful product assistant.
Instructions:
- Use the provided context to answer.
- Support both English and Myanmar (Burmese).
- Be concise and helpful.
"""

def chat_with_rag_stream(prompt: str, model: str = DEFAULT_MODEL, top_k: int = 5):
    # 1. ROUTER STEP
    intent = route_query(prompt, model)

    if intent == "sql":
        # SQL / analytical branch: stream chunks from handle_sql_query
        for chunk in handle_sql_query(prompt, model):
            if chunk:
                yield chunk
        return

    # 2. SEMANTIC STEP
    print(f"\n[DEBUG] SEMANTIC SEARCH")  # Useful for debugging
    products = retrieve_similar_products(prompt, top_k=top_k)
    context = build_context(products)

    messages = f"""
        System Instructions: {SYSTEM_PROMPT}
        Context Information: {context}
        User Question: {prompt}
        Answer:"""

    stream = gemini_client.models.generate_content_stream(
        model=model,
        contents=messages,
    )

    for chunk in stream:
        if chunk.text:
            yield chunk.text

def chat_with_rag(prompt: str, model: str = DEFAULT_MODEL, top_k: int = 5):
    for text in chat_with_rag_stream(prompt, model=model, top_k=top_k):
        print(text.rstrip(), end="", flush=True)
    print()

def main():
    print("RAG-Enhanced Product Chatbot (Router Enabled)")
    print("Examples:")
    print(" - Semantic: 'Find me a comfortable running shoe'")
    print(" - SQL: 'How many Nike products do we have?' or 'Show items under $50'")
    print("Commands: /exit, /quit\n")

    model = DEFAULT_MODEL
    while True:
        print()
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Goodbye.")
            break

        try:
            chat_with_rag(user_input, model=model)
        except Exception as e:
            print(f"[ERROR] Chat failed: {e}")

if __name__ == "__main__":
    main()