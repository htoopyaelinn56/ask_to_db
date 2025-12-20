import json
import traceback

import psycopg2
import psycopg2.extras
from ai_service import gemini_client, DEFAULT_MODEL
from db_service import get_connection
from embedding_service import embed_text


# ---------------------------------------------------------
# 1. DATABASE SCHEMA & HELPERS
# ---------------------------------------------------------
def get_table_schema():
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


def _to_pgvector_literal(vec: list[float]) -> str:
    return "[" + ", ".join(f"{float(x):.6f}" for x in vec) + "]"


# ---------------------------------------------------------
# 2. DATA RETRIEVAL FUNCTIONS
# ---------------------------------------------------------

def execute_sql_query(sql_query: str):
    """Executes a generated SQL query safely."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            if cur.description:
                columns = [desc[0] for desc in cur.description]
                results = cur.fetchall()
                return columns, results
            return None, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()


def retrieve_similar_products(query: str, top_k: int = 5):
    query_vec = embed_text(query)
    vec_literal = _to_pgvector_literal(query_vec)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT id,
                       name,
                       name_mm,
                       description,
                       description_mm,
                       category,
                       brand,
                       price,
                       stock_quantity,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM products
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                """,
                (vec_literal, vec_literal, top_k),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def retrieve_similar_shop_info(query: str, top_k: int = 5):
    query_vec = embed_text(query)
    vec_literal = _to_pgvector_literal(query_vec)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT id,
                       chunk_index,
                       chunk_text,
                       contextualized_text,
                       -- Calculate cosine similarity (1 - cosine distance)
                       1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
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


# ---------------------------------------------------------
# 3. CONTEXT BUILDERS
# ---------------------------------------------------------

def build_context_for_products(products: list[dict]) -> str:
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


def build_context_for_shop_info(info_chunks: list[dict]) -> str:
    if not info_chunks:
        return "No relevant shop information found."

    lines = ["Here is the relevant shop information:\n"]
    for idx, chunk in enumerate(info_chunks, start=1):
        # The rows from `document_chunks` use the column names `contextualized_text` and `chunk_text`.
        # Older code used `text` which is not present in the query results and caused empty context.
        text = chunk.get('contextualized_text') or chunk.get('chunk_text') or chunk.get('text') or ""
        # Keep a concise single-line entry per chunk
        lines.append(f"{idx}. {text}")
        lines.append("")
    return "\n".join(lines)


def get_sql_data_context(sub_query: str, model: str) -> str:
    """Generates SQL and returns the raw result as context string."""
    schema = get_table_schema()
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
    
    User Question: {sub_query}
    SQL:
    """

    response = gemini_client.models.generate_content(model=model, contents=sql_prompt)
    generated_sql = response.text.replace("```sql", "").replace("```", "").strip()

    cols, results = execute_sql_query(generated_sql)
    if isinstance(results, str):
        return f"Database data could not be retrieved. Error: {results}"
    return f"Analytical Data (from query {generated_sql}): {results}"


# ---------------------------------------------------------
# 4. ROUTER & DECOMPOSER (Option 2 Implementation)
# ---------------------------------------------------------

def route_and_decompose_query(user_query: str, previous_context: str, model: str) -> list[dict]:
    """
    Analyzes the query and breaks it into sub-tasks with specific intents.
    """
    router_prompt = f"""
    You are an intelligent query router. Break the user query into independent sub-tasks.
    Also reference and consider the previous conversation context if relevant.
    If the question is vague, try to infer the user's intent based on the previous context as provided below.
    Previous Context: {previous_context}
    
    Assign one intent to each sub-task: 
    1. "sql": For questions about:
       - Show me products by brand/category or all products
       - Counting items (how many, total number)
       - Aggregations (average price, max price, sum)
       - Strict Filtering (products under $50, price > 100)
       - Checking stock levels specifically
       - Any analytical query requiring precise data retrieval
       
    2. "semantic_product": For questions about:
       - Finding products by description ("comfortable shoes")
       - Features, recommendations, or general info about products
       - "Do you have something like X?"
       - "I want to buy x and ....."
       
    3. "semantic_shop": For questions about:
        - Shop policies, shipping, returns, store hours
        - Information other than products in the database
        
    ** If the user asks for products, check in both vector database and postgres which means
       query in both sql and semantic_product to handle cases where user typo is wrong or inaccurate. **

    Return ONLY a JSON list of objects.
    Example: 
    Query: "Show me shoes under $50 and do you have free shipping?"
    Output: [
      {{"sub_query": "products under 50 dollars", "intent": "sql"}},
      {{"sub_query": "free shipping policy", "intent": "semantic_shop"}}
    ]
    
    Example: 
    Query: "hi, is there iPhone in stock, explain about it if available and where is the shop located"
    Output: [
      {{"sub_query": "iPhone stock available", "intent": "sql"}},
      {{"sub_query": "about iPhone", "intent": "semantic_product"}},
      {{"sub_query": "shop address", "intent": "semantic_shop"}},
    ]
    
     Example: 
     Query: "is there Toner alpha in stock and can I come to shop?"
     Output: [
      {{"sub_query": "Toner alpha available", "intent": "sql"}},
      {{"sub_query": "Toner alpha description", "intent": "semantic_product"}},
      {{"sub_query": "shop address", "intent": "semantic_shop"}},
     ]

    User Query: {user_query}
    Output:
    """

    print("[DEBUG] Router Prompt:", router_prompt)

    response = gemini_client.models.generate_content(
        model=model,
        contents=router_prompt,
    )
    print("[DEBUG] Router Response:", response.text)
    try:
        return json.loads(response.text.replace("```json", "").replace("```", "").strip())
    except:
        return [{"sub_query": user_query, "intent": "semantic_shop"}]


# ---------------------------------------------------------
# 5. MAIN CHATBOT LOGIC (The Synthesizer)
# ---------------------------------------------------------


SYSTEM_PROMPT = """You are a helpful product assistant.
Instructions:
- Use the provided context to answer.
- Answer ONLY in Myanmar (Burmese) language.
- You can keep Product names, brand names, and technical terms in English.
- If multiple pieces of information are requested, combine them into one smooth response.
- **IMPORTANT: If the user ask about stock or availability, just say in stock or out of stock, don't say exact stock number.**
"""


def chat_with_rag_stream(prompt: str, previous_message: str, model: str = DEFAULT_MODEL, top_k: int = 5):
    # 1. DECOMPOSE
    print("[DEBUG] Full Prompt for Decomposition:", prompt)
    sub_tasks = route_and_decompose_query(prompt, previous_message, model)

    # 2. RETRIEVE DATA FOR EACH TASK
    combined_contexts = []
    for task in sub_tasks:
        intent = task.get('intent')
        sub_q = task.get('sub_query')

        if intent == "sql":
            combined_contexts.append(get_sql_data_context(sub_q, model))
        elif intent == "semantic_product":
            similar = retrieve_similar_products(sub_q, top_k=top_k)
            combined_contexts.append(build_context_for_products(similar))
        elif intent == "semantic_shop":
            similar = retrieve_similar_shop_info(sub_q, top_k=top_k)
            combined_contexts.append(build_context_for_shop_info(similar))

    # 3. SYNTHESIZE & STREAM
    full_context = "\n\n".join(combined_contexts)
    final_input = f"""
    System Instructions: {SYSTEM_PROMPT}
    Gathered Context:
    {full_context}
    
    User Question: {prompt}
    Answer (in Myanmar):
    """

    stream = gemini_client.models.generate_content_stream(model=model, contents=final_input)
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def chat_with_rag(prompt: str, previous_message: str, model: str = DEFAULT_MODEL, top_k: int = 5):
    for text in chat_with_rag_stream(prompt=prompt, previous_message=previous_message, model=model, top_k=top_k):
        print(text.rstrip(), end="", flush=True)
    print()


def main():
    print("RAG Hybrid Chatbot (SQL + Vector + Shop Info)")
    print("Commands: exit, quit\n")
    model = DEFAULT_MODEL
    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input: continue
            if user_input.lower() in {"exit", "quit"}: break
            chat_with_rag(prompt=user_input, previous_message="", model=model)
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
