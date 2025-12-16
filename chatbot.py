import psycopg2
import psycopg2.extras

from ai_service import gemini_client, DEFAULT_MODEL
from db_service import get_connection
from embedding_service import embed_text

def _to_pgvector_literal(vec: list[float]) -> str:
    """Convert a list of floats into pgvector text representation: [v1, v2, ...]."""
    return "[" + ", ".join(f"{float(x):.6f}" for x in vec) + "]"


def retrieve_similar_products(query: str, top_k: int = 5):
    """
    Embed the query, search for top_k most similar products using cosine distance,
    and return a list of product dicts with their similarity scores.
    """
    # Embed the user query
    query_vec = embed_text(query)
    vec_literal = _to_pgvector_literal(query_vec)

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Use cosine distance: 1 - (embedding <=> query_embedding)
            # Smaller distance = more similar
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
    """Format the retrieved products into a context string for the LLM."""
    if not products:
        return "No relevant products found in the database."

    lines = ["Here are the relevant products from our database:\n"]
    for idx, p in enumerate(products, start=1):
        lines.append(f"{idx}. **{p.get('name')}** ({p.get('name_mm')})")
        lines.append(f"   - Category: {p.get('category')}, Brand: {p.get('brand')}")
        lines.append(f"   - Price: ${p.get('price')}, Stock: {p.get('stock_quantity')}")
        if p.get('description'):
            lines.append(f"   - Description: {p.get('description')}")
        if p.get('description_mm'):
            lines.append(f"   - Description (MM): {p.get('description_mm')}")
        lines.append(f"   - Similarity: {p.get('similarity', 0):.3f}")
        lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a helpful product assistant with access to a product database.
Your role is to answer user questions about products based ONLY on the context provided below.

Instructions:
- Use the product information from the context to answer questions accurately to give easy and understandable response 
  for user friendly way informally.
- If the context doesn't contain relevant information, say "I don't have information about that in our database."
- Support both English and Myanmar (Burmese) language queries.
- Only respond in the language of the user's question.
- If the question is about greeting or similar, respond politely without using the context
- Be concise and helpful.
- Do not make up information that isn't in the context.
"""


def chat_with_rag(prompt: str, model: str = DEFAULT_MODEL, top_k: int = 5):
    """Retrieve relevant products, build context, and chat with RAG-enhanced prompt."""
    # Retrieve similar products
    products = retrieve_similar_products(prompt, top_k=top_k)
    context = build_context(products)

    # Build messages with system prompt and context
    messages = f"""
            System Instructions:
            {SYSTEM_PROMPT}
            
            Context Information:
            {context}
            
            User Question: {prompt}
            
            Answer:"""

    # Stream response
    stream = gemini_client.models.generate_content_stream(
        model=model,
        contents=messages,
    )
    for chunk in stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()  # newline after streaming completes


def main():
    print("RAG-Enhanced Product Chatbot")
    print("Ask questions about products in English or Myanmar.")
    print("Commands: /exit, /quit to exit\n")

    model = DEFAULT_MODEL
    while True:
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

        # RAG-enhanced chat
        try:
            chat_with_rag(user_input, model=model, top_k=5)
        except Exception as e:
            print(f"[ERROR] Chat failed: {e}")


if __name__ == "__main__":
    main()
