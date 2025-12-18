-- 1. Enable pgvector extension (Standard in Supabase/Neon/RDS Postgres 15+)
CREATE
EXTENSION IF NOT EXISTS vector;

-- 2. Create the Products Table
DROP TABLE IF EXISTS products;
CREATE TABLE products
(
    id             SERIAL PRIMARY KEY,
    name           TEXT NOT NULL,
    name_mm        TEXT NOT NULL, -- Myanmar (Burmese) name
    description    TEXT,
    description_mm TEXT,          -- Myanmar (Burmese) description
    category       TEXT,          -- e.g., 'Electronics', 'Footwear'
    brand          TEXT,          -- e.g., 'Apple', 'Nike'
    price          DECIMAL(10, 2),
    stock_quantity INTEGER,       -- To test "Is this in stock?"
    serialized_text TEXT,          -- Text used for generating embeddings id:$id, name:$name, description:$description, ...
    embedding      vector(768)    -- Change 1536 to 768 or 384 depending on your model
);

DROP TABLE IF EXISTS document_chunks;
-- Create the document_chunks table
CREATE TABLE document_chunks (
                                 id SERIAL PRIMARY KEY,
                                 chunk_index INTEGER NOT NULL,
                                 chunk_text TEXT NOT NULL,
                                 contextualized_text TEXT NOT NULL,
                                 chunk_tokens INTEGER NOT NULL,
                                 contextualized_tokens INTEGER NOT NULL,
                                 embedding vector(768),  -- Adjust dimension based on your embedding model
                                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Add a unique constraint to prevent duplicate chunks
                                 UNIQUE(chunk_index)
);

-- Create indexes for better query performance
CREATE INDEX idx_chunk_index ON document_chunks(chunk_index);

-- Create a vector similarity search index using HNSW (Hierarchical Navigable Small World)
-- This significantly speeds up similarity searches
CREATE INDEX idx_embedding ON document_chunks
    USING hnsw (embedding vector_cosine_ops);
