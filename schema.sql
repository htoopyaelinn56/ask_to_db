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