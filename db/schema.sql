-- RxClarify schema. Applied automatically by docker-compose on first boot of an
-- empty pgvector volume, and idempotent enough to re-run by hand:
--   docker compose exec -T db psql -U rx -d rxclarify < db/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per SPL (Structured Product Label) fetched from openFDA.
CREATE TABLE IF NOT EXISTS labels (
    id             BIGSERIAL PRIMARY KEY,
    set_id         TEXT NOT NULL UNIQUE,   -- openFDA openfda.spl_set_id; stable across label revisions
    spl_id         TEXT,                   -- revision-specific id
    query_generic  TEXT NOT NULL,          -- the generic name we searched for (our drug_list.yml key)
    brand_name     TEXT,
    generic_name   TEXT,
    manufacturer   TEXT,
    effective_time TEXT,
    raw            JSONB NOT NULL,
    fetched_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS labels_query_generic_idx ON labels (query_generic);

-- Retrievable units. `text` is the human-readable passage shown to the model;
-- the embedding is computed over a context-prefixed variant (see ingest/chunk.py)
-- so the drug name is in the vector even when the paragraph omits it.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    label_id    BIGINT NOT NULL REFERENCES labels (id) ON DELETE CASCADE,
    section     TEXT NOT NULL,             -- e.g. drug_interactions, boxed_warning
    ordinal     INTEGER NOT NULL,          -- position within (label, section)
    text        TEXT NOT NULL,
    char_count  INTEGER NOT NULL,
    embedding   vector(384),               -- BAAI/bge-small-en-v1.5
    -- Sparse half of the Phase 2 hybrid retriever. Created now because adding a
    -- generated column + GIN index later means re-indexing the whole table.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (label_id, section, ordinal)
);

-- Dense ANN. Cosine because bge embeddings are normalized.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Sparse full-text (unused in Phase 1, used by hybrid retrieval in Phase 2).
CREATE INDEX IF NOT EXISTS chunks_tsv_gin_idx ON chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_label_id_idx ON chunks (label_id);
CREATE INDEX IF NOT EXISTS chunks_section_idx ON chunks (section);
