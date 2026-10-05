CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE cities (
  id          text PRIMARY KEY,
  name        text NOT NULL,
  state       text NOT NULL,
  item_format text NOT NULL CHECK (item_format IN ('legistar', 'granicus'))
);

CREATE TABLE meetings (
  id             text PRIMARY KEY,
  city_id        text NOT NULL REFERENCES cities(id),
  body           text NOT NULL,
  meeting_date   date NOT NULL,
  title          text NOT NULL,
  source_key     text NOT NULL,
  recording_url  text,
  recording_seek text CHECK (recording_seek IN ('granicus_entrytime')),
  UNIQUE (city_id, source_key)
);
-- serves: filter lists and facts ordering by city and date
CREATE INDEX meetings_city_date_idx ON meetings (city_id, meeting_date);

CREATE TABLE documents (
  id             text PRIMARY KEY,
  meeting_id     text NOT NULL REFERENCES meetings(id),
  city_id        text NOT NULL REFERENCES cities(id),
  kind           text NOT NULL CHECK (kind IN ('agenda', 'minutes', 'transcript')),
  source_url     text NOT NULL,
  media_type     text NOT NULL CHECK (media_type IN ('application/pdf', 'application/x-subrip', 'text/vtt')),
  unit_kind      text NOT NULL CHECK (unit_kind IN ('page', 'segment')),
  status         text NOT NULL DEFAULT 'discovered'
                 CHECK (status IN ('discovered', 'downloaded', 'duplicate', 'skipped', 'failed')),
  duplicate_of   text REFERENCES documents(id),
  content_sha256 text,
  byte_size      integer,
  file_path      text,
  unit_count     integer,
  fail_reason    text,
  created_at     timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now(),
  UNIQUE (meeting_id, kind)
);
-- serves: duplicate detection in the download stage
CREATE UNIQUE INDEX documents_city_hash_idx ON documents (city_id, content_sha256) WHERE status = 'downloaded';
-- serves: documents_for and corpus_stats
CREATE INDEX documents_city_kind_status_idx ON documents (city_id, kind, status);

CREATE TABLE units (
  id             bigserial PRIMARY KEY,
  document_id    text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  unit_index     integer NOT NULL CHECK (unit_index >= 1),
  unit_kind      text NOT NULL CHECK (unit_kind IN ('page', 'segment')),
  text           text NOT NULL,
  boxes          jsonb,
  text_source    text NOT NULL CHECK (text_source IN ('pdf', 'ocr', 'caption')),
  needs_ocr      boolean NOT NULL DEFAULT false,
  ocr_confidence real,
  start_ms       integer,
  end_ms         integer,
  speaker        text,
  width_pt       real,
  height_pt      real,
  UNIQUE (document_id, unit_index),
  CHECK ((unit_kind = 'segment') = (start_ms IS NOT NULL AND end_ms IS NOT NULL))
);

CREATE TABLE items (
  id               bigserial PRIMARY KEY,
  document_id      text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  meeting_id       text NOT NULL REFERENCES meetings(id),
  kind             text NOT NULL CHECK (kind IN ('business', 'section')),
  identifier       text NOT NULL,
  title            text NOT NULL,
  ordinal          integer NOT NULL,
  start_unit       integer NOT NULL,
  start_offset     integer NOT NULL,
  end_unit         integer NOT NULL,
  end_offset       integer NOT NULL,
  linked_item_id   bigint REFERENCES items(id) ON DELETE SET NULL,
  UNIQUE (document_id, ordinal)
);
-- serves: item linking per meeting and hit expansion
CREATE INDEX items_meeting_idx ON items (meeting_id, kind);

CREATE TABLE chunks (
  id           bigserial PRIMARY KEY,
  document_id  text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  unit_index   integer NOT NULL,
  item_id      bigint REFERENCES items(id) ON DELETE CASCADE,
  chunker      text NOT NULL CHECK (chunker IN ('fixed', 'item')),
  ordinal      integer NOT NULL,
  start_offset integer NOT NULL,
  end_offset   integer NOT NULL,
  text         text NOT NULL,
  embed_text   text NOT NULL,
  embedding    vector(768),
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  city_id      text NOT NULL,
  body         text NOT NULL,
  meeting_date date NOT NULL,
  doc_kind     text NOT NULL,
  UNIQUE (document_id, chunker, ordinal)
);
-- serves: the keyword query
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
-- serves: the vector query
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
-- serves: the filter clause of both queries
CREATE INDEX chunks_filter_idx ON chunks (chunker, city_id, meeting_date);
-- serves: hit expansion
CREATE INDEX chunks_item_idx ON chunks (item_id, ordinal);

CREATE TABLE facts (
  id            bigserial PRIMARY KEY,
  kind          text NOT NULL CHECK (kind IN ('motion', 'vote', 'ordinance', 'amount', 'statement')),
  document_id   text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  meeting_id    text NOT NULL REFERENCES meetings(id),
  city_id       text NOT NULL,
  body          text NOT NULL,
  meeting_date  date NOT NULL,
  item_id       bigint REFERENCES items(id) ON DELETE CASCADE,
  unit_index    integer NOT NULL,
  quote_start   integer,
  quote_end     integer,
  quote         text NOT NULL,
  data          jsonb NOT NULL,
  verified      boolean NOT NULL,
  verify_error  text,
  model         text
);
-- serves: the facts endpoints
CREATE INDEX facts_kind_city_date_idx ON facts (kind, city_id, meeting_date) WHERE verified;

CREATE TABLE jobs (
  id           bigserial PRIMARY KEY,
  stage        text NOT NULL,
  document_id  text NOT NULL,
  corpus       text NOT NULL CHECK (corpus IN ('fixture', 'dev', 'full')),
  status       text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed', 'dead')),
  attempts     integer NOT NULL DEFAULT 0,
  max_attempts integer NOT NULL DEFAULT 5,
  run_after    timestamptz NOT NULL DEFAULT now(),
  locked_by    text,
  locked_at    timestamptz,
  last_error   text,
  created_at   timestamptz NOT NULL DEFAULT now(),
  finished_at  timestamptz
);
-- serves: idempotent enqueue
CREATE UNIQUE INDEX jobs_active_idx ON jobs (stage, document_id) WHERE status IN ('queued', 'running');
-- serves: the claim query
CREATE INDEX jobs_claim_idx ON jobs (status, run_after, id);

CREATE TABLE stage_runs (
  document_id  text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  stage        text NOT NULL,
  input_sha256 text NOT NULL,
  status       text NOT NULL CHECK (status IN ('done', 'failed')),
  detail       jsonb NOT NULL DEFAULT '{}',
  finished_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (document_id, stage)
);

CREATE TABLE labels (
  id             text PRIMARY KEY,
  label_type     text NOT NULL CHECK (label_type IN ('question', 'agenda_count', 'extraction', 'agent_task')),
  city_id        text NOT NULL REFERENCES cities(id),
  payload        jsonb NOT NULL,
  author         text NOT NULL CHECK (author IN ('agent', 'human')),
  created_utc    timestamptz NOT NULL,
  human_reviewed boolean NOT NULL DEFAULT false,
  note           text NOT NULL DEFAULT ''
);
-- serves: label listing and rule checks
CREATE INDEX labels_type_city_idx ON labels (label_type, city_id);

CREATE TABLE llm_calls (
  id                bigserial PRIMARY KEY,
  created_at        timestamptz NOT NULL DEFAULT now(),
  correlation_id    text NOT NULL,
  purpose           text NOT NULL,
  model             text NOT NULL,
  cache_key         text NOT NULL,
  cached            boolean NOT NULL,
  prompt_tokens     integer NOT NULL,
  completion_tokens integer NOT NULL,
  cost_usd          numeric(12, 6) NOT NULL,
  latency_ms        integer NOT NULL
);
-- serves: cost per query by correlation ID
CREATE INDEX llm_calls_correlation_idx ON llm_calls (correlation_id);

CREATE TABLE answer_cache (
  key        text PRIMARY KEY,
  response   jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE VIEW v_meetings AS
  SELECT id, city_id, body, meeting_date, title FROM meetings;
CREATE VIEW v_documents AS
  SELECT d.id, d.meeting_id, d.city_id, d.kind, d.unit_kind, d.unit_count, m.body, m.meeting_date
  FROM documents d JOIN meetings m ON m.id = d.meeting_id WHERE d.status = 'downloaded';
CREATE VIEW v_units AS
  SELECT u.document_id, u.unit_index, u.unit_kind, u.text, u.start_ms, u.end_ms, u.speaker FROM units u;
CREATE VIEW v_items AS
  SELECT id, document_id, meeting_id, kind, identifier, title, ordinal, start_unit, end_unit, linked_item_id FROM items;
CREATE VIEW v_facts AS
  SELECT id, kind, document_id, meeting_id, city_id, body, meeting_date, item_id, unit_index, quote, data
  FROM facts WHERE verified;
CREATE VIEW v_motions AS
  SELECT id, document_id, meeting_id, city_id, body, meeting_date, unit_index,
         data->>'text' AS text, data->>'mover' AS mover, data->>'seconder' AS seconder,
         data->>'outcome' AS outcome, quote
  FROM facts WHERE verified AND kind = 'motion';
CREATE VIEW v_votes AS
  SELECT id, document_id, meeting_id, city_id, body, meeting_date, unit_index,
         data->>'subject_identifier' AS subject_identifier,
         (data->>'ayes')::int AS ayes, (data->>'noes')::int AS noes,
         (data->>'abstain')::int AS abstain, (data->>'absent')::int AS absent,
         data->>'outcome' AS outcome, quote
  FROM facts WHERE verified AND kind = 'vote';
CREATE VIEW v_vote_members AS
  SELECT f.id AS vote_id, f.city_id, f.meeting_date, m->>'name' AS member, m->>'value' AS value
  FROM facts f, jsonb_array_elements(f.data->'members') AS m
  WHERE f.verified AND f.kind = 'vote';
CREATE VIEW v_ordinances AS
  SELECT id, document_id, meeting_id, city_id, body, meeting_date, unit_index,
         data->>'identifier' AS identifier, data->>'legislation_type' AS legislation_type,
         data->>'title' AS title, data->>'status' AS status, quote
  FROM facts WHERE verified AND kind = 'ordinance';
CREATE VIEW v_amounts AS
  SELECT id, document_id, meeting_id, city_id, body, meeting_date, unit_index,
         (data->>'amount_usd')::numeric AS amount_usd, data->>'purpose' AS purpose,
         data->>'payee' AS payee, quote
  FROM facts WHERE verified AND kind = 'amount';
CREATE VIEW v_statements AS
  SELECT id, document_id, meeting_id, city_id, body, meeting_date, unit_index,
         data->>'speaker' AS speaker, data->>'role' AS role, data->>'summary' AS summary, quote
  FROM facts WHERE verified AND kind = 'statement';
