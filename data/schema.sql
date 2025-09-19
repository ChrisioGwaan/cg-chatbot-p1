CREATE TABLE IF NOT EXISTS users (
  user_id        BIGSERIAL PRIMARY KEY,
  username       TEXT NOT NULL UNIQUE,
  email          TEXT NOT NULL UNIQUE,
  is_admin       BOOLEAN NOT NULL DEFAULT FALSE,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_proc WHERE proname = 'touch_row_updated_at'
  ) THEN
    CREATE OR REPLACE FUNCTION touch_row_updated_at()
    RETURNS TRIGGER AS $f$
    BEGIN
      NEW.updated_at := NOW();
      RETURN NEW;
    END
    $f$ LANGUAGE plpgsql;
  END IF;
END$$;

DROP TRIGGER IF EXISTS users_touch ON users;
CREATE TRIGGER users_touch
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION touch_row_updated_at();

CREATE INDEX IF NOT EXISTS idx_users_username_lower ON users (LOWER(username));
CREATE INDEX IF NOT EXISTS idx_users_email_lower    ON users (LOWER(email));

CREATE TABLE IF NOT EXISTS documents (
  document_id        BIGSERIAL PRIMARY KEY,
  owner_user_id      BIGINT,
  title              TEXT NOT NULL,
  content_url        TEXT,L
  content_text       TEXT,
  checksum_sha256    TEXT,
  mime_type          TEXT,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DROP TRIGGER IF EXISTS documents_touch ON documents;
CREATE TRIGGER documents_touch
BEFORE UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION touch_row_updated_at();

CREATE INDEX IF NOT EXISTS idx_documents_owner        ON documents (owner_user_id);
CREATE INDEX IF NOT EXISTS idx_documents_title_lower  ON documents (LOWER(title));
CREATE INDEX IF NOT EXISTS idx_documents_created_at   ON documents (created_at);

CREATE TABLE IF NOT EXISTS conversations (
  conversation_id    BIGSERIAL PRIMARY KEY,
  user_id            BIGINT,
  title              TEXT,
  started_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_activity_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_proc WHERE proname = 'touch_conversation_activity'
  ) THEN
    CREATE OR REPLACE FUNCTION touch_conversation_activity()
    RETURNS TRIGGER AS $f$
    BEGIN
      NEW.last_activity_at := NOW();
      RETURN NEW;
    END
    $f$ LANGUAGE plpgsql;
  END IF;
END$$;

DROP TRIGGER IF EXISTS conversations_touch ON conversations;
CREATE TRIGGER conversations_touch
BEFORE UPDATE ON conversations
FOR EACH ROW EXECUTE FUNCTION touch_conversation_activity();

CREATE INDEX IF NOT EXISTS idx_conversations_user          ON conversations (user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_last_activity ON conversations (last_activity_at DESC);

CREATE TABLE IF NOT EXISTS messages (
  message_id        BIGSERIAL PRIMARY KEY,
  conversation_id   BIGINT NOT NULL,
  role              TEXT NOT NULL,
  content           TEXT NOT NULL,
  token_count       INT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_conv_created ON messages (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_role         ON messages (role);

CREATE TABLE IF NOT EXISTS completions (
  completion_id     BIGSERIAL PRIMARY KEY,
  conversation_id   BIGINT,
  message_id        BIGINT,
  model_name        TEXT,
  prompt_tokens     INT,
  completion_tokens INT,
  total_tokens      INT,
  latency_ms        INT,
  status            TEXT,
  error_message     TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_completions_conv_created    ON completions (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_completions_message         ON completions (message_id);
CREATE INDEX IF NOT EXISTS idx_completions_model_created   ON completions (model_name, created_at);
