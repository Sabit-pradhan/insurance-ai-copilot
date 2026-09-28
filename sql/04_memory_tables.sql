-- =========================================================
-- INSURE AI
-- CONVERSATIONAL MEMORY TABLES
-- =========================================================


-- =========================================================
-- 1. APPLICATION SCHEMA
-- =========================================================

CREATE SCHEMA IF NOT EXISTS app;


-- =========================================================
-- 2. COPILOT SESSIONS
-- =========================================================

CREATE TABLE IF NOT EXISTS app.copilot_sessions
(
    session_id VARCHAR(100) PRIMARY KEY,

    last_route VARCHAR(20),

    context_topic TEXT,

    created_at TIMESTAMPTZ
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMPTZ
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);


-- =========================================================
-- 3. COPILOT MESSAGES
-- =========================================================

CREATE TABLE IF NOT EXISTS app.copilot_messages
(
    message_id BIGSERIAL PRIMARY KEY,

    session_id VARCHAR(100)
        NOT NULL,

    role VARCHAR(20)
        NOT NULL,

    content TEXT
        NOT NULL,

    route VARCHAR(20),

    created_at TIMESTAMPTZ
        NOT NULL
        DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_copilot_session
        FOREIGN KEY (session_id)
        REFERENCES app.copilot_sessions(session_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_copilot_role
        CHECK (
            role IN (
                'user',
                'assistant'
            )
        )
);


-- =========================================================
-- 4. INDEXES
-- =========================================================

CREATE INDEX IF NOT EXISTS
idx_copilot_messages_session
ON app.copilot_messages(session_id);


CREATE INDEX IF NOT EXISTS
idx_copilot_messages_created
ON app.copilot_messages(created_at);


CREATE INDEX IF NOT EXISTS
idx_copilot_messages_session_created
ON app.copilot_messages(
    session_id,
    created_at
);


-- =========================================================
-- 5. VALIDATION
-- =========================================================

SELECT
    table_schema,
    table_name
FROM information_schema.tables
WHERE table_schema = 'app'
ORDER BY table_name;