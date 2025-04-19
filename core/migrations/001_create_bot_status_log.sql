-- Migration: Create bot_status_log table for operational status tracking
CREATE TABLE IF NOT EXISTS bot_status_log (
    id SERIAL PRIMARY KEY,
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(64) NOT NULL,
    stage VARCHAR(128),
    details JSONB,
    error_info TEXT
);
