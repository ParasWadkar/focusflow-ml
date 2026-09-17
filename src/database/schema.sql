-- FocusFlow ML relational schema (SQLite).
-- Dates are stored as ISO-8601 TEXT (YYYY-MM-DD), times as zero-padded HH:MM
-- so that lexicographic comparison matches chronological order.

CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS habits (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    description TEXT,
    schedule    TEXT    NOT NULL DEFAULT 'daily'
                CHECK (schedule IN ('daily', 'weekdays', 'weekends')),
    start_date  TEXT    NOT NULL,
    archived    INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0, 1)),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, name)
);

CREATE TABLE IF NOT EXISTS habit_records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    habit_id    INTEGER NOT NULL REFERENCES habits(id) ON DELETE CASCADE,
    date        TEXT    NOT NULL,
    completed   INTEGER NOT NULL CHECK (completed IN (0, 1)),
    note        TEXT,
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (habit_id, date)
);

CREATE TABLE IF NOT EXISTS time_blocks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date        TEXT    NOT NULL,
    start_time  TEXT    NOT NULL,
    end_time    TEXT    NOT NULL,
    task_name   TEXT    NOT NULL,
    category    TEXT    NOT NULL
                CHECK (category IN ('study', 'deep_work', 'work', 'exercise',
                                    'admin', 'personal', 'other')),
    priority    INTEGER NOT NULL DEFAULT 2 CHECK (priority BETWEEN 1 AND 3),
    completed   INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1)),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    CHECK (end_time > start_time)
);

CREATE TABLE IF NOT EXISTS daily_records (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date                TEXT    NOT NULL,
    sleep_hours         REAL    CHECK (sleep_hours IS NULL OR sleep_hours BETWEEN 0 AND 24),
    energy_level        INTEGER CHECK (energy_level IS NULL OR energy_level BETWEEN 1 AND 10),
    mood_score          INTEGER CHECK (mood_score IS NULL OR mood_score BETWEEN 1 AND 10),
    exercise_minutes    REAL    CHECK (exercise_minutes IS NULL OR exercise_minutes BETWEEN 0 AND 1440),
    interruptions       INTEGER CHECK (interruptions IS NULL OR interruptions >= 0),
    journal             TEXT,
    productivity_score  REAL    CHECK (productivity_score IS NULL OR productivity_score BETWEEN 0 AND 100),
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, date)
);

CREATE TABLE IF NOT EXISTS day_photos (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    daily_record_id  INTEGER NOT NULL REFERENCES daily_records(id) ON DELETE CASCADE,
    file_path        TEXT    NOT NULL,
    caption          TEXT,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_habit_records_date ON habit_records (date);
CREATE INDEX IF NOT EXISTS idx_time_blocks_user_date ON time_blocks (user_id, date);
CREATE INDEX IF NOT EXISTS idx_daily_records_user_date ON daily_records (user_id, date);
