import sqlite3

# Database file
DATABASE_URL = "airline_bot_enhanced.db"

# Connect to SQLite
conn = sqlite3.connect(DATABASE_URL)
cursor = conn.cursor()

# ===========================
# Create tables
# ===========================

# 1. Predictions table
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    predicted_request_type TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    user_ip TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
""")

# 2. Feedback table
cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    is_correct BOOLEAN NOT NULL,
    correct_request_type TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);
""")

# 3. Logs table
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_type TEXT NOT NULL,
    message TEXT NOT NULL,
    severity TEXT DEFAULT 'INFO',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    additional_data TEXT
);
""")

# Commit changes and close connection
conn.commit()
conn.close()

print(f"✅ Database '{DATABASE_URL}' with 3 tables created successfully!")
