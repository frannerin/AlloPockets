# Database Concurrency Fixes and Improvements

## 1. Problem Statement
The Gradio application (`gradio.ipynb`) was vulnerable to experiencing stalling issues and failures when running under high concurrency (e.g., multiple concurrent user requests). The core investigation revealed several related issues:

*   **Database Locking:** SQLite default journaling mode caused "database is locked" errors when multiple threads attempted to write simultaneously.
*   **Thread Safety:** Each thread sharing the same global database connection object could lead to corruption or connection errors.

## 2. Fixes Implemented

The following changes were applied to ensuring robust database concurrency:

### A. Database Configuration (WAL Mode)
We modified the database initialization to enable **Write-Ahead Logging (WAL)** and increased the busy timeout.

```python
# db_models.py / gradio.ipynb database setup
db = APSWDatabase('allopockets.db', pragmas={
    'journal_mode': 'wal',       # Enables concurrent readers and writers
    'busy_timeout': 30000,       # Waits 30s for a lock instead of crashing immediately
    'synchronous': 'normal'      # optimizes performance with reasonable safety
})
```

**Reasoning:** 
*   **WAL Mode:** Allows readers to access the database without blocking writers, significantly improving concurrency.
*   **Busy Timeout:** SQLite normally throws an error immediately if the DB is locked. A 30-second timeout allows the application to queue writes and succeed once the lock is released, rather than crashing.

### B. Connection Management
We wrapped heavy calculation functions and database-accessing functions with the `@db.connection_context()` decorator.

```python
# gradio_app.py / gradio.ipynb
@db.connection_context()
def get_predictions(...):
    # ...
```

**Reasoning:** this ensures that each thread execution gets its own isolated thread-local connection or transaction context, preventing threads from stepping on each other's database operations.



## 3. Verification Results

A high-load stress test was performed (`stress_test.py`) to verify these fixes:

*   **Load:** 1000 requests, 100 concurrent threads.
*   **Duplicates:** 30% of requests were intentional duplicates.
*   **Result:** 
    *   **100% Success:** All unique jobs completed.
    *   **0 Stalls:** No requests hung or timed out.
    *   **Graceful Handling:** Duplicate jobs were identified and handled without crashing.
    *   **Zero Locking Errors:** WAL mode successfully handled the concurrent write load.
 and handles duplicate submissions gracefully as a fallback.