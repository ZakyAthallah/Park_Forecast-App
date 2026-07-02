"""
seed_db.py — One-time setup script for Prototype 4B database.

Clears the existing parkiran table and seeds it with the
initial 30-day sliding window data from a CSV file.

Usage:
    python seed_db.py
    python seed_db.py path/to/your_file.csv   (optional: specify CSV)
"""

import sys
import sqlite3
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DB_PATH  = DATA_DIR / "parkiran.db"

# Default: look for the seed CSV in the data/ folder
DEFAULT_CSV = DATA_DIR / "Data_1Jan-24Juni_2026.csv"
WINDOW_ROWS = 56  # 8 minggu = 56 hari

# ── Column names (must match parking_forecast_legacy.py) ───────
DATE_COL   = "Tanggal"
DAY_COL    = "Hari"
TARGET_COL = "Jumlah Total"
COL_MOBIL  = "Jumlah Roda 4"
COL_MOTOR  = "Jumlah Roda 2"
STATUS_COL = "Status Minggu"
COND_COL   = "Kondisi Parkiran"

KEEP_COLS = [DATE_COL, DAY_COL, TARGET_COL, COL_MOBIL, COL_MOTOR, STATUS_COL]


def seed(csv_path: Path):
    print(f"[INFO] Reading CSV  : {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Keep only columns the DB schema knows about
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[c for c in KEEP_COLS if c in df.columns]]

    # Take only the last WINDOW_ROWS rows (sliding window seed)
    if len(df) > WINDOW_ROWS:
        df = df.tail(WINDOW_ROWS).reset_index(drop=True)

    n_rows = len(df)
    print(f"   {n_rows} rows found  ({df[DATE_COL].iloc[0]} -> {df[DATE_COL].iloc[-1]})")

    DATA_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    # Drop and recreate for a clean slate
    conn.execute("DROP TABLE IF EXISTS parkiran")
    conn.execute(f"""
        CREATE TABLE parkiran (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            "{DATE_COL}"       TEXT    NOT NULL UNIQUE,
            "{DAY_COL}"        TEXT,
            "{TARGET_COL}"     INTEGER NOT NULL,
            "{COL_MOBIL}"      INTEGER,
            "{COL_MOTOR}"      INTEGER,
            "{STATUS_COL}"     TEXT,
            "{COND_COL}"       TEXT
        )
    """)

    df.to_sql("parkiran", conn, if_exists="append", index=False)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM parkiran").fetchone()[0]
    oldest = conn.execute(f'SELECT MIN("{DATE_COL}") FROM parkiran').fetchone()[0]
    newest = conn.execute(f'SELECT MAX("{DATE_COL}") FROM parkiran').fetchone()[0]
    conn.close()

    print(f"\n[OK] Database ready: {DB_PATH}")
    print(f"   Rows   : {count}")
    print(f"   Range  : {oldest}  ->  {newest}")
    print(f"\nNext step: open Prototype 4B and click Run Forecasting.")


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print(f"\nCopy your 30-day CSV into the data/ folder as:")
        print(f"   {DEFAULT_CSV}")
        print(f"Or pass the path as an argument:")
        print(f"   python seed_db.py path/to/file.csv")
        sys.exit(1)

    seed(csv_path)
