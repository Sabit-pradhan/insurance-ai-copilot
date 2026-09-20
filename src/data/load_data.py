# src/data/load_data.py

from pathlib import Path
import pandas as pd

# Import PostgreSQL connection engine
from src.data.db import engine


# --------------------------------------------------
# Project paths
# --------------------------------------------------

# Project root:
# insurance-ai-copilot/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw CSV folder
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# --------------------------------------------------
# CSV file → PostgreSQL table mapping
# --------------------------------------------------

TABLES = {
    "customers.csv": "customers",
    "policies.csv": "policies",
    "claims.csv": "claims",
    "premiums.csv": "premiums",
    "renewals.csv": "renewals",
    "underwriting.csv": "underwriting",
    "agents.csv": "agents"
}


# --------------------------------------------------
# Function to load one CSV
# --------------------------------------------------

def load_csv_to_postgres(file_name, table_name):

    # Full CSV path
    file_path = RAW_DATA_DIR / file_name

    print(f"\nLoading {file_name}...")

    # Read CSV into pandas
    df = pd.read_csv(file_path)

    # Convert column names to lowercase
    # CUSTOMER_ID → customer_id
    df.columns = df.columns.str.lower()

    # Convert date-like columns to datetime
    for column in df.columns:

        if (
            "date" in column
            or "dte" in column
            or "since" in column
        ):
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    # Load DataFrame into PostgreSQL
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=2000,
        method="multi"
    )

    print(
        f"{table_name} loaded successfully "
        f"({len(df):,} rows)"
    )


# --------------------------------------------------
# Load all tables
# --------------------------------------------------

def load_all_tables():

    for file_name, table_name in TABLES.items():

        load_csv_to_postgres(
            file_name=file_name,
            table_name=table_name
        )

    print("\nAll tables loaded successfully!")


# --------------------------------------------------
# Run script
# --------------------------------------------------

if __name__ == "__main__":

    load_all_tables()