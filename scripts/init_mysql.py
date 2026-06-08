from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_insurance_data.csv"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def get_config():
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "insurance_fraud_db"),
    }


def main():
    cfg = get_config()

    server_url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}"
    )
    db_url = server_url + f"/{cfg['database']}"

    print("Connecting to MySQL server...")
    server_engine = create_engine(server_url, pool_pre_ping=True)

    # Create database
    with server_engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {cfg['database']}"))
        conn.commit()

    print(f"Database ready: {cfg['database']}")

    engine = create_engine(db_url, pool_pre_ping=True)

    df = pd.read_csv(DATA_PATH)
    df.to_sql("insurance_applications", engine, if_exists="replace", index=False)

    print(f"Loaded {len(df)} records into table: insurance_applications")
    print("MySQL initialization completed successfully.")


if __name__ == "__main__":
    main()