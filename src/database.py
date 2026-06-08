import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_insurance_data.csv"


def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_mysql_config():
    load_env()
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "insurance_fraud_db"),
    }


def get_mysql_engine(include_database=True):
    cfg = get_mysql_config()
    database_part = f"/{cfg['database']}" if include_database else ""
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}{database_part}"
    )
    return create_engine(url, pool_pre_ping=True)


def test_mysql_connection():
    try:
        engine = get_mysql_engine(include_database=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "MySQL connection successful."
    except SQLAlchemyError as exc:
        return False, f"MySQL connection failed: {exc}"
    except Exception as exc:
        return False, f"MySQL connection failed: {exc}"


def load_from_csv():
    return pd.read_csv(DATA_PATH)


def load_from_mysql(limit=None):
    engine = get_mysql_engine(include_database=True)
    query = "SELECT * FROM insurance_applications"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return pd.read_sql(query, engine)


def save_predictions_to_mysql(predictions_df, table_name="fraud_predictions"):
    engine = get_mysql_engine(include_database=True)
    predictions_df.to_sql(table_name, engine, if_exists="replace", index=False)