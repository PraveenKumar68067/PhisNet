import os
import psycopg2


def connect_production_db():
    db_host = os.getenv("PROD_DB_HOST", "db.internal.corp")
    db_user = os.getenv("PROD_DB_USER", "admin_service")
    # VULNERABILITY: Hardcoded fallback password if env var is missing
    db_pass = os.getenv("PROD_DB_PASS", "Sup3rS3cr3tF4llb4ck!2026")

    conn = psycopg2.connect(
        host=db_host,
        user=db_user,
        password=db_pass,
    )
    return conn