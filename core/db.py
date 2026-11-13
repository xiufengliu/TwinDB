"""Database connection and utilities for TwinDB"""
import os
import warnings
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

DB_CONFIG = {
    'host': os.getenv('TWINDB_HOST', 'localhost'),
    'port': int(os.getenv('TWINDB_PORT', 5432)),
    'database': os.getenv('TWINDB_DB', 'twindb'),
    'user': os.getenv('TWINDB_USER', 'twindb'),
    'password': os.getenv('TWINDB_PASSWORD', 'twindb123'),
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def execute_query(sql, params=None, as_df=True):
    with get_connection() as conn:
        if as_df:
            return pd.read_sql(sql, conn, params=params)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall() if cur.description else None

def execute_many(sql, data):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, data)
        conn.commit()

def execute_sql(sql, params=None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
