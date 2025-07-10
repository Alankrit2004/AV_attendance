# import pyodbc

# def connect_db():
#     conn = pyodbc.connect(
#         'DRIVER={ODBC Driver 17 for SQL Server};'
#         'SERVER=GlitchPC\\SQLEXPRESS;'
#         'DATABASE=ATTENDACE;'
#         'Trusted_Connection=yes;'
#     )
#     return conn

from sqlalchemy import create_engine


CONN_STR = (
    "mssql+pyodbc:///?odbc_connect="
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=GlitchPC\\SQLEXPRESS;"
    "DATABASE=ATTENDACE;"
    "Trusted_Connection=yes;"
)

engine = create_engine(CONN_STR, pool_size=5, max_overflow=5, fast_executemany=True)

def connect_db():
    return engine.raw_connection()