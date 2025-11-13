# consultas_postgres.py

import psycopg2
import pandas as pd

# --- Configuración ---
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "nombre_base"
DB_USER = "postgres"
DB_PASS = "tu_contraseña"

# --- Conexión ---
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)

# --- Consulta de ejemplo ---
query = "SELECT * FROM mi_tabla LIMIT 10;"  # ajusta el nombre de la tabla

df = pd.read_sql(query, conn)
print(df)

# --- Otra consulta: estadísticas ---
query_stats = """
SELECT columna1, AVG(columna2) as promedio
FROM mi_tabla
GROUP BY columna1
LIMIT 10;
"""
df_stats = pd.read_sql(query_stats, conn)
print(df_stats)

# --- Cerrar conexión ---
conn.close()
