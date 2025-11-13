# cargar_postgres.py

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# --- Configuración ---
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "nombre_base"
DB_USER = "postgres"        # tu usuario
DB_PASS = "tu_contraseña"   # tu contraseña

CSV_PATH = "data/df_imputado.csv"
TABLE_NAME = "mi_tabla"     # nombre de la tabla que se creará

df = pd.read_csv(CSV_PATH)

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
print(f"✅ Datos cargados en la tabla '{TABLE_NAME}' de la base '{DB_NAME}'")
