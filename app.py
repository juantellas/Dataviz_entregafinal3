import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io, base64
import joblib 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import plotly.graph_objects as go
import os
import statsmodels
# -------------------------------------
# Configuración general
# -------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE],suppress_callback_exceptions=True)
app.title = "Dashboard Dataviz"
server = app.server

# -------------------------------------
# Cargar datos
# -------------------------------------
df_imputado = pd.read_csv("data/df_imputado.csv")

# --- Cargar modelos ---
# --- Cargar packs ---
modelos_pack = {
    "XGBoost": joblib.load("models/xgboost_pack.pkl"),
    "Random Forest": joblib.load("models/random_forest_pack.pkl"),
    "Gradient Boosting": joblib.load("models/gradient_boosting_pack.pkl")
}
# --- Evaluar métricas ---
# -------------------------------------
# Función auxiliar para convertir figuras Matplotlib a imágenes
# -------------------------------------
def fig_to_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return "data:image/png;base64,{}".format(encoded)

# -------------------------------------
# Botonera horizontal (1–9)
# -------------------------------------
nav_buttons = dbc.ButtonGroup(
    [
        dbc.Button("0. Cover Page", id="btn-0", outline=True, color="primary"),  # Nuevo botón
        dbc.Button("1. Introducción", id="btn-1", outline=True, color="primary"),
        dbc.Button("2. Contexto", id="btn-2", outline=True, color="primary"),
        dbc.Button("3. Planteamiento del Problema", id="btn-3", outline=True, color="primary"),
        dbc.Button("4. Objetivos y Justificación", id="btn-4", outline=True, color="primary"),
        dbc.Button("5. Marco Teorico", id="btn-5", outline=True, color="primary"),
        dbc.Button("6. Metodologia", id="btn-6", outline=True, color="primary"),
        dbc.Button("7. Resultados/Analisis Final", id="btn-7", outline=True, color="primary"),
        dbc.Button("8. Conclusiones", id="btn-8", outline=True, color="primary")
    ],
    className="d-flex justify-content-around mb-4 flex-wrap gap-2",
)

# -------------------------------------
# Layout principal
# -------------------------------------
app.layout = dbc.Container(
    [
        html.H1("Análisis interactivo de la prevalencia de infección latente por tuberculosis (LTBI) alrededor del mundo (2015 - 2023)", className="text-center mt-3 mb-4"),
        html.Hr(),
        nav_buttons,
        html.Div(id="content-area"),
    ],
    fluid=True,
)

# -------------------------------------
# Callbacks principales
# -------------------------------------
@app.callback(
    Output("content-area", "children"),
    [Input(f"btn-{i}", "n_clicks") for i in range(0, 9)]
)
def mostrar_contenido(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.P("Selecciona una sección del informe para comenzar.", className="text-muted text-center")

    boton_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if boton_id == "btn-0":
        return dbc.Card([
            dbc.CardBody([
               html.Div(
    [
        html.H1(
            "Análisis Global de LTBI en Contactos Domiciliarios",
            className="text-center mb-4",
        ),

        html.H3(
            "Estimaciones de Tuberculosis Latente (2000–2024)",
            className="text-center mb-3",
        ),

        # Imagen principal (opcional)
  # Imagen principal con menor margen
        html.Img(
            src="assets/logo.png",
            style={
                "width": "20%",           # (opcional) también puedes ajustar el tamaño
                "display": "block",
                "margin": "10px auto"     # margen reducido
            },
        ),


        html.H4(
            "Colaboración entre:",
            className="text-center mt-4"
        ),

        html.P(
            "Departamento de Ciencias Básicas – Universidad del Norte",
            className="text-center",
        ),

        html.P(
            "Este dashboard presenta un análisis exploratorio de las estimaciones globales "
            "de infección latente por tuberculosis (LTBI) en contactos domiciliarios, "
            "utilizando datos publicados por la Organización Mundial de la Salud (OMS). "
            "El propósito es visualizar tendencias, comparar regiones y facilitar la "
            "comprensión del comportamiento epidemiológico de la LTBI a lo largo del tiempo.",
            className="text-center mt-4",
        ),

        html.P(
            "Ubicación: Barranquilla, Atlántico – Colombia",
            className="text-center",
        ),

        html.P(
            "Periodo analizado: 2000–2024",
            className="text-center",
        ),

        html.P(
            "Fuente principal: OMS – Global Tuberculosis Programme",
            className="text-center mb-4",
        ),

        html.H4("Créditos", className="text-center mt-4"),

        html.Ul(
            [
                html.Li("Miguel Ángel Pérez", className="text-center"),
                html.Li("Camilo Vargas Escorcia", className="text-center"),
                html.Li("Juan Camilo Aguirre", className="text-center"),
            ],
            style={"list-style-type": "none", "padding": 0},
        ),
    ],
    className="container mt-4",
),

            ])
        ])

    
    elif boton_id == "btn-1":
        return html.Div([
    
            # Tarjeta del título
            dbc.Card(
                dbc.CardBody([
                    html.H4("Introducción", className="card-title")
                ]),
                style={
                    "backgroundColor": "#f3f3f3",
                    "borderRadius": "14px",
                    "marginBottom": "20px",
                    "padding": "10px",
                    "boxShadow": "0 3px 8px rgba(0,0,0,0.12)"
                }
            ),
    
            # Tarjeta del texto completo
            dbc.Card(
                dbc.CardBody([
                    html.P("""
                    La tuberculosis (TB) sigue siendo una de las enfermedades infecciosas más relevantes
                    a nivel global, con un impacto sostenido en la salud pública a pesar de los avances
                    en diagnóstico y tratamiento. Una proporción significativa de la población mundial
                    vive con una infección latente por Mycobacterium tuberculosis (LTBI), lo que implica
                    la presencia de la bacteria sin manifestación activa de la enfermedad, pero con riesgo
                    potencial de desarrollarla en el futuro. Comprender la magnitud y evolución de esta
                    infección es fundamental para orientar las políticas de prevención y control que
                    promueve la Organización Mundial de la Salud (OMS).
                    """),
    
                    html.P("""
                    El presente análisis se basa en las estimaciones publicadas por la OMS dentro del
                    Global Tuberculosis Report 2024, específicamente en el apartado sobre infección
                    latente por tuberculosis (LTBI) en contactos domiciliarios, que representa a uno de
                    los grupos poblacionales con mayor vulnerabilidad frente a la transmisión de la
                    enfermedad. Este tipo de información permite observar la situación global de la
                    infección y los avances alcanzados en la detección y contención de la tuberculosis
                    durante las últimas décadas.
                    """),
    
                    html.P("Repositorio: https://github.com/juantellas")
                ]),
                style={
                    "backgroundColor": "#fafafa",
                    "borderRadius": "14px",
                    "padding": "20px",
                    "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                }
            )
    
        ])


    elif boton_id == "btn-2":
        return html.Div([
    
            # Tarjeta del título
            dbc.Card(
                dbc.CardBody([
                    html.H4("Contexto Global", className="card-title")
                ]),
                style={
                    "backgroundColor": "#f3f3f3",
                    "borderRadius": "14px",
                    "marginBottom": "20px",
                    "padding": "10px",
                    "boxShadow": "0 3px 8px rgba(0,0,0,0.12)"
                }
            ),
    
            # Tarjeta del texto + viñetas
            dbc.Card(
                dbc.CardBody([
                    html.P("""
                    Las regiones operativas de la OMS reportan anualmente estimaciones sobre la infección
                    latente por tuberculosis (LTBI) en contactos domiciliarios. Estos valores permiten observar
                    diferencias entre territorios, así como cambios en la prevalencia a lo largo del tiempo.
                    El conjunto de datos utilizado reúne estas estimaciones entre 2000 y 2024, incluyendo la
                    prevalencia central y sus intervalos de incertidumbre, el porcentaje de niños evaluados y el
                    número de personas elegibles para tratamiento preventivo. Esta información resume los
                    principales indicadores usados en la vigilancia global de la LTBI.
                    """),
    
                    html.Ul([
                        html.Li("Source: OMS – Global Tuberculosis Programme"),
                        html.Li("Link: https://www.who.int/teams/global-programme-on-tuberculosis-and-lung-health/data"),
                        html.Li("Period: 2000–2024"),
                        html.Li("Variables: Prevalencia, intervalos inferior/superior, % de niños, elegibles para tratamiento")
                    ])
                ]),
                style={
                    "backgroundColor": "#fafafa",
                    "borderRadius": "14px",
                    "padding": "20px",
                    "marginBottom": "20px",
                    "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                }
            ),
    
            # Tarjeta de la imagen
            dbc.Card(
                dbc.CardBody([
                    html.Img(
                        src="assets/map.jpeg",
                        style={
                            "width": "80%",
                            "display": "block",
                            "margin": "0 auto"
                        },
                    )
                ]),
                style={
                    "backgroundColor": "#fafafa",
                    "borderRadius": "14px",
                    "padding": "20px",
                    "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                }
            )
    
        ])


    elif boton_id == "btn-3":
        return html.Div([
    
            # Tarjeta del título
            dbc.Card(
                dbc.CardBody([
                    html.H4("Planteamiento del Problema", className="card-title")
                ]),
                style={
                    "backgroundColor": "#f3f3f3",
                    "borderRadius": "14px",
                    "marginBottom": "20px",
                    "padding": "10px",
                    "boxShadow": "0 3px 8px rgba(0,0,0,0.12)"
                }
            ),
    
            # Tarjeta del texto
            dbc.Card(
                dbc.CardBody([
                    html.P("""
                    La infección latente por tuberculosis (LTBI) representa una condición en la que la bacteria
                    Mycobacterium tuberculosis está presente sin causar enfermedad activa.
                    La probabilidad de progresar a TB activa depende de factores inmunológicos y ambientales.
    
                    Con esto en mente, surge la pregunta central:
                    ¿Cómo ha variado la prevalencia estimada de infección latente por tuberculosis (LTBI) en
                    contactos domiciliarios a nivel mundial entre los años 2000 y 2024, y qué regiones presentan los
                    mayores cambios en sus estimaciones durante este período?
                    """)
                ]),
                style={
                    "backgroundColor": "#fafafa",
                    "borderRadius": "14px",
                    "padding": "20px",
                    "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                }
            )
    
        ])



    elif boton_id == "btn-4":
        return html.Div([
    
            # Tarjeta del título principal
            dbc.Card(
                dbc.CardBody([
                    html.H2("Objetivos y Justificación", className="card-title")
                ]),
                style={
                    "backgroundColor": "#f3f3f3",
                    "borderRadius": "14px",
                    "marginBottom": "20px",
                    "padding": "10px",
                    "boxShadow": "0 3px 8px rgba(0,0,0,0.12)"
                }
            ),
    
            # Fila con Objetivo General (izquierda) y Objetivos Específicos (derecha)
            dbc.Row([
    
                # Columna izquierda - Objetivo General
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H3("Objetivo General"),
                            html.P("""
                            Analizar la evolución mundial de la prevalencia de infección latente por 
                            tuberculosis (LTBI) en contactos domiciliarios entre 2000 y 2024, utilizando 
                            las estimaciones oficiales de la OMS para identificar cambios temporales y 
                            diferencias regionales relevantes para la salud pública global.
                            """)
                        ]),
                        style={
                            "backgroundColor": "#fafafa",
                            "borderRadius": "14px",
                            "padding": "20px",
                            "marginBottom": "20px",
                            "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                        }
                    ),
                    width=6
                ),
    
                # Columna derecha - Objetivos Específicos
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H3("Objetivos Específicos"),
                            html.Ul([
                                html.Li("Realizar un análisis descriptivo y exploratorio (EDA) de las tendencias globales y regionales de la prevalencia estimada de LTBI."),
                                html.Li("Comparar la evolución por regiones de la OMS para identificar aumentos, disminuciones o estancamientos en la carga latente."),
                                html.Li("Evaluar los patrones temporales y su relación con intervenciones de salud pública o cambios epidemiológicos."),
                                html.Li("Proveer una base analítica que apoye nuevas políticas de control de tuberculosis en poblaciones vulnerables."),
                            ])
                        ]),
                        style={
                            "backgroundColor": "#fafafa",
                            "borderRadius": "14px",
                            "padding": "20px",
                            "marginBottom": "20px",
                            "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                        }
                    ),
                    width=6
                ),
    
            ]),
    
            # Tarjeta inferior: Justificación (ancho completo)
            dbc.Card(
                dbc.CardBody([
                    html.H3("Justificación"),
                    html.P("""
                    La infección latente por tuberculosis representa uno de los mayores desafíos 
                    para la eliminación global de la TB, ya que constituye el reservorio desde el 
                    cual emergen nuevos casos activos. Los contactos domiciliarios son un grupo 
                    prioritario debido a su exposición directa y sostenida a personas enfermas.
                    """),
    
                    html.P("Analizar su evolución es esencial para evaluar:"),
    
                    html.Ul([
                        html.Li("La efectividad de las estrategias globales implementadas por la OMS y países miembros."),
                        html.Li("Los avances en vigilancia epidemiológica y acceso a herramientas diagnósticas."),
                        html.Li("La persistencia de desigualdades regionales que afectan la eliminación de la TB."),
                        html.Li("Los cambios epidemiológicos asociados a factores sociales, económicos y sanitarios."),
                    ]),
    
                    html.P("""
                    Este análisis contribuye a fortalecer la toma de decisiones, optimizar recursos 
                    y priorizar intervenciones en poblaciones vulnerables.
                    """)
                ]),
                style={
                    "backgroundColor": "#fafafa",
                    "borderRadius": "14px",
                    "padding": "20px",
                    "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                }
            )
    
        ])



    elif boton_id == "btn-5":
        return html.Div([
    
            # Tarjeta del título principal
            dbc.Card(
                dbc.CardBody([
                    html.H2("Marco Teórico", className="card-title")
                ]),
                style={
                    "backgroundColor": "#f3f3f3",
                    "borderRadius": "14px",
                    "marginBottom": "20px",
                    "padding": "10px",
                    "boxShadow": "0 3px 8px rgba(0,0,0,0.12)"
                }
            ),
    
            # ============================
            # Fila 1
            # ============================
            dbc.Row([
    
                # Tarjeta 1
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("1. Tuberculosis como problema de salud pública"),
                            html.P(
                                "La tuberculosis (TB) es una enfermedad infecciosa causada por Mycobacterium "
                                "tuberculosis y continúa siendo uno de los principales desafíos globales en salud. "
                                "Su persistencia se asocia a pobreza, hacinamiento, debilidad institucional y "
                                "desigualdades estructurales."
                            )
                        ]),
                        style={
                            "backgroundColor": "#fafafa",
                            "borderRadius": "14px",
                            "padding": "20px",
                            "marginBottom": "20px",
                            "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                        }
                    ),
                    width=6
                ),
    
                # Tarjeta 2
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("2. Infección latente por tuberculosis (LTBI)"),
                            html.P(
                                "La LTBI ocurre cuando una persona se infecta con M. tuberculosis pero la bacteria "
                                "permanece inactiva, sin síntomas ni capacidad de transmisión. Sin embargo, existe "
                                "riesgo de progresión a enfermedad activa, especialmente en poblaciones vulnerables."
                            )
                        ]),
                        style={
                            "backgroundColor": "#fafafa",
                            "borderRadius": "14px",
                            "padding": "20px",
                            "marginBottom": "20px",
                            "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                        }
                    ),
                    width=6
                )
    
            ]),
    
            # ============================
            # Fila 2
            # ============================
            dbc.Row([
    
                # Tarjeta 3
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("3. Contactos domiciliarios como población prioritaria"),
                            html.P(
                                "Los contactos domiciliarios presentan un riesgo significativamente mayor de "
                                "adquirir LTBI debido a la exposición prolongada a casos activos. Son un grupo "
                                "prioritario en rastreo, diagnóstico y tratamiento preventivo según la OMS."
                            )
                        ]),
                        style={
                            "backgroundColor": "#fafafa",
                            "borderRadius": "14px",
                            "padding": "20px",
                            "marginBottom": "20px",
                            "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                        }
                    ),
                    width=6
                ),
    
                # Tarjeta 4
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("4. Clasificación regional de la OMS y vigilancia epidemiológica"),
                            html.P(
                                "La OMS organiza la vigilancia por regiones geográficas (AFR, AMR, EMR, EUR, SEAR, "
                                "WPR), lo que permite comparar tendencias, identificar desigualdades y evaluar "
                                "la efectividad de intervenciones sanitarias."
                            )
                        ]),
                        style={
                            "backgroundColor": "#fafafa",
                            "borderRadius": "14px",
                            "padding": "20px",
                            "marginBottom": "20px",
                            "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                        }
                    ),
                    width=6
                )
    
            ]),
    
            # ============================
            # Fila 3
            # ============================
            dbc.Row([
    
                # Tarjeta 5
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("5. Indicadores epidemiológicos relevantes para LTBI"),
                            html.P(
                                "Un indicador clave es el porcentaje de contactos domiciliarios elegibles o con "
                                "tratamiento previo para profilaxis. Sus intervalos de incertidumbre reflejan "
                                "dinámica epidemiológica y acceso a intervenciones."
                            )
                        ]),
                        style={
                            "backgroundColor": "#fafafa",
                            "borderRadius": "14px",
                            "padding": "20px",
                            "marginBottom": "20px",
                            "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                        }
                    ),
                    width=6
                ),
    
                # Tarjeta 6
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("6. Relevancia del análisis temporal (2000–2024)"),
                            html.P(
                                "Estudiar la evolución entre 2000 y 2024 permite identificar tendencias globales "
                                "y regionales, avances, rezagos y su relación con determinantes epidemiológicos "
                                "y sociales."
                            )
                        ]),
                        style={
                            "backgroundColor": "#fafafa",
                            "borderRadius": "14px",
                            "padding": "20px",
                            "marginBottom": "20px",
                            "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"
                        }
                    ),
                    width=6
                )
    
            ])
    
        ])

    elif boton_id == "btn-6":
        return html.Div([
            
            dcc.Tabs(
                id="tabs-metodologia",
                colors={
                    "border": "#ccc",
                    "primary": "#1B4965",  # azul OMS / salud pública
                    "background": "#f3f3f3"
                },
                children=[
    
                    # ==========================================================
                    #  1. METODOLOGÍA INICIAL
                    # ==========================================================
                    dcc.Tab(label="Metodología Inicial", children=[
                        html.Br(),
    
                        # ---- Tarjeta de texto ----
                        dbc.Card(
                            dbc.CardBody([
    
                                html.H3("Metodología Inicial", className="text-center"),
                                html.Hr(),
    
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H4("Tratamiento de valores faltantes", style={"color": "#1B4965"}),
                                        html.P("""
                                        Se identificaron valores faltantes mediante mapas de missingness y cálculo
                                        de porcentajes de ausencia. Según el mecanismo (MCAR, MAR, MNAR), se aplicó
                                        imputación por mediana global, mediana por subgrupos o retención de NA.
                                        Adicionalmente se comparó una imputación mediante MICE-PMM.
                                        """, style={"textAlign": "justify"})
                                    ]),
                                    style={"backgroundColor": "#fafafa", "borderRadius": "14px",
                                           "padding": "15px", "marginBottom": "20px",
                                           "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"}
                                ),
    
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H4("Detección y corrección de outliers", style={"color": "#1B4965"}),
                                        html.P("""
                                        Se identificaron valores atípicos con el rango intercuartílico (IQR) y 
                                        el modified Z-score. Los outliers fueron imputados por la mediana hasta
                                        lograr su completa estabilización.
                                        """, style={"textAlign": "justify"})
                                    ]),
                                    style={"backgroundColor": "#fafafa", "borderRadius": "14px",
                                           "padding": "15px", "marginBottom": "20px",
                                           "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"}
                                ),
    
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H4("Análisis exploratorio y modelado", style={"color": "#1B4965"}),
                                        html.P("""
                                        Se realizó un análisis exploratorio de la prevalencia de LTBI a través del
                                        tiempo y por regiones OMS. Posteriormente se implementaron modelos como
                                        XGBoost, Random Forest y Gradient Boosting para evaluar el desempeño 
                                        predictivo.
                                        """, style={"textAlign": "justify"})
                                    ]),
                                    style={"backgroundColor": "#fafafa", "borderRadius": "14px",
                                           "padding": "15px",
                                           "boxShadow": "0 3px 8px rgba(0,0,0,0.10)"}
                                ),
    
                            ]),
                            style={"backgroundColor": "#ffffff", "borderRadius": "14px",
                                   "padding": "20px", "marginBottom": "30px",
                                   "boxShadow": "0 4px 10px rgba(0,0,0,0.12)"}
                        ),
    
                    ]),
    
                    # ==========================================================
                    #  2. IMPUTACIÓN DE DATOS
                    # ==========================================================
                    dcc.Tab(label="Imputación de Datos", children=[
                        html.Br(),
    
                        dbc.Card(
                            dbc.CardBody([
                                html.H3("Imputación de Datos", className="text-center"),
                                html.Hr(),
    
                                html.P("""
                                Se evaluó el patrón de valores faltantes y se aplicaron distintas técnicas
                                de imputación: mediana global, mediana estratificada y MICE-PMM. El objetivo
                                fue preservar la estructura estadística del dataset y minimizar sesgos.
                                """, style={"textAlign": "justify"}),
    
                                html.Br(),
    
                                # Galería en tarjetas
                                dbc.Row([
    
                                    # IMAGEN 1
                                    dbc.Col(
                                        dbc.Card([
                                            html.Img(src="assets/met1.png",
                                                     style={"width": "100%", "borderRadius": "10px"}),
                                            dbc.CardBody(html.P("Mapa de valores faltantes"))
                                        ], style={"borderRadius": "14px",
                                                  "boxShadow": "0 3px 6px rgba(0,0,0,0.12)",
                                                  "marginBottom": "20px"}),
                                        md=4
                                    ),
    
                                    # IMAGEN 2
                                    dbc.Col(
                                        dbc.Card([
                                            html.Img(src="assets/met2.png",
                                                     style={"width": "100%", "borderRadius": "10px"}),
                                            dbc.CardBody(html.P("Distribución antes de imputar"))
                                        ], style={"borderRadius": "14px",
                                                  "boxShadow": "0 3px 6px rgba(0,0,0,0.12)",
                                                  "marginBottom": "20px"}),
                                        md=4
                                    ),
    
                                    # IMAGEN 3
                                    dbc.Col(
                                        dbc.Card([
                                            html.Img(src="assets/met3.png",
                                                     style={"width": "100%", "borderRadius": "10px"}),
                                            dbc.CardBody(html.P("Comparación imputaciones"))
                                        ], style={"borderRadius": "14px",
                                                  "boxShadow": "0 3px 6px rgba(0,0,0,0.12)",
                                                  "marginBottom": "20px"}),
                                        md=4
                                    ),
    
                                ]),
    
                                dbc.Row([
                                    # IMAGEN 4 — fila aparte
                                    dbc.Col(
                                        dbc.Card([
                                            html.Img(src="assets/met4.png",
                                                     style={"width": "100%", "borderRadius": "10px"}),
                                            dbc.CardBody(html.P("Resultado final de MICE-PMM"))
                                        ], style={"borderRadius": "14px",
                                                  "boxShadow": "0 3px 6px rgba(0,0,0,0.12)",
                                                  "marginBottom": "20px"}),
                                        md=4
                                    ),
                                ]),
    
                            ]),
                            style={"backgroundColor": "#ffffff", "borderRadius": "14px",
                                   "padding": "20px", "marginBottom": "30px",
                                   "boxShadow": "0 4px 10px rgba(0,0,0,0.12)"}
                        ),
    
                    ]),
    
                    # ==========================================================
                    #  3. VALIDACIÓN ESTADÍSTICA
                    # ==========================================================
                    dcc.Tab(label="Validación Estadística", children=[
                        html.Br(),
    
                        dbc.Card(
                            dbc.CardBody([
    
                                html.H3("Validación Estadística", className="text-center"),
                                html.Hr(),
    
                                html.P("""
                                Se evaluó la integridad del dataset imputado verificando preservación de
                                varianzas, sesgos potenciales y estabilidad estadística posterior al proceso
                                de imputación.
                                """, style={"textAlign": "justify"}),
    
                                html.Br(),
    
                                dbc.Row([
    
                                    # Validación 1
                                    dbc.Col(
                                        dbc.Card([
                                            html.Img(src="assets/met5.png",
                                                     style={"width": "100%", "borderRadius": "10px"}),
                                            dbc.CardBody(html.P("Comparación densidades"))
                                        ], style={"borderRadius": "14px",
                                                  "boxShadow": "0 3px 6px rgba(0,0,0,0.12)",
                                                  "marginBottom": "20px"}),
                                        md=4
                                    ),
    
                                    # Validación 2
                                    dbc.Col(
                                        dbc.Card([
                                            html.Img(src="assets/met6.png",
                                                     style={"width": "100%", "borderRadius": "10px"}),
                                            dbc.CardBody(html.P("Validación por grupos"))
                                        ], style={"borderRadius": "14px",
                                                  "boxShadow": "0 3px 6px rgba(0,0,0,0.12)",
                                                  "marginBottom": "20px"}),
                                        md=4
                                    ),
    
                                    # Validación 3
                                    dbc.Col(
                                        dbc.Card([
                                            html.Img(src="assets/met7.png",
                                                     style={"width": "100%", "borderRadius": "10px"}),
                                            dbc.CardBody(html.P("Análisis de varianza residual"))
                                        ], style={"borderRadius": "14px",
                                                  "boxShadow": "0 3px 6px rgba(0,0,0,0.12)",
                                                  "marginBottom": "20px"}),
                                        md=4
                                    ),
    
                                ]),
    
                            ]),
                            style={"backgroundColor": "#ffffff", "borderRadius": "14px",
                                   "padding": "20px", "marginBottom": "30px",
                                   "boxShadow": "0 4px 10px rgba(0,0,0,0.12)"}
                        ),
    
                    ]),
    
                    # ==========================================================
                    #  4. IMPLEMENTACIÓN
                    # ==========================================================
                    dcc.Tab(label="Implementación", children=[
                        html.Br(),
    
                        dbc.Card(
                            dbc.CardBody([
    
                                html.H3("Implementación del Análisis y Modelado", className="text-center"),
                                html.Hr(),
    
                                html.P("""
                                Se construyeron los pipelines de preprocesamiento y posteriormente se
                                entrenaron modelos como XGBoost, Random Forest y Gradient Boosting para
                                comparar su rendimiento en la clasificación.
                                """, style={"textAlign": "justify"}),
    
                                html.Br(),
    
                                dbc.Row([
    
                                    dbc.Col(
                                        dbc.Card([
                                            html.Img(src="assets/met8.png",
                                                     style={"width": "100%", "borderRadius": "10px"}),
                                            dbc.CardBody(html.P("Pipeline general"))
                                        ], style={"borderRadius": "14px",
                                                  "boxShadow": "0 3px 6px rgba(0,0,0,0.12)",
                                                  "marginBottom": "20px"}),
                                        md=4
                                    ),
    
                                    dbc.Col(
                                        dbc.Card([
                                            html.Img(src="assets/met9.png",
                                                     style={"width": "100%", "borderRadius": "10px"}),
                                            dbc.CardBody(html.P("Entrenamiento modelos"))
                                        ], style={"borderRadius": "14px",
                                                  "boxShadow": "0 3px 6px rgba(0,0,0,0.12)",
                                                  "marginBottom": "20px"}),
                                        md=4
                                    ),
    
                                    dbc.Col(
                                        dbc.Card([
                                            html.Img(src="assets/met10.png",
                                                     style={"width": "100%", "borderRadius": "10px"}),
                                            dbc.CardBody(html.P("Comparación final de métricas"))
                                        ], style={"borderRadius": "14px",
                                                  "boxShadow": "0 3px 6px rgba(0,0,0,0.12)",
                                                  "marginBottom": "20px"}),
                                        md=4
                                    ),
    
                                ]),
    
                            ]),
                            style={"backgroundColor": "#ffffff", "borderRadius": "14px",
                                   "padding": "20px", "marginBottom": "30px",
                                   "boxShadow": "0 4px 10px rgba(0,0,0,0.12)"}
                        ),
    
                    ]),
    
                ]
            )
        ])




    elif boton_id == "btn-7":
        tabs = dcc.Tabs([
            # === Pestaña: EDA ===
            dcc.Tab(label="EDA", children=[
                html.Br(),
                html.H3("Evolución global de LTBI"),

                html.Label("Selecciona el rango de años:", style={"fontWeight": "bold"}),
                dcc.RangeSlider(
                    id="slider-rango-anios",
                    min=df_imputado["anio"].min(),
                    max=df_imputado["anio"].max(),
                    step=1,
                    value=[df_imputado["anio"].min(), df_imputado["anio"].max()],
                    marks={int(y): str(y) for y in df_imputado["anio"].unique()},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Br(),
                dcc.Graph(id="grafico-evolucion-global"),
                html.Hr(),

                dbc.Row([
                    dbc.Col(dcc.Graph(id="heatmap-region-anio"), md=8),
                    dbc.Col(html.Div(id="tarjetas-resumen"), md=4)
                ]),
                html.Hr(),

                html.H5("Mapa mundial — Prevalencia LTBI"),
                dcc.Graph(id="mapa-prevalencia-global")
            ]),

            # === Pestaña: EDA2 ===
            dcc.Tab(label="EDA2", children=[
                html.Br(),
                html.H3("Evolución por región"),
                html.Hr(style={"border": "1px solid #ccc", "margin": "40px 0"}),

                dcc.Graph(
                    figure=px.line(
                        df_imputado.groupby(["anio", "region_OMS"], as_index=False)
                        .agg(media_prevalencia=("prevalencia_contactos", "mean")),
                        x="anio", y="media_prevalencia", color="region_OMS",
                        title="Evolución de la prevalencia LTBI por región OMS",
                        markers=True, template="plotly_white"
                    )
                )
            ]),

            # === Pestaña: EDA3 ===
            dcc.Tab(label="EDA3", children=[
                html.Br(),
                html.Hr(),
                html.H3("Análisis Estadístico de Variables Numéricas"),
                html.P("Selecciona una variable para visualizar su distribución:"),
                dcc.Dropdown(
                    id="dropdown-variable-eda",
                    options=[{"label": col, "value": col}
                             for col in df_imputado.select_dtypes(include="number").columns],
                    value=df_imputado.select_dtypes(include="number").columns[0],
                    clearable=False
                ),
                html.Hr(style={"border": "1px solid #ccc", "margin": "40px 0"}),
                html.Br(),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="histograma-variable-eda"), md=6),
                    dbc.Col(dcc.Graph(id="boxplot-variable-eda"), md=6)
                ]),
                html.Br(),
                html.Hr(style={"border": "1px solid #ccc", "margin": "40px 0"}),
                html.H3("Resumen estadístico general"),
                html.Div(id="tabla-resumen-eda")
            ]),

            # === Pestaña: Visualización del modelo ===
            dcc.Tab(label="Visualización del modelo", children=[
                html.H3("Visualización de Modelos"),
                html.Div([
                    html.Label("Selecciona un modelo:"),
                    dcc.Dropdown(
                        id="dropdown-modelo",
                        options=[{"label": k, "value": k} for k in modelos_pack.keys()],
                        value=list(modelos_pack.keys())[0],
                        clearable=False,
                        style={"width": "50%"}
                    )
                ], style={"marginBottom": "30px"}),
                html.Hr(style={"border": "1px solid #ccc", "margin": "40px 0"}),
                html.Div([
                    dcc.Graph(id="grafico1", style={"height": "400px"}),
                    dcc.Graph(id="grafico2", style={"height": "400px"}),
                    dcc.Graph(id="grafico3", style={"height": "400px"}),
                    dcc.Graph(id="grafico4", style={"height": "400px"})
                ], style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gridGap": "20px",
                    "padding": "10px 40px"
                })
            ]),

            # === Pestaña: Métricas de Modelos ===
            dcc.Tab(label="Métricas de Modelos", children=[
                html.Hr(style={"border": "1px solid #ccc", "margin": "40px 0"}),
                html.H3("Análisis de Métricas de Modelos"),

                html.Div([
                    html.Label("Selecciona un modelo:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="dropdown-modelo-metricas",
                        options=[{"label": k, "value": k} for k in modelos_pack.keys()],
                        value=list(modelos_pack.keys())[0],
                        clearable=False,
                        style={"width": "50%"}
                    )
                ], style={"marginBottom": "30px"}),

                html.Div(id="tarjetas-metricas", style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "20px",
                    "marginBottom": "40px",
                    "padding": "0 40px"
                }),

                html.Hr(style={"border": "1px solid #ccc", "margin": "40px 0"}),

                html.H3("Comparador de métricas entre modelos"),

                html.Div([
                    html.Label("Selecciona los modelos a comparar:", style={
                        "fontWeight": "bold",
                        "display": "block",
                        "marginBottom": "10px"
                    }),
                    html.Div([
                        dcc.Checklist(
                            id="checklist-metricas",
                            options=[{"label": k, "value": k} for k in modelos_pack.keys()],
                            value=[list(modelos_pack.keys())[0]],
                            inline=False                        )
                    ], style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "10px",
                        "marginBottom": "30px"
                    }),
                    dcc.Graph(id="grafico-metricas", style={"height": "500px", "marginBottom": "40px"})
                ])
            ]),
        ])
        return tabs



    elif boton_id == "btn-8":
        return dbc.Card([
            dbc.CardBody([
               html.Div(
    [
        html.H2("Conclusiones", style={"marginTop": "20px"}),

        html.P(
            """El análisis del dashboard evidencia que la prevalencia global de infección latente 
            por tuberculosis (LTBI) ha mostrado una disminución gradual en los últimos años. 
            Sin embargo, esta reducción aún es insuficiente para alcanzar los objetivos de eliminación 
            de la tuberculosis si no se interviene de forma decidida sobre el reservorio latente.
            """
        ),

        html.P(
            """Se observa además una marcada heterogeneidad regional: regiones como África y el 
            Sudeste Asiático concentran la mayor carga de LTBI, lo que refuerza la necesidad de 
            estrategias diferenciadas según el contexto epidemiológico."""
        ),

        html.P(
            """Los resultados resaltan también la importancia de priorizar a poblaciones vulnerables 
            y grupos de mayor riesgo para pruebas y tratamiento preventivo. La calidad variable de los 
            datos entre países subraya la utilidad del dashboard como herramienta para visualizar 
            tendencias, identificar brechas y apoyar decisiones basadas en evidencia."""
        ),

        html.P(
            """En conjunto, estos hallazgos ofrecen una visión clara de los desafíos actuales y de 
            las oportunidades para avanzar en el control de la LTBI."""
        ),

        html.H2("Recomendaciones", style={"marginTop": "30px"}),

        html.Ul(
            [
                html.Li(
                    "Fortalecer el tamizaje y tratamiento de LTBI en regiones con alta prevalencia, "
                    "especialmente entre contactos cercanos de casos activos."
                ),
                html.Li(
                    "Ampliar la cobertura de terapia preventiva para reducir la progresión hacia tuberculosis activa."
                ),
                html.Li(
                    "Mejorar la calidad y consistencia de los datos mediante sistemas de reporte estandarizados a nivel nacional y regional."
                ),
                html.Li(
                    "Expandir el dashboard con análisis adicionales, como escenarios futuros, estimación de reactivaciones evitadas "
                    "o desagregación por grupos poblacionales."
                ),
                html.Li(
                    "Alinear estrategias nacionales con las recomendaciones de la OMS, promoviendo colaboración entre instituciones locales "
                    "e internacionales."
                ),
            ]
        ),
    ],
    style={
        "padding": "20px",
        "textAlign": "justify",
        "lineHeight": "1.6",
    },
)

            ])
        ])

# ====================================================
# 📊 CALLBACKS para los gráficos de análisis EDA (btn-7)
# ====================================================

@app.callback(
    [
        Output("histograma-variable-eda", "figure"),
        Output("boxplot-variable-eda", "figure"),
        Output("tabla-resumen-eda", "children")
    ],
    Input("dropdown-variable-eda", "value")
)
def actualizar_analisis_eda(variable):
    if variable is None:
        raise dash.exceptions.PreventUpdate

    # --- Histograma ---
    fig_hist = px.histogram(
        df_imputado, x=variable, nbins=30, marginal="rug",
        title=f"Histograma — {variable}",
        color_discrete_sequence=["#2C3E50"]
    )
    fig_hist.update_layout(template="plotly_white")

    # --- Boxplot ---
    fig_box = px.box(
        df_imputado, y=variable, points="all",
        title=f"Boxplot — {variable}",
        color_discrete_sequence=["#2C3E50"]
    )
    fig_box.update_layout(template="plotly_white")

    # --- Tabla resumen ---
    num_cols = df_imputado.select_dtypes(include="number").columns
    stats = pd.DataFrame({
        "Promedio": df_imputado[num_cols].mean(),
        "Varianza": df_imputado[num_cols].var(),
        "Máximo": df_imputado[num_cols].max(),
        "Mínimo": df_imputado[num_cols].min()
    }).round(2)

    table = dbc.Table.from_dataframe(stats, striped=True, bordered=True, hover=True)

    return fig_hist, fig_box, table


@app.callback(
    [
        Output("grafico-evolucion-global", "figure"),
        Output("heatmap-region-anio", "figure"),
        Output("tarjetas-resumen", "children"),
        Output("mapa-prevalencia-global", "figure")
    ],
    Input("slider-rango-anios", "value")
)
def actualizar_vista_global(rango_anios):
    anio_min, anio_max = rango_anios

    df_filtrado = df_imputado[
        (df_imputado["anio"] >= anio_min) & (df_imputado["anio"] <= anio_max)
    ]

    evol_global = df_filtrado.groupby("anio", as_index=False).agg(
        media_prevalencia=("prevalencia_contactos", "mean")
    )
    fig_linea = px.line(
        evol_global, x="anio", y="media_prevalencia",
        markers=True, title=f"Evolución global LTBI ({anio_min}–{anio_max})",
        color_discrete_sequence=["#2C3E50"], template="plotly_white"
    )

    # --- Heatmap por región ---
    heat_data = df_filtrado.groupby(["region_OMS", "anio"], as_index=False).agg(
        media_prevalencia=("prevalencia_contactos", "mean")
    )
    heatmap_data = heat_data.pivot(index="region_OMS", columns="anio", values="media_prevalencia")
    fig_heat = px.imshow(
        heatmap_data, color_continuous_scale="ice",
        labels=dict(x="Año", y="Región OMS", color="Prevalencia (%)"),
        title="Prevalencia promedio LTBI por región y año"
    )

    # --- Tarjetas resumen ---
    tarjetas = [
        ("Años Analizados", anio_max - anio_min + 1, "#2980B9"),
        ("Prevalencia Promedio", f"{df_filtrado['prevalencia_contactos'].mean():.2f}%", "#27AE60"),
        ("Diferencia Promedio (Sup - Inf)", f"{(df_filtrado['prevalencia_superior'] - df_filtrado['prevalencia_inferior']).mean():.2f}%", "#E67E22")
    ]
    cards = [
        html.Div(
            [
                html.H5(titulo, style={"color": "#2C3E50"}),
                html.H3(valor, style={"color": color}),
            ],
            style={
                "backgroundColor": "#f0f4f8",
                "padding": "20px",
                "borderRadius": "10px",
                "border": "1px solid #ddd",
                "marginBottom": "10px"
            }
        )
        for titulo, valor, color in tarjetas
    ]

    # --- Mapa mundial ---
    anio_sel = anio_max
    df_anio = df_filtrado[df_filtrado["anio"] == anio_sel]
    fig_map = px.choropleth(
        df_anio,
        locations="country",
        locationmode="country names",
        color="prevalencia_contactos",
        hover_name="country",
        color_continuous_scale="ice",
        title=f"Mapa mundial de prevalencia LTBI ({anio_sel})",
        labels={"prevalencia_contactos": "Prevalencia (%)"}
    )
    fig_map.update_layout(margin=dict(r=0, t=50, l=0, b=0))

    return fig_linea, fig_heat, cards, fig_map

from dash import Input, Output
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

@app.callback(
    [Output("grafico1", "figure"),
     Output("grafico2", "figure"),
     Output("grafico3", "figure"),
     Output("grafico4", "figure")],
    [Input("dropdown-modelo", "value")]
)
def actualizar_visualizacion(modelo_sel):
    pack = modelos_pack[modelo_sel]

    y_test = pack["y_test"]
    y_pred = pack["y_pred"]
    metricas = pack["metricas"]

    # Paleta azul oscuro consistente
    azul_principal = "#1A5276"
    azul_suave = "#2874A6"
    azul_claro = "#85C1E9"

    # --- 1️⃣ Real vs Predicho ---
    fig1 = px.scatter(
        x=y_test, y=y_pred,
        title=f"{modelo_sel}: Reales vs Predichos",
        labels={"x": "Reales", "y": "Predichos"},
        color_discrete_sequence=[azul_principal]
    )
    fig1.add_trace(go.Scatter(
        x=[min(y_test), max(y_test)],
        y=[min(y_test), max(y_test)],
        mode="lines",
        line=dict(color="#E74C3C", dash="dash"),
        name="Ideal"
    ))
    fig1.update_layout(template="plotly_white", font=dict(color="#2C3E50"))

    # --- 2️⃣ Curva de regresión ---
    fig2 = px.scatter(
        x=y_test, y=y_pred,
        trendline="ols",
        title=f"{modelo_sel}: Curva de regresión ajustada",
        labels={"x": "Reales", "y": "Predichos"},
        color_discrete_sequence=[azul_suave]
    )
    fig2.update_traces(marker=dict(size=6))
    fig2.update_layout(template="plotly_white", font=dict(color="#2C3E50"))

    # --- 3️⃣ Residuos vs Predicción ---
    residuos = y_test - y_pred
    fig3 = px.scatter(
        x=y_pred, y=residuos,
        title=f"{modelo_sel}: Residuos vs Predicción",
        labels={"x": "Predichos", "y": "Residuos"},
        color_discrete_sequence=[azul_principal]
    )
    fig3.add_hline(y=0, line_dash="dash", line_color="#E74C3C")
    fig3.update_layout(template="plotly_white", font=dict(color="#2C3E50"))

    # --- 4️⃣ Distribución Reales vs Predichos ---
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=y_test, name="Reales", opacity=0.6, marker_color=azul_principal))
    fig4.add_trace(go.Histogram(x=y_pred, name="Predichos", opacity=0.6, marker_color=azul_claro))
    fig4.update_layout(
        barmode="overlay",
        title=f"{modelo_sel}: Distribución Reales vs Predichos",
        template="plotly_white",
        font=dict(color="#2C3E50"),
        legend=dict(title="Tipo de dato")
    )
    fig4.update_traces(opacity=0.5)


    # --- Tabla de métricas ---
    tabla = dash_table.DataTable(
        columns=[{"name": k, "id": k} for k in metricas.keys()],
        data=[metricas],
        style_table={"margin": "auto"},
        style_header={"backgroundColor": "#2C3E50", "color": "white"},
        style_cell={"textAlign": "center", "fontSize": 15}
    )

    return fig1, fig2, fig3, fig4


@app.callback(
    [Output("tarjetas-metricas", "children"),
     Output("grafico-metricas", "figure")],
    [Input("dropdown-modelo-metricas", "value"),
     Input("checklist-metricas", "value")]
)
def actualizar_metricas(modelo_seleccionado, modelos_comparar):
    tarjetas = []
    df_metricas = []

    # === Tarjetas individuales ===
    for nombre, datos in modelos_pack.items():
        if nombre == modelo_seleccionado:
            metricas = datos["metricas"]
            for k, v in metricas.items():
                tarjetas.append(
                    html.Div([
                        html.H4(k, style={"color": "#1A5276", "fontWeight": "bold"}),
                        html.H2(f"{v:.3f}", style={"color": "#2980B9", "fontWeight": "bold"})
                    ], style={
                        "backgroundColor": "#EAF2F8",
                        "padding": "20px",
                        "borderRadius": "12px",
                        "textAlign": "center",
                        "width": "90%",
                        "boxShadow": "2px 2px 6px rgba(0,0,0,0.1)",
                        "border": "1px solid #AED6F1"
                    }))
    
    # === Comparador de métricas ===
    for modelo in modelos_comparar:
        m = modelos_pack[modelo]["metricas"]
        for k, v in m.items():
            df_metricas.append({"Modelo": modelo, "Métrica": k, "Valor": v})

    df_metricas = pd.DataFrame(df_metricas)

    # Paleta de tonos azules
    palette_azul = ["#0D47A1", "#1565C0", "#1976D2", "#1E88E5", "#42A5F5", "#64B5F6"]

    fig = px.bar(
        df_metricas,
        x="Métrica",
        y="Valor",
        color="Modelo",
        barmode="group",
        title="Comparación de métricas entre modelos",
        text_auto=True,
        color_discrete_sequence=palette_azul
    )

    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=20, color="#154360"),
        xaxis_title="Métrica",
        yaxis_title="Valor",
        legend_title="Modelo",
        plot_bgcolor="#F8FBFF",
        paper_bgcolor="#F8FBFF",
        font=dict(color="#2C3E50"),
        margin=dict(t=60, l=40, r=40, b=40)
    )

    fig.update_traces(
        textfont_size=12,
        textangle=0,
        textposition="outside",
        cliponaxis=False
    )

    return tarjetas, fig


if __name__ == "__main__":
       app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)






























