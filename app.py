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
    [Input(f"btn-{i}", "n_clicks") for i in range(1, 9)]
)
def mostrar_contenido(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.P("Selecciona una sección del informe para comenzar.", className="text-muted text-center")

    boton_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if boton_id == "btn-1":
        return dbc.Card([
            dbc.CardBody([
                html.H4("Introducción", className="card-title"),
                html.P("""
                La tuberculosis (TB) sigue siendo una de las enfermedades infecciosas más relevantes a nivel global, con un impacto sostenido en la salud pública a pesar de los avances en diagnóstico y tratamiento. Una proporción significativa de la población mundial vive con una infección latente por Mycobacterium tuberculosis (LTBI), lo que implica la presencia de la bacteria sin manifestación activa de la enfermedad, pero con riesgo potencial de desarrollarla en el futuro. Comprender la magnitud y evolución de esta infección es fundamental para orientar las políticas de prevención y control que promueve la Organización Mundial de la Salud (OMS).
                """),
                html.P("""
                El presente análisis se basa en las estimaciones publicadas por la OMS dentro del Global Tuberculosis Report 2024, específicamente en el apartado sobre infección latente por tuberculosis (LTBI) en contactos domiciliarios, que representa a uno de los grupos poblacionales con mayor vulnerabilidad frente a la transmisión de la enfermedad. Este tipo de información permite observar la situación global de la infección y los avances alcanzados en la detección y contención de la tuberculosis durante las últimas décadas."""),
                html.P("""
                En este contexto, el análisis busca responder a la siguiente pregunta:"""),
                html.P("""
                ¿Cómo ha variado la prevalencia estimada de infección latente por tuberculosis (LTBI) en contactos domiciliarios a nivel mundial entre los años 2000 y 2024, y qué regiones presentan los mayores cambios en sus estimaciones durante este período?"""),
                html.H4("Créditos", className="card-title"),
                html.Ul([
                    html.Li("Miguel Ángel Pérez"),  
                    html.Li("Camilo Vargas Escorcia"),
                    html.Li("Juan Camilo Aguirre"),
                ]),
                html.P("Repositorio: https://github.com/juantellas")
            ])
        ])


    elif boton_id == "btn-2":
        return dbc.Card([
            dbc.CardBody([
                html.H4("Contexto Global", className="card-title"),
                html.P("""
                La OMS ha reportado que la carga de tuberculosis varía ampliamente entre regiones, siendo más alta
                en países de bajos ingresos. El análisis se centra en la prevalencia estimada de infección latente
                en contactos domiciliarios.
                """)
            ])
        ])

    elif boton_id == "btn-3":
        return dbc.Card([
            dbc.CardBody([
                html.H4("Planteamiento del Problema", className="card-title"),
                html.P("""
                La infección latente por tuberculosis (LTBI) representa una condición en la que la bacteria
                Mycobacterium tuberculosis está presente sin causar enfermedad activa.
                La probabilidad de progresar a TB activa depende de factores inmunológicos y ambientales.
                """)
            ])
        ])

    elif boton_id == "btn-4":
        return dbc.Card([
            dbc.CardBody([
                html.H4("Objetivos y Justificación", className="card-title"),
                html.P("""
                - Analizar la evolución de la prevalencia de LTBI en el tiempo.
                - Identificar regiones con mayores cambios.
                - Promover estrategias basadas en evidencia para el control de la tuberculosis.
                """)
            ])
        ])

    elif boton_id == "btn-5":
        return dbc.Card([
            dbc.CardBody([
                html.H4("Marco Teorico", className="card-title"),
                html.P("""
                Se utilizaron datos de la OMS (2015-2023). El procesamiento de datos incluyó imputación de valores
                faltantes y análisis descriptivo mediante Pandas, Seaborn y Plotly.
                """)
            ])
        ])
    

    elif boton_id == "btn-6":
        return dbc.Card([
            dbc.CardBody([
                html.H4("Metodologia", className="card-title"),
                html.P("Proyecto académico — Visualización de Datos 2025.")
            ])
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
                html.H4("Conclusiones", className="card-title"),
                html.P("""
                Los resultados reflejan una tendencia general a la disminución de la prevalencia LTBI.
                Sin embargo, persisten brechas regionales que requieren atención focalizada.
                """)
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




