{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
from pmdarima import auto_arima\
import matplotlib.pyplot as plt\
from io import BytesIO\
\
st.set_page_config(page_title="Pron\'f3stico semanal (ARIMA)", layout="wide")\
\
st.title("\uc0\u55357 \u56520  Pron\'f3stico semanal por producto (ARIMA autom\'e1tico)")\
\
# -----------------------\
# Plantilla descargable\
# -----------------------\
@st.cache_data\
def generar_plantilla_bytes():\
    df = pd.DataFrame(\{\
        "producto": ["A", "A", "A", "B", "B", "B"],\
        "fecha": ["2024-01-01", "2024-01-02", "2024-01-03",\
                  "2024-01-01", "2024-01-02", "2024-01-03"],\
        "demanda": [100, 120, 90, 40, 38, 45]\
    \})\
    buffer = BytesIO()\
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:\
        df.to_excel(writer, index=False, sheet_name="plantilla")\
    return buffer.getvalue()\
\
st.subheader("\uc0\u55357 \u56516  Plantilla de datos")\
st.write("Descarga una plantilla Excel con las columnas: `producto`, `fecha`, `demanda`.")\
st.download_button(\
    label="\uc0\u55357 \u56549  Descargar plantilla (Excel)",\
    data=generar_plantilla_bytes(),\
    file_name="plantilla_datos_demanda.xlsx",\
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"\
)\
\
st.markdown("---")\
\
# -----------------------\
# Subir archivo\
# -----------------------\
st.subheader("\uc0\u55357 \u56513  Carga tus datos")\
archivo = st.file_uploader("Sube tu archivo (CSV o XLSX) con columnas: producto, fecha, demanda", type=["csv", "xlsx"])\
\
if not archivo:\
    st.info("Sube un archivo para ver pron\'f3sticos. Puedes usar la plantilla descargable.")\
    st.stop()\
\
# -----------------------\
# Leer y validar archivo\
# -----------------------\
try:\
    if archivo.name.endswith(".csv"):\
        df = pd.read_csv(archivo)\
    else:\
        df = pd.read_excel(archivo)\
except Exception as e:\
    st.error(f"No se pudo leer el archivo: \{e\}")\
    st.stop()\
\
st.write("Vista previa de los datos:")\
st.dataframe(df.head())\
\
columnas_requeridas = ["producto", "fecha", "demanda"]\
if not all(col in df.columns for col in columnas_requeridas):\
    st.error(f"El archivo debe contener las columnas: \{columnas_requeridas\}")\
    st.stop()\
\
# convertir fecha y normalizar columnas\
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")\
if df["fecha"].isna().any():\
    st.warning("Hay filas con fecha inv\'e1lida. Se eliminar\'e1n las filas sin fecha.")\
    df = df.dropna(subset=["fecha"])\
\
df = df.sort_values(["producto", "fecha"]).reset_index(drop=True)\
\
# -----------------------\
# Par\'e1metros fijos\
# -----------------------\
st.write("La app agrupa los datos a **frecuencia semanal** (suma por semana) y genera **siempre 2 semanas** de pron\'f3stico.")\
pasos = 2  # siempre 2 semanas\
\
productos = df["producto"].unique()\
st.write(f"Productos detectados: **\{len(productos)\}**")\
\
# -----------------------\
# Procesar por producto\
# -----------------------\
resultados_finales = []\
progress = st.progress(0)\
total = len(productos)\
i = 0\
\
for prod in productos:\
    i += 1\
    progress.progress(int(i / total * 100))\
    st.markdown(f"### \uc0\u55357 \u56633  Producto: **\{prod\}**")\
\
    df_prod = df[df["producto"] == prod].copy()\
    df_prod = df_prod.sort_values("fecha")\
\
    # Agrupar por semana (suma)\
    serie = (\
        df_prod.set_index("fecha")["demanda"]\
        .resample("W")   # semanal\
        .sum()\
        .fillna(0)\
    )\
\
    if len(serie) < 2:\
        st.warning(f"No hay suficientes semanas para ajustar modelo en \{prod\}. Se necesita al menos 2 semanas con datos.")\
        continue\
\
    # Ajustar auto_arima\
    try:\
        modelo = auto_arima(\
            serie,\
            seasonal=False,\
            stepwise=True,\
            suppress_warnings=True,\
            error_action="ignore"\
        )\
    except Exception as e:\
        st.error(f"No se pudo ajustar ARIMA para \{prod\}: \{e\}")\
        continue\
\
    # Pron\'f3stico 2 semanas\
    try:\
        forecast = modelo.predict(n_periods=pasos)\
    except Exception as e:\
        st.error(f"Error al predecir para \{prod\}: \{e\}")\
        continue\
\
    fechas_futuras = pd.date_range(\
        start=serie.index.max() + pd.Timedelta(weeks=1),\
        periods=pasos,\
        freq="W"\
    )\
\
    # Hist\'f3ricos en formato requerido\
    df_hist = pd.DataFrame(\{\
        "producto": prod,\
        "semana": serie.index.strftime("%G-W%V"),\
        "demanda_semana": serie.values,\
        "pronostico_sem_1": [None] * len(serie),\
        "pronostico_sem_2": [None] * len(serie)\
    \})\
\
    # Futuras (2 filas) con pron\'f3sticos distribuidos en las columnas solicitadas\
    df_fut = pd.DataFrame(\{\
        "producto": prod,\
        "semana": fechas_futuras.strftime("%G-W%V"),\
        "demanda_semana": [None] * pasos,\
        "pronostico_sem_1": [forecast[0], None] if pasos == 2 else ([forecast[0]] + [None] * (pasos-1)),\
        "pronostico_sem_2": [None, forecast[1]] if pasos == 2 else ([None]* (pasos-1) + [forecast[-1]])\
    \})\
\
    df_total = pd.concat([df_hist, df_fut], ignore_index=True)\
    resultados_finales.append(df_total)\
\
    # Gr\'e1fica (hist\'f3rico semanal + pron\'f3stico)\
    fig, ax = plt.subplots(figsize=(8, 3.5))\
    ax.plot(serie.index, serie.values, label="Hist\'f3rico (semanal)")\
    ax.plot(fechas_futuras, forecast, linestyle="--", marker="o", label="Pron\'f3stico 2 semanas")\
    ax.set_title(f"Producto: \{prod\}")\
    ax.set_xlabel("Fecha")\
    ax.set_ylabel("Demanda (suma semanal)")\
    ax.legend()\
    st.pyplot(fig)\
\
progress.empty()\
\
if not resultados_finales:\
    st.error("No se generaron pron\'f3sticos para ning\'fan producto.")\
    st.stop()\
\
excel_final = pd.concat(resultados_finales, ignore_index=True)\
\
st.subheader("Tabla final con pron\'f3sticos")\
st.dataframe(excel_final.head(200))\
\
# -----------------------\
# Bot\'f3n de descarga del Excel final\
# -----------------------\
buffer = BytesIO()\
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:\
    excel_final.to_excel(writer, index=False, sheet_name="forecast_final")\
    # opcional: hojas separadas por producto\
    for prod in productos:\
        try:\
            df_prod_sheet = excel_final[excel_final["producto"] == prod]\
            df_prod_sheet.to_excel(writer, index=False, sheet_name=str(prod))\
        except Exception:\
            pass\
\
st.download_button(\
    label="\uc0\u55357 \u56549  Descargar pron\'f3sticos (formato solicitado)",\
    data=buffer.getvalue(),\
    file_name="pronosticos_semanales.xlsx",\
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"\
)\
\
st.success("Pron\'f3stico generado. Descarga el Excel usando el bot\'f3n de arriba.")}