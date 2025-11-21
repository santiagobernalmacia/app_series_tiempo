import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from pmdarima import auto_arima

st.title("📈 Pronóstico de Demanda por Producto (ARIMA Semanal)")

# ------------------------------------------------------------
# DESCARGA DE PLANTILLA
# ------------------------------------------------------------
def generar_plantilla():
    df = pd.DataFrame({
        "producto": ["A", "A", "A", "B", "B", "B"],
        "fecha": ["2024-01-01", "2024-01-02", "2024-01-03",
                  "2024-01-01", "2024-01-02", "2024-01-03"],
        "demanda": [100, 120, 90, 40, 38, 45]
    })

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="plantilla")
    return buffer.getvalue()

st.subheader("📄 Descargar plantilla de datos")
st.download_button(
    label="📥 Descargar plantilla Excel",
    data=generar_plantilla(),
    file_name="plantilla_demanda.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.write("---")


# ------------------------------------------------------------
# SUBIR ARCHIVO
# ------------------------------------------------------------
archivo = st.file_uploader("📤 Sube tu archivo (Excel o CSV)", type=["xlsx", "csv"])

if archivo is not None:
    # Cargar archivo
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)

    # ------------------------------------------------------------
    # VALIDACIÓN DE COLUMNAS
    # ------------------------------------------------------------
    columnas_necesarias = {"producto", "fecha", "demanda"}
    if not columnas_necesarias.issubset(df.columns):
        st.error("❌ El archivo debe tener las columnas: producto, fecha, demanda")
        st.stop()

    # Convertir fecha
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Validar que no haya fechas vacías
    if df["fecha"].isna().any():
        st.error("❌ Hay fechas inválidas en el archivo.")
        st.stop()

    st.success("✅ Archivo cargado correctamente")

    # ------------------------------------------------------------
    # AGRUPACIÓN SEMANAL + ARIMA + PRONÓSTICO 2 SEMANAS
    # ------------------------------------------------------------
    productos = df["producto"].unique()
    resultados_finales = []

    for prod in productos:
        df_prod = df[df["producto"] == prod].copy()
        df_prod = df_prod.sort_values("fecha")

        # ---- Agrupación semanal ----
        serie = (
            df_prod.set_index("fecha")["demanda"]
            .resample("W")  # semana
            .sum()
            .fillna(0)
        )

        if len(serie) < 3:
            # Muy poca data → saltamos el producto
            continue

        # ---- Ajuste ARIMA ----
        try:
            modelo = auto_arima(
                serie,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
            )
        except:
            continue

        # Pronóstico fijo 2 semanas
        pasos = 2
        forecast = modelo.predict(n_periods=pasos)

        fechas_futuras = pd.date_range(
            start=serie.index.max() + pd.Timedelta(weeks=1),
            periods=pasos,
            freq="W"
        )

        # ---- Construcción del dataset final ----
        df_hist = pd.DataFrame({
            "producto": prod,
            "semana": serie.index.strftime("%G-W%V"),
            "demanda_semana": serie.values,
            "pronostico_sem_1": [None] * len(serie),
            "pronostico_sem_2": [None] * len(serie)
        })

        df_fut = pd.DataFrame({
            "producto": prod,
            "semana": fechas_futuras.strftime("%G-W%V"),
            "demanda_semana": [None, None],
            "pronostico_sem_1": [forecast[0], None],
            "pronostico_sem_2": [None, forecast[1]]
        })

        resultados_finales.append(pd.concat([df_hist, df_fut], ignore_index=True))

    # ------------------------------------------------------------
    # UNIR Y EXPORTAR EXCEL
    # ------------------------------------------------------------
    if resultados_finales:
        excel_final = pd.concat(resultados_finales, ignore_index=True)

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            excel_final.to_excel(writer, index=False, sheet_name="pronosticos")

        st.download_button(
            label="📥 Descargar pronósticos semanales",
            data=buffer.getvalue(),
            file_name="pronosticos_semanales.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("✅ Pronósticos generados correctamente")

    else:
        st.warning("⚠ No se pudo generar pronósticos para ningún producto.")
