# ---------- Imports ----------
import re
import unicodedata
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Formateador COP ----------
def fmt_cop(valor):
    try:
        if pd.isna(valor):
            return "$0 COP"
        return f"${valor:,.0f} COP"
    except Exception:
        return "$0 COP"

# ---------- Utilidades de texto/parseo ----------
def normalizar_texto(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s

def parse_fecha(x):
    if pd.isna(x) or str(x).strip() == "":
        return pd.NaT
    # intentos comunes
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(x, format=fmt, errors="raise")
        except Exception:
            continue
    # último recurso
    return pd.to_datetime(x, errors="coerce")

def to_number(x):
    if pd.isna(x):
        return 0.0
    s = str(x)
    # cambiar puntos de miles y coma decimal estilo ES/LATAM
    s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0

# ---------- Mapeo flexible de columnas ----------
# Nombres internos -> posibles variantes en SECOP/portales (incluye los que mostraste)
COLUMN_MAP = {
    "entidad": [
        "Entidad", "entidad", "nombre_entidad", "entidad contratante", "nombre de la entidad",
        "entidad_estatal", "buyer", "buyername"
    ],
    "proveedor": [
        "Nombre del Proveedor Adjudicado", "Proveedor", "Contratista", "nombre_proveedor",
        "supplier", "nombre del contratista"
    ],
    "nit_proveedor": [
        "NIT del Proveedor Adjudicado", "nit proveedor", "supplier_id", "nit", "identificacion proveedor"
    ],
    "objeto": [
        "Descripción del Procedimiento", "Objeto", "objeto del contrato", "descripcion", "description"
    ],
    "tipo_contrato": [
        "Tipo de Contrato", "tipo de contrato", "modalidad", "tipocontrato",
        "modalidad de contratacion", "contracttype"
    ],
    "valor": [  # columna de dinero
        "Valor Total Adjudicacion", "Valor del contrato", "valor", "valor total",
        "monto", "amount", "valor adjudicado", "valor_contrato", "Precio Base"
    ],
    "fecha": [
        "Fecha Adjudicacion", "Fecha de Publicacion del Proceso", "Fecha de Ultima Publicación",
        "fecha", "publicationdate", "awarddate", "fecha publicacion"
    ],
    "departamento": [
        "Departamento Entidad", "departamento", "region", "ubicacion", "departamento_ejecucion"
    ],
    "codigo_proceso": [
        "ID del Proceso", "codigo de proceso", "proceso", "processid",
        "id_proceso", "id proceso", "codigo_proceso"
    ],
}

def encontrar_columna(df, candidatos):
    cols_norm = {normalizar_texto(c): c for c in df.columns}
    for cand in candidatos:
        key = normalizar_texto(cand)
        if key in cols_norm:
            return cols_norm[key]
    # heurística simple de similitud
    for c in df.columns:
        if any(normalizar_texto(x) in normalizar_texto(c) for x in candidatos):
            return c
    return None

def estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for interno, variantes in COLUMN_MAP.items():
        real = encontrar_columna(df, variantes)
        if real:
            mapping[real] = interno
    return df.rename(columns=mapping).copy()

# ---------- Limpieza principal (SIN renombrar para la lógica) ----------
def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    # 1) estandarizar columnas
    df = estandarizar_columnas(df)

    # 2) asegurar columnas esenciales
    for c in ["entidad", "proveedor", "objeto", "tipo_contrato", "valor", "fecha",
              "departamento", "codigo_proceso", "nit_proveedor"]:
        if c not in df.columns:
            df[c] = pd.NA

    # 3) normalizaciones texto
    df["entidad"] = df["entidad"].apply(normalizar_texto).str.upper()
    df["proveedor"] = df["proveedor"].apply(normalizar_texto).str.upper()
    df["objeto"] = df["objeto"].apply(normalizar_texto)
    df["tipo_contrato"] = df["tipo_contrato"].apply(normalizar_texto).str.upper()

    # 4) tipos
    df["valor"] = df["valor"].apply(to_number)
    df["fecha"] = df["fecha"].apply(parse_fecha)

    # 5) derivadas
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month

    # 6) duplicados
    subset = [c for c in ["codigo_proceso", "proveedor", "valor", "fecha"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")

    # 7) filtros básicos
    df = df[df["valor"] >= 0]

    # 8) redondeo a pesos
    df["valor"] = df["valor"].round(0)

    return df

# ---------- UI ----------
st.set_page_config(page_title="Analizador SECOP - COP", layout="wide")
st.title("🔎 Prototipo: ETL + Análisis básico de SECOP (COP)")

archivo = st.file_uploader("Sube un archivo SECOP (.csv / .xlsx)", type=["csv", "xlsx"])
if archivo is None:
    st.info("Sube un archivo para comenzar.")
    st.stop()

# Lectura flexible
if archivo.name.lower().endswith(".csv"):
    df_raw = pd.read_csv(archivo, low_memory=False)
else:
    df_raw = pd.read_excel(archivo)  # requiere openpyxl instalado

st.subheader("Vista previa (crudo)")
st.dataframe(df_raw.head(20), use_container_width=True)

# ETL
df = limpiar(df_raw)

# ---- Filtros (usamos nombres internos) ----
with st.sidebar:
    st.header("Filtros")
    anios = sorted([int(a) for a in df["anio"].dropna().unique()]) if "anio" in df else []
    anio_sel = st.multiselect("Año", anios, default=anios)
    entidades = sorted(df["entidad"].dropna().unique()) if "entidad" in df else []
    entidad_sel = st.multiselect("Entidad", entidades, default=entidades[:10] if len(entidades) > 10 else entidades)

df_f = df.copy()
if anio_sel:
    df_f = df_f[df_f["anio"].isin(anio_sel)]
if entidad_sel:
    df_f = df_f[df_f["entidad"].isin(entidad_sel)]

# ---- KPIs (COP) ----
col1, col2, col3 = st.columns(3)
total_contratos = len(df_f)
total_valor = float(df_f["valor"].sum()) if "valor" in df_f else 0.0
proveedores_unicos = int(df_f["proveedor"].nunique()) if "proveedor" in df_f else 0

col1.metric("Contratos", f"{total_contratos:,}".replace(",", "."))
col2.metric("Valor total (COP)", fmt_cop(total_valor))
col3.metric("Proveedores únicos", f"{proveedores_unicos:,}".replace(",", "."))

st.markdown("---")

# ---- Tabla “Datos limpios (muestra)” con nombres claros SOLO para mostrar ----
st.subheader("Datos limpios (muestra)")
df_display = df_f.copy().rename(columns={
    "entidad": "Entidad",
    "nit_proveedor": "NIT Proveedor",
    "proveedor": "Proveedor",
    "departamento": "Departamento",
    "tipo_contrato": "Tipo de Contrato",
    "valor": "Valor (COP)",
    "fecha": "Fecha del Proceso",
    "anio": "Año",
    "mes": "Mes",
})
if "Valor (COP)" in df_display.columns:
    df_display["Valor (COP) (formateado)"] = df_display["Valor (COP)"].apply(fmt_cop)

st.dataframe(df_display.head(20), use_container_width=True)

# ---- Gráfico 1: Evolución temporal (COP) ----
if "fecha" in df_f.columns and df_f["fecha"].notna().any():
    serie = df_f.dropna(subset=["fecha"]).copy()
    serie["periodo"] = pd.to_datetime(serie["fecha"]).dt.to_period("M").dt.to_timestamp()
    g1 = (serie.groupby("periodo", as_index=False)["valor"].sum().sort_values("periodo"))
    fig1 = px.line(
        g1, x="periodo", y="valor", markers=True,
        title="Evolución del monto contratado (COP)",
        labels={"periodo": "Periodo", "valor": "Pesos (COP)"},
    )
    fig1.update_layout(yaxis_tickformat=",")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("No se encontraron columnas de fecha/valor para graficar la evolución.")

# ---- Gráfico 2: Top 10 proveedores por monto (COP) ----
if "proveedor" in df_f.columns and "valor" in df_f.columns:
    top_prov = (df_f.groupby("proveedor", as_index=False)["valor"]
                .sum().sort_values("valor", ascending=False).head(10))
    fig2 = px.bar(
        top_prov, x="proveedor", y="valor",
        title="Top 10 proveedores por monto (COP)",
        labels={"proveedor": "Proveedor", "valor": "Pesos (COP)"},
    )
    fig2.update_layout(xaxis=dict(tickangle=-45), yaxis_tickformat=",")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No se encontraron columnas de proveedor/valor.")

# --- Descargas: Excel ---
import io

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="DatosLimpios")
    writer.close()

st.download_button(
    "📥 Descargar datos limpios (Excel)",
    data=buffer.getvalue(),
    file_name="secop_limpio.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)