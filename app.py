# ---------- Imports ----------
import re
import unicodedata
from datetime import datetime
import io

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Config ----------
st.set_page_config(page_title="Analizador SECOP - COP", layout="wide")

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
    "valor": [  # dinero
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

# ---------- Limpieza principal ----------
@st.cache_data(show_spinner=False)
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

# ---------- Lectura de archivo ----------
st.title("🔎 Prototipo: ETL + Análisis básico de SECOP (COP)")
archivo = st.file_uploader("Sube un archivo SECOP (.csv / .xlsx)", type=["csv", "xlsx"])
if archivo is None:
    st.info("Sube un archivo para comenzar.")
    st.stop()

@st.cache_data(show_spinner=False)
def read_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        # intento con utf-8 y backup con latin-1
        try:
            return pd.read_csv(file, low_memory=False)
        except Exception:
            file.seek(0)
            return pd.read_csv(file, low_memory=False, encoding="latin-1")
    else:
        return pd.read_excel(file)  # requiere openpyxl

df_raw = read_any(archivo)

st.subheader("Vista previa (crudo)")
st.dataframe(df_raw.head(20), use_container_width=True)

# ---------- ETL ----------
df = limpiar(df_raw)

# ---------- Filtros (mejorados) ----------
with st.sidebar:
    st.header("Filtros")

    # Años como rango
    if "anio" in df.columns and df["anio"].notna().any():
        min_anio = int(df["anio"].min())
        max_anio = int(df["anio"].max())
        anio_rango = st.slider(
            "Rango de años",
            min_value=min_anio,
            max_value=max_anio,
            value=(min_anio, max_anio),
            step=1,
        )
    else:
        anio_rango = None

    # Entidades con "Seleccionar todas"
    entidades = sorted([x for x in df["entidad"].dropna().unique()]) if "entidad" in df else []
    sel_all_ent = st.checkbox("Seleccionar todas las entidades", value=True)
    entidad_sel = st.multiselect(
        "Entidad",
        entidades,
        default=entidades if sel_all_ent else entidades[:10]
    )

    # Filtros avanzados
    with st.expander("Filtros avanzados"):
        # Departamento
        if "departamento" in df.columns:
            deps = sorted([x for x in df["departamento"].dropna().unique()])
            dep_sel = st.multiselect("Departamento", deps, default=[])
        else:
            dep_sel = []

        # Tipo de contrato
        if "tipo_contrato" in df.columns:
            tipos = sorted([x for x in df["tipo_contrato"].dropna().unique()])
            tipo_sel = st.multiselect("Tipo de contrato", tipos, default=[])
        else:
            tipo_sel = []

        # Rango de valores
        vmin, vmax = 0.0, float(df["valor"].max()) if "valor" in df else 0.0
        if vmax is None or pd.isna(vmax):
            vmax = 0.0
        rango_valor = st.slider(
            "Rango de valores (COP)",
            min_value=float(vmin),
            max_value=float(vmax),
            value=(float(vmin), float(vmax)),
            step=1.0
        )

        # Búsqueda en objeto
        term_obj = st.text_input("Buscar en objeto (palabra clave)", value="").strip().lower()

# aplicar filtros
df_f = df.copy()

# rango de años
if anio_rango and "anio" in df_f.columns:
    a0, a1 = anio_rango
    df_f = df_f[(df_f["anio"] >= a0) & (df_f["anio"] <= a1)]

# entidad
if entidad_sel:
    df_f = df_f[df_f["entidad"].isin(entidad_sel)]

# departamento
if dep_sel:
    df_f = df_f[df_f["departamento"].isin(dep_sel)]

# tipo contrato
if tipo_sel:
    df_f = df_f[df_f["tipo_contrato"].isin(tipo_sel)]

# rango valor
if "valor" in df_f.columns and rango_valor:
    v0, v1 = rango_valor
    df_f = df_f[(df_f["valor"] >= v0) & (df_f["valor"] <= v1)]

# objeto contiene
if term_obj and "objeto" in df_f.columns:
    df_f = df_f[df_f["objeto"].str.contains(term_obj, na=False)]

# ---------- KPIs ----------
col1, col2, col3 = st.columns(3)
total_contratos = len(df_f)
total_valor = float(df_f["valor"].sum()) if "valor" in df_f else 0.0
proveedores_unicos = int(df_f["proveedor"].nunique()) if "proveedor" in df_f else 0

col1.metric("Contratos", f"{total_contratos:,}".replace(",", "."))
col2.metric("Valor total (COP)", fmt_cop(total_valor))
col3.metric("Proveedores únicos", f"{proveedores_unicos:,}".replace(",", "."))

st.markdown("---")

# ---------- Selector de columnas (usuario decide) ----------
st.subheader("Datos limpios (muestra)")

# nombres “amigables” para la vista/descarga
friendly_cols = {
    "entidad": "Entidad",
    "nit_proveedor": "NIT Proveedor",
    "proveedor": "Proveedor",
    "departamento": "Departamento",
    "tipo_contrato": "Tipo de Contrato",
    "valor": "Valor (COP)",
    "fecha": "Fecha del Proceso",
    "anio": "Año",
    "mes": "Mes",
    "codigo_proceso": "ID / Código del Proceso",
    "objeto": "Objeto / Descripción",
}

# columnas disponibles en el df filtrado (solo las que existan)
available_internal = [c for c in friendly_cols.keys() if c in df_f.columns]

# plantillas rápidas
plantilla = st.radio(
    "Plantilla de columnas",
    ["Básico", "Detallado", "Personalizado"],
    horizontal=True
)

tpl_basico = [c for c in ["entidad", "proveedor", "valor", "fecha", "anio"] if c in available_internal]
tpl_detallado = [c for c in ["entidad", "nit_proveedor", "proveedor", "departamento",
                             "tipo_contrato", "valor", "fecha", "anio", "mes",
                             "codigo_proceso", "objeto"] if c in available_internal]

if plantilla == "Básico":
    pre_sel = tpl_basico
elif plantilla == "Detallado":
    pre_sel = tpl_detallado
else:
    # Personalizado arranca con detallado por comodidad (puedes cambiar a vacío [])
    pre_sel = tpl_detallado

cols_sel = st.multiselect(
    "Elige las columnas a mostrar/descargar",
    options=available_internal,
    default=pre_sel,
    format_func=lambda c: friendly_cols.get(c, c)
)

# construir df para mostrar
if cols_sel:
    df_view = df_f[cols_sel].rename(columns=friendly_cols).copy()
    if "Valor (COP)" in df_view.columns:
        df_view["Valor (COP) (formateado)"] = df_view["Valor (COP)"].apply(fmt_cop)
else:
    st.info("Selecciona al menos una columna para ver la tabla.")
    df_view = pd.DataFrame()

st.dataframe(df_view.head(20), use_container_width=True)

# ---------- Gráfico 1: Evolución temporal ----------
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

# ---------- Gráfico 2: Top 10 proveedores ----------
if "proveedor" in df_f.columns and "valor" in df_f.columns and not df_f.empty:
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

st.markdown("---")

# ---------- Descargas (respetan columnas escogidas) ----------
colA, colB = st.columns(2)
if not df_view.empty:
    # CSV
    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    colA.download_button(
        "📥 Descargar (CSV)",
        data=csv_bytes,
        file_name="secop_limpio.csv",
        mime="text/csv"
    )

    # Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_view.to_excel(writer, index=False, sheet_name="DatosLimpios")
        writer.close()
    colB.download_button(
        "📥 Descargar (Excel)",
        data=buffer.getvalue(),
        file_name="secop_limpio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("No hay datos para descargar con la selección actual.")
