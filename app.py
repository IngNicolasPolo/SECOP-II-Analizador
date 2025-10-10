# ---------- Imports ----------
import re
import unicodedata
from datetime import datetime
import io

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Integración Google Sheets (para Looker Studio) ---
import gspread
from google.oauth2.service_account import Credentials


# ---------- Config ----------
st.set_page_config(page_title="Analizador SECOP - COP", layout="wide")

# ---------- Parámetros Looker/Sheets ----------
SPREADSHEET_ID = "1wpvS2y8jBB4mKurj_EtMh6t5WZFrnAfT4lA_LbRNQQ"  # <<< AQUI: ID de tu hoja
SHEET_NAME = "Hoja 1"                                            # <<< AQUI: nombre de la pestaña

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
_CREDS = Credentials.from_service_account_file(
    "verdant-branch-474621-d7-f60501841517.json",  # nombre del .json en tu carpeta
    scopes=_SCOPES,
)
_GC = gspread.authorize(_CREDS)


# ---------- Funciones auxiliares ----------
def fmt_cop(valor):
    try:
        if pd.isna(valor):
            return "$0 COP"
        return f"${valor:,.0f} COP"
    except Exception:
        return "$0 COP"


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
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(x, format=fmt, errors="raise")
        except Exception:
            continue
    return pd.to_datetime(x, errors="coerce")


def to_number(x):
    if pd.isna(x):
        return 0.0
    s = str(x)
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
    "valor": [
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


# ---------- ETL Limpieza ----------
@st.cache_data(show_spinner=False)
def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = estandarizar_columnas(df)

    for c in ["entidad", "proveedor", "objeto", "tipo_contrato", "valor", "fecha",
              "departamento", "codigo_proceso", "nit_proveedor"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["entidad"] = df["entidad"].apply(normalizar_texto).str.upper()
    df["proveedor"] = df["proveedor"].apply(normalizar_texto).str.upper()
    df["objeto"] = df["objeto"].apply(normalizar_texto)
    df["tipo_contrato"] = df["tipo_contrato"].apply(normalizar_texto).str.upper()

    df["valor"] = df["valor"].apply(to_number)
    df["fecha"] = df["fecha"].apply(parse_fecha)

    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month

    subset = [c for c in ["codigo_proceso", "proveedor", "valor", "fecha"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")

    df = df[df["valor"] >= 0]
    df["valor"] = df["valor"].round(0)
    return df


# ---------- App ----------
st.title("🔎 Prototipo: ETL + Análisis básico de SECOP (COP)")
archivo = st.file_uploader("Sube un archivo SECOP (.csv / .xlsx)", type=["csv", "xlsx"])
if archivo is None:
    st.info("Sube un archivo para comenzar.")
    st.stop()


@st.cache_data(show_spinner=False)
def read_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, low_memory=False)
        except Exception:
            file.seek(0)
            return pd.read_csv(file, low_memory=False, encoding="latin-1")
    else:
        return pd.read_excel(file)


df_raw = read_any(archivo)
st.subheader("Vista previa (crudo)")
st.dataframe(df_raw.head(20), use_container_width=True)

# ---------- ETL ----------
df = limpiar(df_raw)

# ---------- Filtros (sidebar) ----------
with st.sidebar:
    st.header("Filtros")
    if "anio" in df.columns and df["anio"].notna().any():
        min_anio = int(df["anio"].min())
        max_anio = int(df["anio"].max())
        anio_rango = st.slider("Rango de años", min_value=min_anio, max_value=max_anio,
                               value=(min_anio, max_anio), step=1)
    else:
        anio_rango = None

    entidades = sorted([x for x in df["entidad"].dropna().unique()]) if "entidad" in df else []
    sel_all_ent = st.checkbox("Seleccionar todas las entidades", value=True)
    entidad_sel = st.multiselect("Entidad", entidades, default=entidades if sel_all_ent else entidades[:10])

    with st.expander("Filtros avanzados"):
        if "departamento" in df.columns:
            deps = sorted([x for x in df["departamento"].dropna().unique()])
            dep_sel = st.multiselect("Departamento", deps, default=[])
        else:
            dep_sel = []

        if "tipo_contrato" in df.columns:
            tipos = sorted([x for x in df["tipo_contrato"].dropna().unique()])
            tipo_sel = st.multiselect("Tipo de contrato", tipos, default=[])
        else:
            tipo_sel = []

        vmin, vmax = 0.0, float(df["valor"].max()) if "valor" in df else 0.0
        if vmax is None or pd.isna(vmax):
            vmax = 0.0
        rango_valor = st.slider("Rango de valores (COP)", min_value=float(vmin), max_value=float(vmax),
                                value=(float(vmin), float(vmax)), step=1.0)

        term_obj = st.text_input("Buscar en objeto (palabra clave)", value="").strip().lower()

# aplicar filtros
df_f = df.copy()
if anio_rango and "anio" in df_f.columns:
    a0, a1 = anio_rango
    df_f = df_f[(df_f["anio"] >= a0) & (df_f["anio"] <= a1)]
if entidad_sel:
    df_f = df_f[df_f["entidad"].isin(entidad_sel)]
if 'dep_sel' in locals() and dep_sel:
    df_f = df_f[df_f["departamento"].isin(dep_sel)]
if 'tipo_sel' in locals() and tipo_sel:
    df_f = df_f[df_f["tipo_contrato"].isin(tipo_sel)]
if "valor" in df_f.columns and rango_valor:
    v0, v1 = rango_valor
    df_f = df_f[(df_f["valor"] >= v0) & (df_f["valor"] <= v1)]
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

# ---------- Selector de columnas ----------
st.subheader("Datos limpios (muestra)")
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
available_internal = [c for c in friendly_cols.keys() if c in df_f.columns]
plantilla = st.radio("Plantilla de columnas", ["Básico", "Detallado", "Personalizado"], horizontal=True)
tpl_basico = [c for c in ["entidad", "proveedor", "valor", "fecha", "anio"] if c in available_internal]
tpl_detallado = [c for c in ["entidad", "nit_proveedor", "proveedor", "departamento",
                             "tipo_contrato", "valor", "fecha", "anio", "mes",
                             "codigo_proceso", "objeto"] if c in available_internal]
pre_sel = tpl_basico if plantilla == "Básico" else tpl_detallado
cols_sel = st.multiselect(
    "Elige las columnas a mostrar/descargar",
    options=available_internal, default=pre_sel,
    format_func=lambda c: friendly_cols.get(c, c)
)
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
    fig1 = px.line(g1, x="periodo", y="valor", markers=True,
                   title="Evolución del monto contratado (COP)",
                   labels={"periodo": "Periodo", "valor": "Pesos (COP)"})
    fig1.update_layout(yaxis_tickformat=",")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("No se encontraron columnas de fecha/valor para graficar la evolución.")

# ---------- Gráfico 2: Top 10 proveedores ----------
if "proveedor" in df_f.columns and "valor" in df_f.columns and not df_f.empty:
    top_prov = (df_f.groupby("proveedor", as_index=False)["valor"]
                .sum().sort_values("valor", ascending=False).head(10))
    fig2 = px.bar(top_prov, x="proveedor", y="valor",
                  title="Top 10 proveedores por monto (COP)",
                  labels={"proveedor": "Proveedor", "valor": "Pesos (COP)"})
    fig2.update_layout(xaxis=dict(tickangle=-45), yaxis_tickformat=",")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No se encontraron columnas de proveedor/valor.")

st.markdown("---")

# ---------- Descargas ----------
colA, colB = st.columns(2)
if not df_view.empty:
    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    colA.download_button("📥 Descargar (CSV)", data=csv_bytes,
                         file_name="secop_limpio.csv", mime="text/csv")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_view.to_excel(writer, index=False, sheet_name="DatosLimpios")
    colB.download_button("📥 Descargar (Excel)", data=buffer.getvalue(),
                         file_name="secop_limpio.xlsx",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("No hay datos para descargar con la selección actual.")

# ---------- Publicar en Google Sheets (para Looker) ----------
st.markdown("---")
st.subheader("📡 Publicar datos al dashboard (Looker Studio)")

target_df = df_view if not df_view.empty else df_f

def publicar_df_en_sheets(df):
    if df is None or df.empty:
        st.warning("No hay datos para publicar.")
        return False
    df2 = df.copy()
    # si viene con nombres 'bonitos', mapea de nuevo a internos para consistencia opcional
    # (puedes omitir esta parte si tu Looker trabaja con los nombres bonitos)
    if "Fecha del Proceso" in df2.columns:
        df2 = df2.rename(columns={
            "Entidad": "entidad",
            "NIT Proveedor": "nit_proveedor",
            "Proveedor": "proveedor",
            "Departamento": "departamento",
            "Tipo de Contrato": "tipo_contrato",
            "Valor (COP)": "valor",
            "Fecha del Proceso": "fecha",
            "Año": "anio",
            "Mes": "mes",
            "ID / Código del Proceso": "codigo_proceso",
            "Objeto / Descripción": "objeto",
        })
    if "fecha" in df2.columns:
        df2["fecha"] = pd.to_datetime(df2["fecha"], errors="coerce").dt.strftime("%Y-%m-%d")

    try:
        sh = _GC.open_by_key(SPREADSHEET_ID)
        try:
            ws = sh.worksheet(SHEET_NAME)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=SHEET_NAME, rows=100, cols=26)
        ws.clear()

        rows = [df2.columns.tolist()] + df2.fillna("").astype(object).values.tolist()
        ws.update("A1", [rows[0]])
        chunk = 2000
        for i in range(1, len(rows), chunk):
            ws.update(f"A{i+1}", rows[i:i+chunk], value_input_option="RAW")
        return True
    except Exception as e:
        st.error(f"Error publicando en Google Sheets: {e}")
        return False

if st.button("Actualizar dashboard con estos datos"):
    ok = publicar_df_en_sheets(target_df)
    if ok:
        st.success("✅ Datos publicados en Google Sheets. Refresca tu reporte de Looker Studio.")
    else:
        st.error("❌ No se pudieron publicar los datos.")

# (Opcional) embeber Looker cuando tengas el iframe:
# st.markdown("## 📊 Dashboard interactivo (Looker Studio)")
# looker_url = "https://lookerstudio.google.com/embed/reporting/TU_ENLACE_AQUI"
# st.markdown(f'<iframe width="100%" height="850" src="{looker_url}" frameborder="0" style="border:0" allowfullscreen></iframe>',
#             unsafe_allow_html=True)