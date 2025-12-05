# ---------- Imports ----------
import re
import unicodedata
import io
import pandas as pd
import streamlit as st
# >>> NEW: para codificar el parámetro en la URL
from urllib.parse import quote

# ---------- Config ----------
st.set_page_config(page_title="Analizador SECOP - COP", layout="wide")

# >>> NEW: URL base de tu reporte de Looker Studio (modo Ver)
LOOKER_BASE_URL = "https://lookerstudio.google.com/u/1/reporting/bc878e05-ec76-4918-ace9-d32bb296b45a/page/wh5cF"

# ---------- Estilos globales ----------
STYLES = """
<style>
:root {
  --bg:#0e1117; --panel:#161a23; --card:#1b2030;
  --text:#e5e7eb; --muted:#9ca3af; --accent:#22c55e; --accent2:#3b82f6;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); }
header[data-testid="stHeader"] { background: transparent; }

/* Header bar */
.header-bar { position: sticky; top: 0; z-index: 999;
  background: linear-gradient(180deg, rgba(14,17,23,.95) 0%, rgba(14,17,23,.85) 100%);
  backdrop-filter: blur(6px); border-bottom: 1px solid rgba(255,255,255,0.06);
  padding: 10px 16px; margin-bottom: 8px;
}
.header-flex { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.brand { display:flex; align-items:center; gap:12px; }
.brand .logo { width:34px; height:34px; border-radius:10px; background:#111827;
  display:grid; place-items:center; font-size:18px; }
.brand .title { font-weight:700; font-size:18px; color:var(--text); }
.brand .subtitle { color:var(--muted); font-size:12px; margin-top:-2px; }

/* KPI cards */
.stat-grid { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:16px; margin: 8px 0 6px; }
.stat-card { background: var(--card); border:1px solid rgba(255,255,255,0.06);
  border-radius:14px; padding:14px 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.25); }
.stat-top { display:flex; align-items:center; justify-content:space-between; }
.stat-ico { font-size:20px; opacity:.9 }
.stat-label { color:var(--muted); font-size:12px; letter-spacing:.3px; }
.stat-value { font-size:22px; font-weight:800; color:var(--text); margin-top:4px; }

/* Sidebar / Popovers / Expander */
div[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid rgba(255,255,255,0.06); }
details[data-testid="stExpander"] > summary { background: var(--panel); border:1px solid rgba(255,255,255,0.06);
  border-radius:10px; padding:8px 12px; }
div[data-testid="stPopover"] > div { background: var(--panel) !important; border:1px solid rgba(255,255,255,0.08);
  border-radius:12px; box-shadow: 0 12px 30px rgba(0,0,0,.35); }

/* DataFrame */
div[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid rgba(255,255,255,0.06); }
div[data-testid="stDataFrame"] thead tr th { background:#121723; color:#e5e7eb; font-weight:700; }

/* Botones */
div[data-testid="stDownloadButton"] > button,
div[data-testid="stButton"] > button {
  background: linear-gradient(180deg, var(--accent2), #2563eb);
  color:white; border:0; border-radius:12px; padding:10px 14px; font-weight:700;
}
div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stButton"] > button:hover {
  filter:brightness(1.05); transform: translateY(-1px); transition: all .15s ease;
}

/* Footer */
.footer { margin-top: 24px; padding: 10px 0 30px; color: var(--muted);
  border-top:1px solid rgba(255,255,255,.06); text-align:center; font-size:12px;}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ---------- Utilidades ----------
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
    s = str(x).replace(".", "").replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0

# >>> NEW: helper para armar el URL con p_entidad
def build_looker_url(entidad_text: str) -> str:
    if not entidad_text:
        return LOOKER_BASE_URL
    # pasamos a minúsculas y codificamos espacios/acentos
    param = f"p_entidad:{entidad_text.strip().lower()}"
    return f"{LOOKER_BASE_URL}?params=" + quote(param, safe=":,")

# ---------- Mapeo flexible ----------
COLUMN_MAP = {
    "entidad": ["Entidad","entidad","nombre_entidad","entidad contratante","nombre de la entidad","entidad_estatal","buyer","buyername"],
    "proveedor": ["Nombre del Proveedor Adjudicado","Proveedor","Contratista","nombre_proveedor","supplier","nombre del contratista"],
    "nit_proveedor": ["NIT del Proveedor Adjudicado","nit proveedor","supplier_id","nit","identificacion proveedor"],
    "objeto": ["Descripción del Procedimiento","Objeto","objeto del contrato","descripcion","description"],
    "tipo_contrato": ["Tipo de Contrato","tipo de contrato","modalidad","tipocontrato","modalidad de contratacion","contracttype"],
    "valor": ["Valor Total Adjudicacion","Valor del contrato","valor","valor total","monto","amount","valor adjudicado","valor_contrato","Precio Base"],
    "fecha": ["Fecha Adjudicacion","Fecha de Publicacion del Proceso","Fecha de Ultima Publicación","fecha","publicationdate","awarddate","fecha publicacion"],
    "departamento": ["Departamento Entidad","departamento","region","ubicacion","departamento_ejecucion"],
    "codigo_proceso": ["ID del Proceso","codigo de proceso","proceso","processid","id_proceso","id proceso","codigo_proceso"],
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

# ---------- Limpieza principal ----------
@st.cache_data(show_spinner=False)
def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = estandarizar_columnas(df)

    for c in ["entidad","proveedor","objeto","tipo_contrato","valor","fecha","departamento","codigo_proceso","nit_proveedor"]:
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

    subset = [c for c in ["codigo_proceso","proveedor","valor","fecha"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")

    df = df[df["valor"] >= 0]
    df["valor"] = df["valor"].round(0)
    return df

# ---------- App ----------
# ---------- Header ----------
st.markdown("""
<div class="header-bar">
  <div class="header-flex">
    <div class="brand">
      <div class="logo">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="#1e3a8a" width="22" height="22">
          <circle cx="11" cy="11" r="7" stroke="#1e3a8a" stroke-width="2" fill="none"></circle>
          <line x1="16.65" y1="16.65" x2="22" y2="22" stroke="#1e3a8a" stroke-width="2" stroke-linecap="round"></line>
        </svg>
      </div>
      <div>
        <div class="title">Analizador SECOP I y II — COP</div>
        <div class="subtitle">Sistema de análisis exploratorio de contratación pública</div>
      </div>
    </div>
    <div class="subtitle" style="font-weight:1000;color:#1e3a8a;">v1.0 • Proyecto de Grado</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Carga de archivo
archivo = st.file_uploader("📂 Sube un archivo SECOP (.csv / .xlsx)", type=["csv","xlsx"])
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

st.subheader("📊 Vista previa de los datos sin limpiar")
st.dataframe(df_raw.head(20), use_container_width=True)

# ---------- ETL ----------
df = limpiar(df_raw)


# ---------- Filtros (sidebar) ----------
with st.sidebar:
    st.header("Filtros")

    # Rango de años (si existe la columna)
    if "anio" in df.columns and df["anio"].notna().any():
        min_anio = int(df["anio"].min())
        max_anio = int(df["anio"].max())
        anio_rango = st.slider(
            "Rango de años",
            min_value=min_anio,
            max_value=max_anio,
            value=(min_anio, max_anio),
            step=1
        )
    else:
        anio_rango = None

    # Búsqueda por palabra clave en el objeto
    term_obj = st.text_input(
        "Buscar en objeto (palabra clave)",
        value=""
    ).strip().lower()

# ---------- APLICAR FILTROS ----------
df_f = df.copy()

# Año
if anio_rango and "anio" in df_f.columns:
    a0, a1 = anio_rango
    df_f = df_f[(df_f["anio"] >= a0) & (df_f["anio"] <= a1)]

# Texto en objeto
if term_obj and "objeto" in df_f.columns:
    df_f = df_f[df_f["objeto"].str.contains(term_obj, na=False)]

# Chequeo visual
st.caption(f"Filas filtradas: {len(df_f):,} de {len(df):,}".replace(",", "."))

# ---------- KPIs: cálculos ----------
total_contratos = len(df_f)
total_valor = float(df_f["valor"].sum()) if "valor" in df_f else 0.0
proveedores_unicos = int(df_f["proveedor"].nunique()) if "proveedor" in df_f else 0

# ---------- KPIs (cards) ----------
contratos_fmt = f"{total_contratos:,}".replace(",", ".")
proveedores_fmt = f"{proveedores_unicos:,}".replace(",", ".")
valor_fmt = fmt_cop(total_valor)

st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-top">
      <div class="stat-label">Contratos</div>
      <div class="stat-ico">🧾</div>
    </div>
    <div class="stat-value">{contratos_fmt}</div>
  </div>
  <div class="stat-card">
    <div class="stat-top">
      <div class="stat-label">Valor total (COP)</div>
      <div class="stat-ico">💰</div>
    </div>
    <div class="stat-value">{valor_fmt}</div>
  </div>
  <div class="stat-card">
    <div class="stat-top">
      <div class="stat-label">Proveedores únicos</div>
      <div class="stat-ico">👥</div>
    </div>
    <div class="stat-value">{proveedores_fmt}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------- Tabla configurable ----------
st.subheader("🧾 Datos limpios (selecciona columnas)")

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

# Plantillas
tpl_basico     = [c for c in ["entidad", "proveedor", "valor", "fecha", "anio"] if c in available_internal]
tpl_detallado  = [c for c in ["entidad","nit_proveedor","proveedor","departamento","tipo_contrato","valor","fecha","anio","mes","codigo_proceso","objeto"] if c in available_internal]

plantilla = st.radio("Plantilla de columnas", ["Básico", "Detallado", "Personalizado"], horizontal=True)

# Si eligen plantilla (≠ personalizado), precargamos columnas
if plantilla == "Básico":
    default_cols = tpl_basico
elif plantilla == "Detallado":
    default_cols = tpl_detallado
else:  # Personalizado
    default_cols = []

with st.popover("Elige las columnas a mostrar/descargar", use_container_width=True):
    cols_sel = st.multiselect(
        "Columnas",
        options=available_internal,
        default=default_cols,  # <— aquí la clave: precargamos si NO es personalizado
        format_func=lambda c: friendly_cols.get(c, c),
        placeholder="Selecciona columnas…"
    )

# Si el usuario no toca nada y está en plantilla, usa la plantilla
if not cols_sel and plantilla != "Personalizado":
    cols_sel = default_cols

st.caption(f"{len(cols_sel)} columnas seleccionadas")

# Vista + descargas
if cols_sel:
    df_view = df_f[cols_sel].rename(columns=friendly_cols).copy()
    st.dataframe(df_view.head(20), use_container_width=True)

    st.markdown("---")
    colA, colB = st.columns(2)
    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    colA.download_button("📥 Descargar Datos Limpios (CSV)", data=csv_bytes, file_name="secop_filtrado.csv", mime="text/csv")

    buffer_xlsx = io.BytesIO()
    with pd.ExcelWriter(buffer_xlsx, engine="xlsxwriter") as writer:
        df_view.to_excel(writer, index=False, sheet_name="DatosFiltrados")
    colB.download_button(
        "📥 Descargar Datos Limpios (Excel)",
        data=buffer_xlsx.getvalue(),
        file_name="secop_filtrado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Selecciona columnas (o una plantilla) para ver y descargar los datos.")
    df_view = pd.DataFrame()

st.markdown("---")
st.subheader("📈 Reporte interactivo (Looker Studio)")

# Enlace directo, sin depender de entidades
if getattr(st, "link_button", None):
    st.link_button("🔗 Abrir reporte interactivo en Looker Studio", LOOKER_BASE_URL, help="Se abrirá en una nueva pestaña")
else:
    st.markdown(f'<a href="{LOOKER_BASE_URL}" target="_blank"><button>🔗 Abrir reporte interactivo en Looker Studio</button></a>', unsafe_allow_html=True)

# ---------- Footer (siempre visible) ----------
st.markdown(
    '<div class="footer">Desarrollado por <b>Nicolás Polo</b> — Ingeniería de Sistemas (2025)</div>',
    unsafe_allow_html=True
)
