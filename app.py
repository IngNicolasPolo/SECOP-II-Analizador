# ---------- Imports ----------
import re
import unicodedata
from datetime import datetime
import io
import base64
import json
import pandas as pd
import streamlit as st
import requests

# ---------- Config ----------
st.set_page_config(page_title="Analizador SECOP - COP", layout="wide")

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
    s = str(x)
    s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0

# ---------- Mapeo flexible ----------
COLUMN_MAP = {
    "entidad": [
        "Entidad","entidad","nombre_entidad","entidad contratante","nombre de la entidad",
        "entidad_estatal","buyer","buyername"
    ],
    "proveedor": [
        "Nombre del Proveedor Adjudicado","Proveedor","Contratista","nombre_proveedor",
        "supplier","nombre del contratista"
    ],
    "nit_proveedor": [
        "NIT del Proveedor Adjudicado","nit proveedor","supplier_id","nit","identificacion proveedor"
    ],
    "objeto": [
        "Descripción del Procedimiento","Objeto","objeto del contrato","descripcion","description"
    ],
    "tipo_contrato": [
        "Tipo de Contrato","tipo de contrato","modalidad","tipocontrato",
        "modalidad de contratacion","contracttype"
    ],
    "valor": [
        "Valor Total Adjudicacion","Valor del contrato","valor","valor total",
        "monto","amount","valor adjudicado","valor_contrato","Precio Base"
    ],
    "fecha": [
        "Fecha Adjudicacion","Fecha de Publicacion del Proceso","Fecha de Ultima Publicación",
        "fecha","publicationdate","awarddate","fecha publicacion"
    ],
    "departamento": [
        "Departamento Entidad","departamento","region","ubicacion","departamento_ejecucion"
    ],
    "codigo_proceso": [
        "ID del Proceso","codigo de proceso","proceso","processid",
        "id_proceso","id proceso","codigo_proceso"
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

# ---------- Limpieza principal ----------
@st.cache_data(show_spinner=False)
def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = estandarizar_columnas(df)

    for c in ["entidad","proveedor","objeto","tipo_contrato","valor","fecha",
              "departamento","codigo_proceso","nit_proveedor"]:
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
st.title("🔎 Análisis de Contratación Pública - SECOP I y II")

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
plantilla = st.radio("Plantilla de columnas", ["Básico", "Detallado", "Personalizado"], horizontal=True)
tpl_basico = [c for c in ["entidad", "proveedor", "valor", "fecha", "anio"] if c in available_internal]
tpl_detallado = [c for c in ["entidad","nit_proveedor","proveedor","departamento",
                             "tipo_contrato","valor","fecha","anio","mes",
                             "codigo_proceso","objeto"] if c in available_internal]
pre_sel = tpl_basico if plantilla == "Básico" else tpl_detallado
cols_sel = st.multiselect(
    "Elige las columnas a mostrar/descargar",
    options=available_internal, default=pre_sel,
    format_func=lambda c: friendly_cols.get(c, c)
)
if cols_sel:
    df_view = df_f[cols_sel].rename(columns=friendly_cols).copy()
else:
    st.info("Selecciona al menos una columna para ver la tabla.")
    df_view = pd.DataFrame()

st.dataframe(df_view.head(20), use_container_width=True)

# ---------- Descargas locales ----------
st.markdown("---")
colA, colB = st.columns(2)
if not df_view.empty:
    csv_bytes = df_view.to_csv(index=False).encode("utf-8")
    colA.download_button("📥 Descargar (CSV)", data=csv_bytes,
                         file_name="secop_filtrado.csv", mime="text/csv")

    buffer_xlsx = io.BytesIO()
    with pd.ExcelWriter(buffer_xlsx, engine="xlsxwriter") as writer:
        df_view.to_excel(writer, index=False, sheet_name="DatosFiltrados")
    colB.download_button("📥 Descargar (Excel)",
                         data=buffer_xlsx.getvalue(),
                         file_name="secop_filtrado.xlsx",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------- NUEVO: Reporte con gráficos (Excel) ----------
st.markdown("### 📈 Reporte visual listo para entregar (Excel con gráficos)")

def build_excel_report(df_filtered: pd.DataFrame, df_internal_full: pd.DataFrame) -> bytes:
    # Si el usuario no seleccionó columnas, usa el dataset interno
    base = df_filtered.copy() if df_filtered is not None and not df_filtered.empty else df_internal_full.copy()

    # Asegurar nombres amigables
    rename = {
        "entidad": "Entidad",
        "proveedor": "Proveedor",
        "tipo_contrato": "Tipo de Contrato",
        "valor": "Valor (COP)",
        "anio": "Año"
    }
    base = base.rename(columns=rename)

    # Validar existencia de columnas clave
    col_ent = next((c for c in base.columns if c.lower() == "entidad"), None)
    col_val = next((c for c in base.columns if "valor" in c.lower()), None)
    col_anio = next((c for c in base.columns if "año" in c.lower() or "anio" in c.lower()), None)
    col_tipo = next((c for c in base.columns if "tipo" in c.lower()), None)
    col_prov = next((c for c in base.columns if "proveedor" in c.lower()), None)

    # Si falta algo crítico, crear columnas vacías
    for c in [col_ent, col_val, col_anio, col_tipo, col_prov]:
        if c is None:
            continue
        if c not in base.columns:
            base[c] = pd.NA

    # Agregaciones seguras
    def safe_group(df, by, val):
        if by is None or val is None or by not in df.columns or val not in df.columns:
            return pd.DataFrame(columns=[by or "Columna", val or "Valor"])
        return (df.groupby(by, dropna=True)[val]
                .sum().sort_values(ascending=False)
                .reset_index())

    top_ent = safe_group(base, col_ent, col_val).head(10)
    by_year = safe_group(base, col_anio, col_val).sort_values(col_anio) if col_anio else pd.DataFrame()
    by_tipo = safe_group(base, col_tipo, col_val)

    total_contratos = len(base)
    total_valor = float(base[col_val].sum()) if col_val in base else 0.0
    proveedores_unicos = base[col_prov].nunique() if col_prov in base else 0

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        fmt_title = wb.add_format({"bold": True, "font_size": 14})
        fmt_sub = wb.add_format({"bold": True, "font_size": 11})
        fmt_money = wb.add_format({"num_format": '#,##0" COP"'})
        fmt_int = wb.add_format({"num_format": '#,##0'})

        # KPIs
        ws_kpi = wb.add_worksheet("KPIs")
        ws_kpi.write("A1", "Resumen de Indicadores", fmt_title)
        ws_kpi.write("A3", "Contratos")
        ws_kpi.write_number("B3", total_contratos, fmt_int)
        ws_kpi.write("A4", "Valor Total (COP)")
        ws_kpi.write_number("B4", total_valor, fmt_money)
        ws_kpi.write("A5", "Proveedores Únicos")
        ws_kpi.write_number("B5", proveedores_unicos, fmt_int)
        ws_kpi.set_column("A:B", 25)

        # Tabla principal
        base.to_excel(writer, index=False, sheet_name="Datos")
        ws_tbl = writer.sheets["Datos"]
        for idx, col in enumerate(base.columns):
            ws_tbl.set_column(idx, idx, min(max(len(str(col)) + 2, 12), 45))

        # Gráficos
        ws_cht = wb.add_worksheet("Gráficos")
        ws_cht.write("A1", "Reporte Visual SECOP", fmt_title)

        # Top 10 Entidades
        if not top_ent.empty:
            ws_cht.write("A3", "Top 10 Entidades por Valor (COP)", fmt_sub)
            ws_cht.write_row(4, 0, top_ent.columns, fmt_sub)
            for i, r in enumerate(top_ent.itertuples(index=False), start=5):
                ws_cht.write(i, 0, r[0])
                ws_cht.write_number(i, 1, float(r[1]), fmt_money)

            chart1 = wb.add_chart({"type": "bar"})
            chart1.add_series({
                "categories": f"=Gráficos!$A$6:$A${5 + len(top_ent)}",
                "values": f"=Gráficos!$B$6:$B${5 + len(top_ent)}",
                "data_labels": {"value": True},
            })
            chart1.set_title({"name": "Top 10 Entidades"})
            ws_cht.insert_chart("D4", chart1)

        # Valor por Año
        if not by_year.empty:
            ws_cht.write("A20", "Valor por Año", fmt_sub)
            ws_cht.write_row(21, 0, by_year.columns, fmt_sub)
            for i, r in enumerate(by_year.itertuples(index=False), start=22):
                ws_cht.write(i, 0, int(r[0]))
                ws_cht.write_number(i, 1, float(r[1]), fmt_money)

            chart2 = wb.add_chart({"type": "column"})
            chart2.add_series({
                "categories": f"=Gráficos!$A$23:$A${22 + len(by_year)}",
                "values": f"=Gráficos!$B$23:$B${22 + len(by_year)}",
                "data_labels": {"value": True},
            })
            chart2.set_title({"name": "Valor por Año"})
            ws_cht.insert_chart("D20", chart2)

        # Distribución por Tipo de Contrato
        if not by_tipo.empty:
            ws_cht.write("A36", "Distribución por Tipo de Contrato", fmt_sub)
            ws_cht.write_row(37, 0, by_tipo.columns, fmt_sub)
            for i, r in enumerate(by_tipo.itertuples(index=False), start=38):
                ws_cht.write(i, 0, r[0])
                ws_cht.write_number(i, 1, float(r[1]), fmt_money)

            chart3 = wb.add_chart({"type": "pie"})
            chart3.add_series({
                "categories": f"=Gráficos!$A$39:$A${38 + len(by_tipo)}",
                "values": f"=Gráficos!$B$39:$B${38 + len(by_tipo)}",
                "data_labels": {"percentage": True, "category": True},
            })
            chart3.set_title({"name": "Por Tipo de Contrato"})
            ws_cht.insert_chart("D36", chart3)

    return output.getvalue()

col_btn1, col_btn2 = st.columns([1, 2])
with col_btn1:
    generar = st.button("📊 Descargar Reporte con Gráficos (Excel)", use_container_width=True)

if generar:
    base_interna = df.copy()
    bytes_report = build_excel_report(df_view if not df_view.empty else None, base_interna)
    st.download_button(
        "📥 Guardar Reporte.xlsx",
        data=bytes_report,
        file_name="Reporte_SECOP_Visual.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
