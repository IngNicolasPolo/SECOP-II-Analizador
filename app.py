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

# Controles del reporte
col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    top_n = st.slider("Top de Entidades (N)", min_value=5, max_value=20, value=10, step=1)
with col_cfg2:
    usar_millones = st.checkbox("Mostrar valores en millones (M COP)", value=True)

titulo_reporte = "Reporte Visual SECOP"

def build_excel_report(
    df_filtered: pd.DataFrame,
    df_internal_full: pd.DataFrame,
    top_n: int = 10,
    usar_millones: bool = True,
    titulo: str = "Reporte Visual SECOP",
) -> bytes:
    """
    Genera un Excel con:
      - KPIs
      - Datos (tabla base)
      - Gráficos: Top N Entidades, Valor por Año, Distribución por Tipo
    Usa el dataset filtrado por la app (df_filtered) o, si está vacío, el interno (df_internal_full).
    Los valores pueden mostrarse en COP o en millones (M COP) según 'usar_millones'.
    """
    # Dataset base
    base = df_filtered.copy() if df_filtered is not None and not df_filtered.empty else df_internal_full.copy()

    # Renombrar a etiquetas amigables
    rename = {
        "entidad": "Entidad",
        "proveedor": "Proveedor",
        "tipo_contrato": "Tipo de Contrato",
        "valor": "Valor (COP)",
        "anio": "Año",
        "fecha": "Fecha",
        "departamento": "Departamento",
        "codigo_proceso": "ID Proceso",
    }
    base = base.rename(columns={c: rename[c] for c in rename if c in base.columns})

    # Detectar columnas clave
    col_ent = "Entidad" if "Entidad" in base.columns else None
    col_val = "Valor (COP)" if "Valor (COP)" in base.columns else None
    col_anio = "Año" if "Año" in base.columns else None
    col_tipo = "Tipo de Contrato" if "Tipo de Contrato" in base.columns else None
    col_prov = "Proveedor" if "Proveedor" in base.columns else None

    # Asegurar numérico en valor
    if col_val:
        base[col_val] = pd.to_numeric(base[col_val], errors="coerce").fillna(0)

    # Función segura de agregación
    def safe_group(df, by, val, n=None, sort_by_val_desc=True):
        if not by or not val or by not in df.columns or val not in df.columns:
            return pd.DataFrame(columns=[by or "Columna", val or "Valor"])
        g = df.groupby(by, dropna=True)[val].sum().reset_index()
        if sort_by_val_desc:
            g = g.sort_values(val, ascending=False)
        if n:
            g = g.head(n)
        return g

    top_ent = safe_group(base, col_ent, col_val, n=top_n)
    by_year = safe_group(base.dropna(subset=[col_anio]) if col_anio else base, col_anio, col_val, n=None, sort_by_val_desc=False) if col_anio else pd.DataFrame()
    if not by_year.empty and col_anio in by_year.columns:
        by_year = by_year.sort_values(col_anio)

    by_tipo = safe_group(base, col_tipo, col_val, n=None)

    # KPIs
    total_contratos = len(base)
    total_valor = float(base[col_val].sum()) if col_val in base else 0.0
    proveedores_unicos = int(base[col_prov].nunique()) if col_prov in base else 0

    # Unidades y formatos
    factor = 1_000_000 if usar_millones else 1
    sufijo = ' "M" "COP"' if usar_millones else ' "COP"'
    numfmt = f'#,##0{sufijo}'
    # Series auxiliares escaladas (no alteramos la hoja Datos)
    top_ent_scaled = top_ent.copy()
    by_year_scaled = by_year.copy()
    by_tipo_scaled = by_tipo.copy()
    if col_val:
        for df_ in (top_ent_scaled, by_year_scaled, by_tipo_scaled):
            if not df_.empty and col_val in df_.columns:
                df_[col_val] = (df_[col_val] / factor).round(2)

    # Construcción del Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book

        # Formatos
        fmt_title = wb.add_format({"bold": True, "font_size": 14})
        fmt_sub = wb.add_format({"bold": True, "font_size": 11})
        fmt_money = wb.add_format({"num_format": numfmt})
        fmt_int = wb.add_format({"num_format": '#,##0'})
        fmt_head = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
        fmt_cell = wb.add_format({"border": 1})
        fmt_wrap = wb.add_format({"text_wrap": True, "valign": "top"})

        # --- Hoja KPIs ---
        ws_kpi = wb.add_worksheet("KPIs")
        ws_kpi.write("A1", "Resumen de Indicadores", fmt_title)
        ws_kpi.write("A3", "Contratos", fmt_sub); ws_kpi.write_number("B3", total_contratos, fmt_int)
        ws_kpi.write("A4", f"Valor Total ({'M COP' if usar_millones else 'COP'})", fmt_sub)
        ws_kpi.write_number("B4", (total_valor / factor), fmt_money)
        ws_kpi.write("A5", "Proveedores Únicos", fmt_sub); ws_kpi.write_number("B5", proveedores_unicos, fmt_int)
        ws_kpi.set_column("A:A", 32); ws_kpi.set_column("B:B", 22)

        # --- Hoja Datos ---
        base.to_excel(writer, index=False, sheet_name="Datos")
        ws_tbl = writer.sheets["Datos"]

        # Encabezados con formato y anchos cómodos
        for j, col in enumerate(base.columns):
            ws_tbl.write(0, j, col, fmt_head)
            # ancho basado en el nombre + margen
            ws_tbl.set_column(j, j, max(14, min(len(str(col)) + 6, 46)))
        # Celdas con borde y wrap básico (hasta 5k filas para no pesar demasiado)
        max_fmt_rows = min(len(base), 5000)
        if max_fmt_rows > 0:
            ws_tbl.conditional_format(1, 0, max_fmt_rows, len(base.columns)-1,
                                      {"type": "no_errors", "format": fmt_cell})

        # --- Hoja Gráficos ---
        ws_cht = wb.add_worksheet("Gráficos")
        ws_cht.write("A1", titulo, fmt_title)

        # 1) Top N Entidades
        if not top_ent_scaled.empty and col_ent and col_val:
            ws_cht.write("A3", f"Top {top_n} Entidades por Valor ({'M COP' if usar_millones else 'COP'})", fmt_sub)
            ws_cht.write_row(4, 0, [col_ent, f"Valor ({'M COP' if usar_millones else 'COP'})"], fmt_head)
            for i, r in enumerate(top_ent_scaled.itertuples(index=False), start=5):
                ws_cht.write(i, 0, r[0], fmt_wrap)
                ws_cht.write_number(i, 1, float(r[1]), fmt_money)

            # Rango
            start = 6; end = 5 + len(top_ent_scaled)
            chart1 = wb.add_chart({"type": "bar"})
            chart1.add_series({
                "categories": f"=Gráficos!$A${start}:$A${end}",
                "values":     f"=Gráficos!$B${start}:$B${end}",
                "data_labels": {"value": True, "num_format": numfmt},
            })
            chart1.set_title({"name": f"Top {top_n} Entidades"})
            chart1.set_legend({"none": True})
            chart1.set_x_axis({"num_format": numfmt})
            chart1.set_size({"width": 820, "height": 380})
            ws_cht.insert_chart("D4", chart1)

        # 2) Valor por Año
        if not by_year_scaled.empty and col_anio and col_val:
            ws_cht.write("A20", f"Valor por Año ({'M COP' if usar_millones else 'COP'})", fmt_sub)
            ws_cht.write_row(21, 0, [col_anio, f"Valor ({'M COP' if usar_millones else 'COP'})"], fmt_head)
            for i, r in enumerate(by_year_scaled.itertuples(index=False), start=22):
                ws_cht.write(i, 0, int(r[0]))
                ws_cht.write_number(i, 1, float(r[1]), fmt_money)

            start = 23; end = 22 + len(by_year_scaled)
            chart2 = wb.add_chart({"type": "column"})
            chart2.add_series({
                "categories": f"=Gráficos!$A${start}:$A${end}",
                "values":     f"=Gráficos!$B${start}:$B${end}",
                "data_labels": {"value": True, "num_format": numfmt},
            })
            chart2.set_title({"name": "Valor por Año"})
            chart2.set_legend({"none": True})
            chart2.set_y_axis({"num_format": numfmt})
            chart2.set_size({"width": 820, "height": 380})
            ws_cht.insert_chart("D20", chart2)

        # 3) Distribución por Tipo de Contrato
        if not by_tipo_scaled.empty and col_tipo and col_val:
            ws_cht.write("A36", f"Distribución por Tipo de Contrato ({'M COP' if usar_millones else 'COP'})", fmt_sub)
            ws_cht.write_row(37, 0, [col_tipo, f"Valor ({'M COP' if usar_millones else 'COP'})"], fmt_head)
            for i, r in enumerate(by_tipo_scaled.itertuples(index=False), start=38):
                ws_cht.write(i, 0, r[0], fmt_wrap)
                ws_cht.write_number(i, 1, float(r[1]), fmt_money)

            start = 39; end = 38 + len(by_tipo_scaled)
            chart3 = wb.add_chart({"type": "pie"})
            chart3.add_series({
                "categories": f"=Gráficos!$A${start}:$A${end}",
                "values":     f"=Gráficos!$B${start}:$B${end}",
                "data_labels": {"percentage": True, "num_format": '0.0%'},
            })
            chart3.set_title({"name": "Participación por Tipo"})
            chart3.set_size({"width": 820, "height": 380})
            ws_cht.insert_chart("D36", chart3)

    return output.getvalue()

col_btn1, col_btn2 = st.columns([1, 2])
with col_btn1:
    generar = st.button("📊 Descargar Reporte con Gráficos (Excel)", use_container_width=True)

if generar:
    # Usa las columnas seleccionadas si existen; si no, usa el dataset filtrado interno
    base_interna = df.copy()
    datos_para_excel = df_view if not df_view.empty else None
    bytes_report = build_excel_report(
        datos_para_excel,
        base_interna,
        top_n=top_n,
        usar_millones=usar_millones,
        titulo=titulo_reporte
    )
    st.download_button(
        "📥 Guardar Reporte.xlsx",
        data=bytes_report,
        file_name="Reporte_SECOP_Visual.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )