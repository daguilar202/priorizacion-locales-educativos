# -*- coding: utf-8 -*-
"""
Aplicativo DRE/UGEL para priorización regional de infraestructura educativa.

Ejecución:
    pip install -r requirements_priorizacion.txt
    streamlit run app_priorizacion_regional.py
"""
from __future__ import annotations

import io
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from criterios_priorizacion import calculate_safety, total_score, validate_criterion_weights

ROOT = Path(__file__).resolve().parent
DATA_FILE_GZ = ROOT / "data" / "locales_priorizacion.csv.gz"
DATA_FILE_CSV = ROOT / "data" / "locales_priorizacion.csv"
CONFIG_FILE = ROOT / "config_priorizacion.json"
TEMPLATE_FILE = ROOT / "plantilla_seguridad.csv"

st.set_page_config(page_title="Priorización regional de infraestructura educativa", page_icon="🏫", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
.hero {background:linear-gradient(115deg,#173f35,#246b58);color:white;padding:1.25rem 1.45rem;border-radius:16px;margin-bottom:1rem;}
.hero h1{margin:0;font-size:1.9rem}.hero p{margin:.35rem 0 0;opacity:.95}
.card {border:1px solid #dfe7e3;border-radius:12px;padding:.85rem 1rem;background:white;margin-bottom:.7rem;}
.demo {border-left:5px solid #c17b00;background:#fff8e8;padding:.75rem 1rem;border-radius:8px;margin:.5rem 0 1rem;}
.info {border-left:5px solid #27735f;background:#eef8f4;padding:.75rem 1rem;border-radius:8px;margin:.5rem 0 1rem;}
.critical {border-left:5px solid #b42318;background:#fff0ee;padding:.75rem 1rem;border-radius:8px;margin:.5rem 0 1rem;}
.small {font-size:.86rem;color:#5e6c66}
[data-testid="stMetric"]{border:1px solid #e1e7e4;border-radius:12px;padding:.7rem;background:white}
</style>
""", unsafe_allow_html=True)


def norm(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.casefold()).strip()


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Cargando locales educativos...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"cod_local": str, "ubigeo": str}, low_memory=False)
    for c in ["cod_local", "ubigeo", "nombre_local", "region", "provincia", "distrito", "centro_poblado",
              "dre", "ugel", "area_censal", "busqueda"]:
        if c in df:
            df[c] = df[c].fillna("").astype(str)
    if "busqueda" not in df:
        df["busqueda"] = (df["cod_local"] + " " + df["nombre_local"] + " " + df["distrito"]).map(norm)
    return df


def data_from_upload(upload) -> pd.DataFrame:
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload, dtype={"cod_local": str}, low_memory=False)
    return pd.read_excel(upload, dtype={"cod_local": str})


def safety_records_df(records: dict[str, dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for code, rec in records.items():
        calc = calculate_safety(rec, cfg)
        rows.append({"cod_local": str(code), **rec, **calc})
    return pd.DataFrame(rows)


def merge_scores(base: pd.DataFrame, records: dict[str, dict], cfg: dict, weights: dict[str, float]) -> pd.DataFrame:
    df = base.copy()
    if records:
        s = safety_records_df(records, cfg)
        keep = [c for c in ["cod_local", "s1", "s2", "s3", "s4", "s5", "rd", "ob", "seguridad_score",
                               "seguridad_completa", "seguridad_critica"] if c in s]
        df = df.drop(columns=[c for c in keep if c != "cod_local" and c in df.columns], errors="ignore")
        df = df.merge(s[keep], on="cod_local", how="left")
    else:
        df["seguridad_score"] = pd.NA
        df["seguridad_completa"] = False
        df["seguridad_critica"] = False
    df["seguridad_completa"] = df["seguridad_completa"].fillna(False).astype(bool)
    df["seguridad_critica"] = df["seguridad_critica"].fillna(False).astype(bool)
    df["puntaje_total"] = df.apply(lambda r: total_score(r.to_dict(), weights), axis=1)
    return df


def candidate_table(df: pd.DataFrame, query: str, n: int = 30) -> pd.DataFrame:
    q = norm(query)
    if len(q) < 2:
        return df.iloc[0:0]
    mask = df["busqueda"].str.contains(re.escape(q), regex=True, na=False)
    out = df.loc[mask, ["cod_local", "nombre_local", "region", "provincia", "distrito", "ugel"]].head(n).copy()
    return out


def current_weights(cfg: dict) -> dict[str, float]:
    out = {}
    st.sidebar.markdown("### Pesos por criterio")
    for key, d in cfg["criterion_weights"].items():
        state_key = f"weight_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = float(d["default"])
        out[key] = st.sidebar.number_input(
            d["label"], min_value=float(d["min"]), max_value=float(d["max"]),
            step=1.0, key=state_key,
            help=f"Rango permitido por configuración: {d['min']}%–{d['max']}%"
        )
    ok, message = validate_criterion_weights(out, cfg)
    if ok:
        st.sidebar.success("Pesos válidos · total 100%")
    else:
        st.sidebar.error(message)
    return out


def blank_record() -> dict:
    return {
        "area_demoler_m2": None,
        "area_techada_m2": None,
        "material_sistema": "",
        "norma_sismica_categoria": "",
        "estado_conservacion": "",
        "zona_sismica": "",
        "mm_susceptibilidad": "",
        "mm_ocurrencia": "",
        "inundacion_peligro": "",
        "inundacion_recurrencia_norm": None,
        "heladas_friaje": "",
        "fuente": "",
        "fecha_actualizacion": str(date.today()),
        "observaciones": "",
    }


def render_local_header(row: pd.Series):
    st.markdown(
        f"<div class='card'><b>{row['cod_local']} · {row['nombre_local']}</b><br>"
        f"{row['region']} / {row['provincia']} / {row['distrito']} · {row['ugel']}<br>"
        f"<span class='small'>{row['centro_poblado']} · {row['area_censal']} · matrícula: {row.get('matricula','')}</span></div>",
        unsafe_allow_html=True,
    )


cfg = load_config()
if "security_records" not in st.session_state:
    st.session_state.security_records = {}
if "selected_code" not in st.session_state:
    st.session_state.selected_code = ""

st.markdown("""
<div class="hero">
<h1>Priorización regional de infraestructura educativa</h1>
<p>Herramienta de apoyo para DRE y UGEL: precarga criterios disponibles, completa Seguridad y genera rankings regionales, provinciales y distritales.</p>
</div>
""", unsafe_allow_html=True)

if DATA_FILE_GZ.exists():
    base = load_data(str(DATA_FILE_GZ))
elif DATA_FILE_CSV.exists():
    base = load_data(str(DATA_FILE_CSV))
else:
    st.error("No se encontró data/locales_priorizacion.csv.gz ni data/locales_priorizacion.csv. Ejecute primero preparar_base_priorizacion.py o cargue una base preparada.")
    uploaded_base = st.file_uploader("Cargar base preparada CSV", type=["csv"])
    if not uploaded_base:
        st.stop()
    base = pd.read_csv(uploaded_base, dtype={"cod_local": str}, low_memory=False)

weights = current_weights(cfg)
weights_ok, weights_msg = validate_criterion_weights(weights, cfg)

st.sidebar.markdown("### Avance de Seguridad")
st.sidebar.metric("Locales registrados", len(st.session_state.security_records))
if st.session_state.security_records:
    export_records = safety_records_df(st.session_state.security_records, cfg)
    st.sidebar.download_button(
        "Descargar seguridad registrada", export_records.to_csv(index=False).encode("utf-8-sig"),
        "seguridad_registrada.csv", "text/csv", use_container_width=True
    )

TAB_HOME, TAB_LOCAL, TAB_SECURITY, TAB_RANK, TAB_METHOD = st.tabs(
    ["Inicio", "Buscar local", "Seguridad", "Ranking", "Metodología"]
)

with TAB_HOME:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Locales precargados", f"{len(base):,}")
    c2.metric("Regiones", base["region"].nunique())
    c3.metric("Con Seguridad cargada", len(st.session_state.security_records))
    c4.metric("Pesos", "Válidos" if weights_ok else "Revisar")
    st.markdown("""
    <div class="info"><b>Flujo propuesto:</b> 1) la DIPLAN/DIGEIE prepara la base precargada; 2) la DRE/UGEL busca cada local por código o nombre; 3) completa o carga en bloque Seguridad; 4) define los pesos de los cuatro criterios dentro de rangos permitidos; 5) genera y exporta el ranking del ámbito deseado.</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="demo"><b>Prototipo:</b> Equidad 2, Equidad 4 y Territorio 1–6 contienen valores DEMO reproducibles hasta incorporar sus fuentes oficiales. Los puntajes y pesos técnicos se encuentran separados de la interfaz para poder reemplazarlos sin rehacer el aplicativo.</div>
    """, unsafe_allow_html=True)
    st.write("**Criterios de eficiencia implementados:** Eficiencia 1a para locales urbanos y Eficiencia 2 para locales rurales. Se excluye Eficiencia 1 (utilización de infraestructura).")

with TAB_LOCAL:
    st.subheader("Buscar local educativo")
    query = st.text_input("Buscar por código de local, nombre del servicio, centro poblado o distrito", placeholder="Ej.: 175 o José María Arguedas")
    candidates = candidate_table(base, query)
    if len(query.strip()) < 2:
        st.caption("Escriba al menos 2 caracteres. No se carga una lista completa de 55 mil locales.")
    elif candidates.empty:
        st.warning("No se encontraron coincidencias en los primeros criterios de búsqueda.")
    else:
        labels = {
            r.cod_local: f"{r.cod_local} · {r.nombre_local} · {r.distrito} ({r.region})"
            for r in candidates.itertuples()
        }
        code = st.selectbox("Coincidencias", candidates["cod_local"].tolist(), format_func=lambda x: labels.get(x, x))
        if st.button("Seleccionar local", type="primary"):
            st.session_state.selected_code = code
            st.success(f"Local {code} seleccionado. Puede ir a la pestaña Seguridad.")
        st.dataframe(candidates, use_container_width=True, hide_index=True)

    selected = st.session_state.selected_code
    if selected:
        match = base.loc[base["cod_local"] == selected]
        if not match.empty:
            row = match.iloc[0]
            render_local_header(row)
            a, b, c, d = st.columns(4)
            a.metric("Pobreza", f"{row.get('pobreza_pct', float('nan')):.1f}%" if pd.notna(row.get('pobreza_pct')) else "s/i")
            b.metric("Matrícula", f"{row.get('matricula', 0):,.0f}" if pd.notna(row.get('matricula')) else "s/i")
            c.metric("Brecha estimada", f"S/ {row.get('brecha_soles', 0):,.0f}" if pd.notna(row.get('brecha_soles')) else "s/i")
            d.metric("Eficiencia", f"{row.get('eficiencia_1a_raw', float('nan')):.1f}" if pd.notna(row.get('eficiencia_1a_raw')) else "s/i")
            st.caption("Los siguientes puntajes provienen de la precarga del prototipo y no incluyen Seguridad.")
            st.dataframe(pd.DataFrame([{
                "Eficiencia (0-1)": row.get("eficiencia_score"),
                "Equidad (0-1)": row.get("equidad_score"),
                "Territorio (0-1)": row.get("territorio_score"),
                "Método eficiencia": row.get("eficiencia_metodo"),
            }]), use_container_width=True, hide_index=True)

with TAB_SECURITY:
    st.subheader("Completar Seguridad")
    selected = st.session_state.selected_code
    if not selected:
        st.info("Primero busque y seleccione un local en la pestaña ‘Buscar local’. También puede cargar Seguridad en bloque más abajo.")
    else:
        match = base.loc[base["cod_local"] == selected]
        if not match.empty:
            row = match.iloc[0]
            render_local_header(row)
            rec = {**blank_record(), **st.session_state.security_records.get(selected, {})}
            srec = cfg["security_recodes"]
            with st.form(f"safety_form_{selected}"):
                st.markdown("#### Seguridad 1 · Vulnerabilidad estructural y obsolescencia")
                q1, q2 = st.columns(2)
                area_demo = q1.number_input("Área que requiere demolición (m²)", min_value=0.0, value=float(rec["area_demoler_m2"] or 0), step=1.0)
                area_total = q2.number_input("Área techada total (m²)", min_value=0.0, value=float(rec["area_techada_m2"] or 0), step=1.0)
                material = st.selectbox("Material / sistema predominante", [""] + list(srec["material"].keys()), index=([""] + list(srec["material"].keys())).index(rec["material_sistema"]) if rec["material_sistema"] in srec["material"] else 0)
                norma = st.selectbox("Categoría normativa", [""] + list(srec["normativa"].keys()), index=([""] + list(srec["normativa"].keys())).index(rec["norma_sismica_categoria"]) if rec["norma_sismica_categoria"] in srec["normativa"] else 0)
                conservacion = st.selectbox("Estado de conservación", [""] + list(srec["conservacion"].keys()), index=([""] + list(srec["conservacion"].keys())).index(rec["estado_conservacion"]) if rec["estado_conservacion"] in srec["conservacion"] else 0)

                st.markdown("#### Seguridad 2–5 · Peligros del emplazamiento")
                a, b = st.columns(2)
                zona = a.selectbox("Seguridad 2 · Zona sísmica", [""] + list(srec["zona_sismica"].keys()), index=([""] + list(srec["zona_sismica"].keys())).index(rec["zona_sismica"]) if rec["zona_sismica"] in srec["zona_sismica"] else 0)
                mm_sus = b.selectbox("Seguridad 3 · Susceptibilidad a movimientos en masa", [""] + list(srec["nivel_ordinal"].keys()), index=([""] + list(srec["nivel_ordinal"].keys())).index(rec["mm_susceptibilidad"]) if rec["mm_susceptibilidad"] in srec["nivel_ordinal"] else 0)
                mm_occ = st.selectbox("Seguridad 3 · Ocurrencia registrada", [""] + list(srec["mm_ocurrencia"].keys()), index=([""] + list(srec["mm_ocurrencia"].keys())).index(rec["mm_ocurrencia"]) if rec["mm_ocurrencia"] in srec["mm_ocurrencia"] else 0)
                a, b = st.columns(2)
                flood = a.selectbox("Seguridad 4 · Peligro de inundación", [""] + [x for x in srec["nivel_ordinal"].keys() if x != "Muy bajo"], index=0 if not rec["inundacion_peligro"] else ([""] + [x for x in srec["nivel_ordinal"].keys() if x != "Muy bajo"]).index(rec["inundacion_peligro"]))
                recurrence = b.number_input("Seguridad 4 · Recurrencia histórica normalizada (0–1)", min_value=0.0, max_value=1.0, value=float(rec["inundacion_recurrencia_norm"] or 0), step=.05)
                hf = st.selectbox("Seguridad 5 · Heladas / friaje", [""] + list(srec["heladas_friaje"].keys()), index=([""] + list(srec["heladas_friaje"].keys())).index(rec["heladas_friaje"]) if rec["heladas_friaje"] in srec["heladas_friaje"] else 0)

                st.markdown("#### Trazabilidad")
                source = st.text_input("Fuente / documento / expediente", value=str(rec.get("fuente", "")))
                obs = st.text_area("Observaciones", value=str(rec.get("observaciones", "")))
                submitted = st.form_submit_button("Guardar Seguridad", type="primary")

            proposed = {
                "area_demoler_m2": area_demo if area_total > 0 or area_demo > 0 else None,
                "area_techada_m2": area_total if area_total > 0 else None,
                "material_sistema": material,
                "norma_sismica_categoria": norma,
                "estado_conservacion": conservacion,
                "zona_sismica": zona,
                "mm_susceptibilidad": mm_sus,
                "mm_ocurrencia": mm_occ,
                "inundacion_peligro": flood,
                "inundacion_recurrencia_norm": recurrence,
                "heladas_friaje": hf,
                "fuente": source,
                "fecha_actualizacion": str(date.today()),
                "observaciones": obs,
            }
            preview = calculate_safety(proposed, cfg)
            st.dataframe(pd.DataFrame([{
                "S1": preview["s1"], "S2": preview["s2"], "S3": preview["s3"], "S4": preview["s4"], "S5": preview["s5"],
                "Seguridad agregada": preview["seguridad_score"], "Completa": preview["seguridad_completa"], "Crítica": preview["seguridad_critica"]
            }]), use_container_width=True, hide_index=True)
            if preview["seguridad_critica"]:
                st.markdown("<div class='critical'><b>Bandera de seguridad crítica:</b> S1 supera el umbral configurado. El ranking puede mostrar estos locales primero para evitar que el riesgo estructural quede oculto por compensación con otros criterios.</div>", unsafe_allow_html=True)
            if submitted:
                st.session_state.security_records[selected] = proposed
                st.success("Información de Seguridad guardada para este local.")

    st.divider()
    st.markdown("### Carga en bloque")
    if TEMPLATE_FILE.exists():
        st.download_button("Descargar plantilla de Seguridad", TEMPLATE_FILE.read_bytes(), "plantilla_seguridad.csv", "text/csv")
    bulk = st.file_uploader("Cargar CSV o XLSX de Seguridad", type=["csv", "xlsx"], key="bulk_safety")
    if bulk is not None:
        try:
            bdf = data_from_upload(bulk)
            bdf.columns = [str(c).strip() for c in bdf.columns]
            if "cod_local" not in bdf:
                st.error("El archivo debe incluir la columna cod_local.")
            else:
                bdf["cod_local"] = bdf["cod_local"].astype(str).str.replace(r"\.0$", "", regex=True)
                known = set(base["cod_local"])
                loaded, unknown = 0, []
                for _, rr in bdf.iterrows():
                    code = rr["cod_local"]
                    if code not in known:
                        unknown.append(code)
                        continue
                    rec = blank_record()
                    for k in rec:
                        if k in bdf.columns and pd.notna(rr.get(k)):
                            rec[k] = rr.get(k)
                    st.session_state.security_records[code] = rec
                    loaded += 1
                st.success(f"Carga procesada: {loaded} locales reconocidos.")
                if unknown:
                    st.warning(f"{len(unknown)} códigos no se encontraron en la base. Ejemplos: {', '.join(unknown[:8])}")
        except Exception as exc:
            st.exception(exc)

with TAB_RANK:
    st.subheader("Ranking de priorización")
    if not weights_ok:
        st.error(weights_msg)
    scored = merge_scores(base, st.session_state.security_records, cfg, weights)
    complete = scored[scored["puntaje_total"].notna()].copy()
    p1, p2, p3 = st.columns(3)
    p1.metric("Con puntaje completo", f"{len(complete):,}")
    p2.metric("Pendientes de Seguridad", f"{len(scored)-len(complete):,}")
    p3.metric("Seguridad crítica", int(complete["seguridad_critica"].sum()) if not complete.empty else 0)

    level = st.radio("Ámbito del ranking", ["Regional", "Provincial", "Distrital"], horizontal=True)
    regions = sorted(x for x in scored["region"].dropna().unique() if str(x).strip())
    region = st.selectbox("Región", regions) if regions else ""
    scope = complete[complete["region"] == region].copy()
    province = district = None
    if level in {"Provincial", "Distrital"}:
        provinces = sorted(x for x in scored.loc[scored["region"] == region, "provincia"].dropna().unique() if str(x).strip())
        province = st.selectbox("Provincia", provinces) if provinces else ""
        scope = scope[scope["provincia"] == province]
    if level == "Distrital":
        districts = sorted(x for x in scored.loc[(scored["region"] == region) & (scored["provincia"] == province), "distrito"].dropna().unique() if str(x).strip())
        district = st.selectbox("Distrito", districts) if districts else ""
        scope = scope[scope["distrito"] == district]

    if scope.empty:
        st.info("No hay locales con Seguridad completa en este ámbito todavía.")
    else:
        if cfg["security_rule"].get("order_critical_first", True):
            scope = scope.sort_values(["seguridad_critica", "puntaje_total"], ascending=[False, False])
        else:
            scope = scope.sort_values("puntaje_total", ascending=False)
        scope.insert(0, "ranking", range(1, len(scope) + 1))
        cols_show = ["ranking", "cod_local", "nombre_local", "region", "provincia", "distrito", "seguridad_critica",
                     "seguridad_score", "eficiencia_score", "equidad_score", "territorio_score", "puntaje_total"]
        st.dataframe(scope[cols_show].head(500), use_container_width=True, hide_index=True,
                     column_config={"puntaje_total": st.column_config.NumberColumn("Puntaje total", format="%.2f")})
        filename = "ranking_" + level.lower() + ".csv"
        st.download_button("Descargar ranking completo del ámbito", scope.to_csv(index=False).encode("utf-8-sig"), filename, "text/csv")

    with st.expander("Ver locales pendientes de Seguridad en el ámbito seleccionado"):
        pend = scored[scored["puntaje_total"].isna() & (scored["region"] == region)]
        if province is not None:
            pend = pend[pend["provincia"] == province]
        if district is not None:
            pend = pend[pend["distrito"] == district]
        st.dataframe(pend[["cod_local", "nombre_local", "provincia", "distrito", "ugel"]].head(500), use_container_width=True, hide_index=True)

with TAB_METHOD:
    st.subheader("Metodología del prototipo")
    st.write("**Pesos por criterio:** los decide la DRE dentro de rangos definidos centralmente en `config_priorizacion.json`; deben sumar 100%.")
    st.write("**Subpesos:** son fijos y técnicos. No son editables por el usuario final; también viven en el archivo de configuración.")
    st.write("**Seguridad:** se completa manualmente o mediante carga en bloque. S1–S5 se transforman a escala 0–1 y se agregan con subpesos fijos.")
    st.write("**Eficiencia:** Eficiencia 1a para urbano (alumnos por millón de soles; terciles regionales) y Eficiencia 2 para rural. Eficiencia 1 se excluye.")
    st.write("**Equidad:** E1, E3 y E5 aprovechan campos de la base de brecha; E2 y E4 son DEMO hasta incorporar fuentes específicas.")
    st.write("**Territorio:** T1–T6 son DEMO porque varias fichas todavía requieren cerrar fórmula/fuente operativa. La interfaz ya está preparada para sustituirlos.")
    st.code(json.dumps(cfg, ensure_ascii=False, indent=2), language="json")
