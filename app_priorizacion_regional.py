# -*- coding: utf-8 -*-
"""Aplicativo DRE/UGEL para priorización de locales educativos."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from criterios_priorizacion import calculate_s1, validate_criterion_weights

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "locales_priorizacion.csv"
CONFIG_FILE = ROOT / "config_priorizacion.json"
HEADER_FILE = ROOT / "header_digeie_diplan.png"
TEMPLATE_FILE = ROOT / "plantilla_actualizacion_s1.csv"

st.set_page_config(page_title="Priorización de locales educativos", page_icon="🏫", layout="wide", initial_sidebar_state="expanded")

CRITERION_LABELS = {
    "seguridad": "Seguridad", "eficiencia": "Eficiencia", "equidad": "Equidad", "territorio": "Territorio"
}
SEC_LABELS = {
    "s1": "Seguridad 1: Área techada en riesgo de colapso y obsolescencia",
    "s2": "Seguridad 2: Nivel de peligro sísmico",
    "s3": "Seguridad 3: Susceptibilidad a movimientos en masa",
    "s4": "Seguridad 4: Nivel de peligro por inundación",
    "s5": "Seguridad 5: Exposición a heladas y friaje",
}
EQ_LABELS = {
    "eq1_pobreza": "Equidad 1: Población en edad normativa en situación de pobreza",
    "eq2_discapacidad": "Equidad 2: Presencia de población con discapacidad",
    "eq3_ruralidad": "Equidad 3: Estudiantes matriculados en locales educativos rurales",
    "eq4_pueblos_originarios": "Equidad 4: Comunidades indígenas y pueblos originarios",
    "eq5_ambitos_prioritarios": "Equidad 5: Frontera, VRAEM y Huallaga",
}
T_LABELS = {
    "t1": "Territorio 1: Impacto de corredores logísticos nacionales",
    "t2": "Territorio 2: Representatividad institucional en centros poblados rurales",
    "t3": "Territorio 3: Cluster de equipamientos complementarios",
    "t4": "Territorio 4: Cluster de equipamientos públicos/intervenciones",
    "t5": "Territorio 5: Cluster educativo que asegura trayectoria EBR",
    "t6": "Territorio 6: Cobertura espacial según distancia normativa",
}


def norm(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.casefold()).strip()


def fmt_score(v: object) -> str:
    try:
        x = float(v)
        if math.isnan(x): return "—"
        return f"{x:.3f}"
    except Exception:
        return "—"


def fmt_money(v: object) -> str:
    try:
        x = float(v)
        if math.isnan(x): return "—"
        return f"S/ {x:,.0f}"
    except Exception:
        return "—"


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Cargando locales educativos...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, dtype={"cod_local": str, "ubigeo": str}, low_memory=False)
    for c in ["cod_local", "ubigeo", "nombre_local", "region", "provincia", "distrito", "centro_poblado", "dre", "ugel", "area_censal", "busqueda"]:
        if c in df: df[c] = df[c].fillna("").astype(str)
    return df


def apply_theme(dark: bool) -> None:
    if dark:
        bg, panel, text, muted, line, accent, soft, warning = "#111418", "#1a1f24", "#f3f5f6", "#aeb7be", "#313941", "#cf202f", "#232a30", "#342a16"
    else:
        bg, panel, text, muted, line, accent, soft, warning = "#f7f8fa", "#ffffff", "#1f252b", "#626b73", "#dfe3e7", "#cf202f", "#f1f3f5", "#fff7e5"
    st.markdown(f"""
    <style>
    .stApp {{background:{bg}; color:{text};}}
    [data-testid="stSidebar"] {{background:{panel}; border-right:1px solid {line};}}
    .block-container {{padding-top:3rem;padding-bottom:3rem;max-width:1500px;}}
    .titlebox {{padding:1.1rem 0 .75rem 0;}}
    .titlebox h1 {{margin:0;color:{text};font-size:2rem;}}
    .titlebox p {{margin:.35rem 0 0;color:{muted};font-size:1rem;}}
    .card {{background:{panel};border:1px solid {line};border-radius:12px;padding:.85rem 1rem;margin:.55rem 0;}}
    .info {{background:{soft};border-left:5px solid {accent};border-radius:8px;padding:.75rem 1rem;margin:.6rem 0;}}
    .warning {{background:{warning};border-left:5px solid #b27b16;border-radius:8px;padding:.75rem 1rem;margin:.6rem 0;}}
    .small {{font-size:.86rem;color:{muted};}}
    [data-testid="stMetric"] {{background:{panel};border:1px solid {line};border-radius:12px;padding:.65rem;}}
    div[data-testid="stDataFrame"] {{border:1px solid {line};border-radius:10px;}}
    .pill {{display:inline-block;border:1px solid {line};border-radius:999px;padding:.15rem .55rem;margin-right:.35rem;font-size:.8rem;}}
    [data-baseweb="tab-list"] {{gap:.35rem;}}
    [data-baseweb="tab-list"] button {{border-radius:9px 9px 0 0; padding:.55rem .85rem;}}
    [data-baseweb="tab-list"] button:nth-child(3),
    [data-baseweb="tab-list"] button:nth-child(4),
    [data-baseweb="tab-list"] button:nth-child(5),
    [data-baseweb="tab-list"] button:nth-child(6) {{background:{soft}; border:1px solid {line};}}
    [data-baseweb="tab-list"] button:nth-child(7) {{background:{panel}; border:2px solid {accent}; font-weight:700;}}
    [data-baseweb="tab-list"] button:nth-child(8) {{background:{accent}; border:2px solid {accent}; color:white !important; font-weight:800;}}
    [data-baseweb="tab-list"] button:nth-child(8) p {{color:white !important;}}
    </style>
    """, unsafe_allow_html=True)


def default_weights(cfg: dict) -> dict[str, float]:
    return {k: float(v["default"]) for k, v in cfg["criterion_weights"].items()}


def current_weights(cfg: dict) -> dict[str, float]:
    st.sidebar.markdown("### Pesos por criterio")
    weights = {}
    for key, d in cfg["criterion_weights"].items():
        sk = f"w_{key}"
        if sk not in st.session_state: st.session_state[sk] = float(d["default"])
        weights[key] = st.sidebar.number_input(d["label"], min_value=float(d["min"]), max_value=float(d["max"]), step=1.0, key=sk,
                                                help=f"Rango permitido: {d['min']}%–{d['max']}%")
    ok, msg = validate_criterion_weights(weights, cfg)
    if ok: st.sidebar.success("Pesos válidos · 100%")
    else: st.sidebar.error(msg)
    return weights


def tercile_scores_by_region(df: pd.DataFrame, raw_col: str, mask: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    work = df.loc[mask & df[raw_col].notna(), ["region", raw_col]]
    for _, idx in work.groupby("region", dropna=False).groups.items():
        values = df.loc[idx, raw_col]
        pct = values.rank(method="average", pct=True)
        score = pd.Series(0.0, index=idx)
        score.loc[pct > 1/3] = 0.5
        score.loc[pct > 2/3] = 1.0
        out.loc[idx] = score
    return out


def apply_updates(base: pd.DataFrame, cfg: dict, weights: dict[str, float]) -> pd.DataFrame:
    """Aplica únicamente actualizaciones de Seguridad 1; la brecha permanece precargada."""
    df = base.copy()
    df["s1_current"] = df["s1_base"].astype(float)
    df["s1_actualizado"] = False
    for code, rec in st.session_state.s1_updates.items():
        idx = df.index[df["cod_local"] == str(code)]
        if len(idx):
            calc = calculate_s1(rec, cfg)
            if calc["s1"] is not None:
                df.loc[idx, "s1_current"] = calc["s1"]
                df.loc[idx, "s1_actualizado"] = True

    sw = cfg["subcriterion_weights"]["seguridad"]
    df["seguridad_score"] = (
        df["s1_current"] * sw["s1"] + df["s2_score"] * sw["s2"] + df["s3_score"] * sw["s3"]
        + df["s4_score"] * sw["s4"] + df["s5_score"] * sw["s5"]
    )
    df["seguridad_critica"] = df["s1_current"] >= float(cfg["security_rule"]["critical_s1_threshold"])

    # Eficiencia y brecha son información precargada por DIPLAN.
    df["brecha_soles_current"] = df["brecha_soles_base"].astype(float)
    df["eficiencia_1a_raw_current"] = df["eficiencia_1a_raw_base"].astype(float)
    df["eficiencia_1a_score_current"] = df["eficiencia_1a_score"].astype(float)
    df["eficiencia_score"] = df["eficiencia_score_base"].astype(float)

    df["puntaje_actual"] = 100 * (
        df["seguridad_score"] * weights["seguridad"] / 100
        + df["eficiencia_score"] * weights["eficiencia"] / 100
        + df["equidad_score"] * weights["equidad"] / 100
        + df["territorio_score"] * weights["territorio"] / 100
    )
    return df


def subset_scope(df: pd.DataFrame, profile: str, region: str, ugel: str) -> pd.DataFrame:
    out = df[df["region"] == region]
    if profile == "UGEL": out = out[out["ugel"] == ugel]
    return out.copy()


def rank_scope(df: pd.DataFrame, score_col: str, critical_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_critical"] = out[critical_col].fillna(False).astype(bool).astype(int)
    out["_code"] = pd.to_numeric(out["cod_local"], errors="coerce").fillna(10**12)
    out = out.sort_values(["_critical", score_col, "_code"], ascending=[False, False, True]).copy()
    out[out_col] = np.arange(1, len(out)+1)
    return out.drop(columns=["_critical", "_code"])



def rank_comparison_for_local(df: pd.DataFrame, scope: pd.DataFrame, row: pd.Series, profile: str) -> pd.DataFrame:
    """Devuelve ranking de línea de base, actual y variación para el local seleccionado."""
    code = row["cod_local"]
    specs = [
        ("Región", df[df["region"] == row["region"]].copy(), "ranking_base_regional"),
        ("Provincia", df[(df["region"] == row["region"]) & (df["provincia"] == row["provincia"])].copy(), "ranking_base_provincial"),
        ("Distrito", df[(df["region"] == row["region"]) & (df["provincia"] == row["provincia"]) & (df["distrito"] == row["distrito"])].copy(), "ranking_base_distrital"),
    ]
    if profile == "UGEL":
        specs.insert(0, ("Mi UGEL", scope.copy(), None))
    rows = []
    for label, sub, base_col in specs:
        if sub.empty:
            continue
        if base_col is None:
            b = rank_scope(sub, "puntaje_base_diplan", "seguridad_critica_base", "rank_base")
            base_map = b.set_index("cod_local")["rank_base"]
            rank_base = float(base_map.get(code, np.nan))
        else:
            rank_base = float(row[base_col]) if pd.notna(row[base_col]) else np.nan
        c = rank_scope(sub, "puntaje_actual", "seguridad_critica", "rank_actual")
        cur_map = c.set_index("cod_local")["rank_actual"]
        rank_actual = float(cur_map.get(code, np.nan))
        variation = rank_base - rank_actual if pd.notna(rank_base) and pd.notna(rank_actual) else np.nan
        rows.append({"Ámbito": label, "Ranking línea de base": rank_base, "Ranking actual": rank_actual, "Variación": variation})
    return pd.DataFrame(rows)

def local_card(row: pd.Series) -> None:
    st.markdown(f"""<div class='card'><b>{row['cod_local']} · {row['nombre_local']}</b><br>
    {row['region']} / {row['provincia']} / {row['distrito']} · {row['ugel']}<br>
    <span class='small'>{row['centro_poblado']} · {row['area_censal']} · matrícula: {row.get('matricula', 0):,.0f}</span></div>""", unsafe_allow_html=True)


def select_local_from_scope(scope: pd.DataFrame, key: str) -> None:
    q = st.text_input("Buscar dentro de mi ámbito por código, nombre, centro poblado o distrito", key=f"search_{key}", placeholder="Ej.: 175")
    if q.strip():
        nq = norm(q)
        cand = scope[scope["busqueda"].str.contains(re.escape(nq), na=False)].head(100)
        if cand.empty:
            st.warning("No se encontraron coincidencias en su ámbito.")
        else:
            labels = {r.cod_local: f"{r.cod_local} · {r.nombre_local} · {r.distrito}" for r in cand.itertuples()}
            code = st.selectbox("Seleccionar local", cand["cod_local"].tolist(), format_func=lambda x: labels.get(x,x), key=f"sel_{key}")
            if st.button("Abrir local", key=f"open_{key}", type="primary"):
                st.session_state.selected_code = code
                st.success(f"Local {code} seleccionado.")


def selected_row(df: pd.DataFrame) -> pd.Series | None:
    code = st.session_state.selected_code
    if not code: return None
    m = df[df["cod_local"] == code]
    return None if m.empty else m.iloc[0]


def subcriterion_table(rows: list[dict]) -> None:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 column_config={"Puntaje": st.column_config.NumberColumn(format="%.3f"), "Peso interno": st.column_config.NumberColumn(format="%.0f%%"), "Contribución": st.column_config.NumberColumn(format="%.3f")})


def render_scope_map(scope: pd.DataFrame, selected_code: str = "") -> str | None:
    """Visor GIS del ámbito, vinculado en ambos sentidos con el local seleccionado."""
    if "latitud" not in scope.columns or "longitud" not in scope.columns:
        st.sidebar.caption("No se encontraron coordenadas para construir el visor.")
        return None

    work = scope[["cod_local", "nombre_local", "distrito", "ugel", "latitud", "longitud"]].copy()
    work["latitud"] = pd.to_numeric(work["latitud"], errors="coerce")
    work["longitud"] = pd.to_numeric(work["longitud"], errors="coerce")
    work = work[
        work["latitud"].between(-90, 90)
        & work["longitud"].between(-180, 180)
    ].dropna(subset=["latitud", "longitud"])

    if work.empty:
        st.sidebar.caption("No hay locales georreferenciados en el ámbito seleccionado.")
        return None

    selected = work.loc[work["cod_local"].astype(str) == str(selected_code)]
    if not selected.empty:
        center = [float(selected.iloc[0]["latitud"]), float(selected.iloc[0]["longitud"])]
        zoom_start = 16
    else:
        center = [float(work["latitud"].mean()), float(work["longitud"].mean())]
        zoom_start = 7

    m = folium.Map(location=center, zoom_start=zoom_start, tiles=None, control_scale=True)

    # Las mismas tres capas del visor EMI.
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors",
        name="Calles (OpenStreetMap)",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO",
        name="Mapa claro",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Imagen satelital",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    cluster = MarkerCluster(name="Locales educativos", control=False, options={"showCoverageOnHover": False}).add_to(m)
    bounds = []
    selected_marker = None
    for x in work.itertuples(index=False):
        lat, lon = float(x.latitud), float(x.longitud)
        code = str(x.cod_local)
        popup = folium.Popup(
            f"<b>{code} · {x.nombre_local}</b><br>{x.distrito}<br>{x.ugel}",
            max_width=330,
        )
        tooltip = f"{code} · {x.nombre_local}"
        if code == str(selected_code):
            selected_marker = folium.Marker(
                [lat, lon], popup=popup, tooltip=tooltip,
                icon=folium.Icon(color="red", icon="school", prefix="fa")
            ).add_to(m)
        else:
            folium.CircleMarker(
                [lat, lon], radius=4, color="#2c6e9b", weight=1,
                fill=True, fill_opacity=.85, popup=popup, tooltip=tooltip
            ).add_to(cluster)
        bounds.append([lat, lon])

    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    if selected.empty and bounds:
        m.fit_bounds(bounds, padding=(12, 12), max_zoom=13)

    with st.sidebar:
        map_state = st_folium(
            m,
            height=355,
            use_container_width=True,
            returned_objects=["last_object_clicked"],
            key=f"scope_map_{profile}_{region}_{ugel}",
        )
        st.caption(
            f"{len(work):,} locales georreferenciados · "
            "Calles (OpenStreetMap), Mapa claro e Imagen satelital"
        )

    clicked = (map_state or {}).get("last_object_clicked")
    if clicked and clicked.get("lat") is not None and clicked.get("lng") is not None:
        lat, lon = float(clicked["lat"]), float(clicked["lng"])
        d2 = (work["latitud"] - lat) ** 2 + (work["longitud"] - lon) ** 2
        if not d2.empty:
            return str(work.loc[d2.idxmin(), "cod_local"])
    return None


cfg = load_config()
base = load_data()
for name, default in [("s1_updates", {}), ("selected_code", "")]:
    if name not in st.session_state: st.session_state[name] = default

# Sidebar: tema + ámbito institucional
st.sidebar.markdown("## Ámbito de trabajo")
dark = st.sidebar.toggle("Modo oscuro", value=False)
apply_theme(dark)
profile = st.sidebar.radio("Perfil de trabajo", ["DRE", "UGEL"], horizontal=True)
regions = sorted(x for x in base["region"].unique() if x)
region = st.sidebar.selectbox("Región", regions)
ugel = ""
if profile == "UGEL":
    ugels = sorted(x for x in base.loc[base["region"] == region, "ugel"].unique() if x)
    ugel = st.sidebar.selectbox("UGEL", ugels)
weights = current_weights(cfg)
weights_ok, _ = validate_criterion_weights(weights, cfg)

current = apply_updates(base, cfg, weights)
scope = subset_scope(current, profile, region, ugel)
if st.session_state.selected_code and st.session_state.selected_code not in set(scope["cod_local"]):
    st.session_state.selected_code = ""

st.sidebar.markdown("### Visor de locales educativos")
clicked_code = render_scope_map(scope, st.session_state.selected_code)
if clicked_code and clicked_code != st.session_state.selected_code:
    st.session_state.selected_code = clicked_code
    st.rerun()

# Header
h1, h2 = st.columns([3.2, 2])
with h1:
    st.markdown("<div class='titlebox'><h1>Priorización de locales educativos</h1><p>Elaborado por el Ministerio de Educación DIGEIE-DIPLAN.</p></div>", unsafe_allow_html=True)
with h2:
    if HEADER_FILE.exists(): st.image(str(HEADER_FILE), use_container_width=True)

TABS = st.tabs(["Inicio", "Locales de mi ámbito", "Seguridad", "Eficiencia", "Equidad", "Territorio", "Resumen", "Ranking", "Metodología"])

# Inicio
with TABS[0]:
    c1,c2,c3 = st.columns(3)
    c1.metric("Locales en mi ámbito", f"{len(scope):,}")
    c2.metric("S1 actualizados", f"{int(scope['s1_actualizado'].sum()):,}")
    c3.metric("Puntaje base promedio", f"{scope['puntaje_base_diplan'].mean():.1f}")
    baseline = rank_scope(scope, "puntaje_base_diplan", "seguridad_critica_base", "Ranking línea base").head(20)
    st.subheader("Primeros 20 locales de la línea de base de mi ámbito")
    st.dataframe(baseline[["Ranking línea base","cod_local","nombre_local","provincia","distrito","ugel","puntaje_base_diplan"]], use_container_width=True, hide_index=True)

# Locales
with TABS[1]:
    st.subheader("Locales educativos de mi ámbito")
    st.caption(f"Perfil: {profile} · {region}" + (f" · {ugel}" if profile == "UGEL" else ""))
    list_df = scope[["cod_local","nombre_local","provincia","distrito","ugel","area_censal","ranking_base_regional","ranking_base_provincial","ranking_base_distrital","s1_actualizado"]].copy()
    list_df["Estado S1"] = np.where(list_df["s1_actualizado"], "Actualizado", "Por actualizar")
    list_df = list_df.drop(columns=["s1_actualizado"]).rename(columns={"ranking_base_regional":"Rank base región","ranking_base_provincial":"Rank base provincia","ranking_base_distrital":"Rank base distrito"})
    if st.session_state.selected_code and st.session_state.selected_code in set(list_df["cod_local"]):
        list_df.insert(0, "Seleccionado", np.where(list_df["cod_local"] == st.session_state.selected_code, "Sí", ""))
        list_df = pd.concat([
            list_df[list_df["cod_local"] == st.session_state.selected_code],
            list_df[list_df["cod_local"] != st.session_state.selected_code],
        ], ignore_index=True)
        sel = list_df.iloc[0]
        st.success(f"Local seleccionado: {sel['cod_local']} · {sel['nombre_local']}. El visor GIS está enfocado en este local.")
    table_event = st.dataframe(
        list_df, use_container_width=True, hide_index=True, height=470,
        on_select="rerun", selection_mode="single-row", key="scope_local_table"
    )
    selected_rows = getattr(getattr(table_event, "selection", None), "rows", [])
    if selected_rows:
        selected_from_table = str(list_df.iloc[selected_rows[0]]["cod_local"])
        if selected_from_table != st.session_state.selected_code:
            st.session_state.selected_code = selected_from_table
            st.rerun()
    st.caption("Seleccione una fila para ubicar automáticamente ese local en el visor GIS.")
    st.markdown("### Seleccionar un local para revisar/completar")
    select_local_from_scope(scope, "scope")
    row = selected_row(current)
    if row is not None: local_card(row)

# Seguridad
with TABS[2]:
    st.subheader("Seguridad")
    row = selected_row(current)
    if row is None:
        st.info("Seleccione un local en ‘Locales de mi ámbito’.")
    else:
        local_card(row)
        sw = cfg["subcriterion_weights"]["seguridad"]
        srows=[]
        for k in ["s1","s2","s3","s4","s5"]:
            score = row["s1_current"] if k=="s1" else row[f"{k}_score"]
            srows.append({"Subcriterio":SEC_LABELS[k],"Estado":("Actualizado" if row["s1_actualizado"] else "Línea de base") if k=="s1" else "Precargado","Puntaje":score,"Peso interno":sw[k]*100,"Contribución":score*sw[k]})
        subcriterion_table(srows)
        st.metric("Puntaje Seguridad", f"{row['seguridad_score']:.3f}")
        st.markdown("#### Actualizar únicamente Seguridad 1")
        saved = st.session_state.s1_updates.get(row["cod_local"], {})
        recodes = cfg["security_recodes"]
        with st.form(f"s1_{row['cod_local']}"):
            a,b=st.columns(2)
            area_demo = a.number_input("Área que requiere demolición (m²)", min_value=0.0, value=float(saved.get("area_demoler_m2") or 0), step=1.0)
            area_total = b.number_input("Área techada total (m²)", min_value=0.0, value=float(saved.get("area_techada_m2") or 0), step=1.0)
            mat_opts=[""]+list(recodes["material"]); norm_opts=[""]+list(recodes["normativa"]); cons_opts=[""]+list(recodes["conservacion"])
            material=st.selectbox("Material / sistema predominante", mat_opts, index=mat_opts.index(saved.get("material_sistema","")) if saved.get("material_sistema","") in mat_opts else 0)
            norma=st.selectbox("Categoría normativa", norm_opts, index=norm_opts.index(saved.get("norma_sismica_categoria","")) if saved.get("norma_sismica_categoria","") in norm_opts else 0)
            conservacion=st.selectbox("Estado de conservación", cons_opts, index=cons_opts.index(saved.get("estado_conservacion","")) if saved.get("estado_conservacion","") in cons_opts else 0)
            fuente=st.text_input("Fuente / informe / expediente", value=saved.get("fuente",""))
            observaciones=st.text_area("Observaciones", value=saved.get("observaciones",""))
            save=st.form_submit_button("Guardar actualización S1", type="primary")
        proposed={"area_demoler_m2":area_demo if area_total>0 else None,"area_techada_m2":area_total if area_total>0 else None,"material_sistema":material,"norma_sismica_categoria":norma,"estado_conservacion":conservacion,"fuente":fuente,"observaciones":observaciones,"fecha_actualizacion":str(date.today())}
        calc=calculate_s1(proposed,cfg)
        st.caption(f"Vista previa: RD={fmt_score(calc['rd'])} · OB={fmt_score(calc['ob'])} · S1={fmt_score(calc['s1'])} · Línea base S1={fmt_score(row['s1_base'])}")
        if save:
            if calc["s1"] is None: st.error("Complete información suficiente para calcular Seguridad 1.")
            else:
                st.session_state.s1_updates[row["cod_local"]]=proposed
                st.success("Seguridad 1 actualizada. El ranking se recalculará con este valor.")
                st.rerun()
        if row["s1_actualizado"] and st.button("Revertir S1 a línea de base"):
            st.session_state.s1_updates.pop(row["cod_local"],None); st.rerun()

# Eficiencia
with TABS[3]:
    st.subheader("Eficiencia")
    row=selected_row(current)
    if row is None:
        st.info("Seleccione un local en ‘Locales de mi ámbito’.")
    else:
        local_card(row)
        urban="urb" in norm(row["area_censal"])
        data_rows=[
            {"Subcriterio":"Eficiencia 1a: alumnos beneficiados por millón de soles","Aplicación":"Sí" if urban else "No","Puntaje":row["eficiencia_1a_score"] if urban else np.nan},
            {"Subcriterio":"Eficiencia 2: eficacia rural por alumnos/locales beneficiados","Aplicación":"No" if urban else "Sí","Puntaje":row["eficiencia_2_score"] if not urban else np.nan},
        ]
        st.dataframe(pd.DataFrame(data_rows),use_container_width=True,hide_index=True)
        a,b,c=st.columns(3)
        a.metric("Brecha estimada precargada",fmt_money(row["brecha_soles_base"]))
        b.metric("Método aplicado", "Eficiencia 1a" if urban else "Eficiencia 2")
        c.metric("Puntaje total Eficiencia",fmt_score(row["eficiencia_score"]))
        st.metric("Puntaje Eficiencia", f"{row['eficiencia_score']:.3f}")

# Equidad
with TABS[4]:
    st.subheader("Equidad")
    row=selected_row(current)
    if row is None: st.info("Seleccione un local en ‘Locales de mi ámbito’.")
    else:
        local_card(row); ew=cfg["subcriterion_weights"]["equidad"]
        rows=[]
        for k,label in EQ_LABELS.items():
            score=float(row[k]); rows.append({"Subcriterio":label,"Estado":"Precargado","Puntaje":score,"Peso interno":ew[k]*100,"Contribución":score*ew[k]})
        subcriterion_table(rows); st.metric("Puntaje Equidad",f"{row['equidad_score']:.3f}")

# Territorio
with TABS[5]:
    st.subheader("Territorio")
    row=selected_row(current)
    if row is None: st.info("Seleccione un local en ‘Locales de mi ámbito’.")
    else:
        local_card(row); tw=cfg["subcriterion_weights"]["territorio"]
        rows=[]
        for k,label in T_LABELS.items():
            score=float(row[f"{k}_score"]); rows.append({"Subcriterio":label,"Rango":int(row[f"{k}_rango"]),"Estado":"Precargado","Puntaje":score,"Peso interno":tw[k]*100,"Contribución":score*tw[k]})
        subcriterion_table(rows); st.metric("Puntaje Territorio",f"{row['territorio_score']:.3f}")

# Resumen
with TABS[6]:
    st.subheader("Resumen del local seleccionado")
    row = selected_row(current)
    if row is None:
        st.info("Seleccione un local en ‘Locales de mi ámbito’.")
    else:
        local_card(row)
        criterion_rows = pd.DataFrame([
            {"Criterio":"Seguridad", "Puntaje línea de base":row["seguridad_score_base"], "Puntaje actual":row["seguridad_score"], "Peso (%)":weights["seguridad"], "Contribución actual (puntos)":row["seguridad_score"]*weights["seguridad"]},
            {"Criterio":"Eficiencia", "Puntaje línea de base":row["eficiencia_score_base"], "Puntaje actual":row["eficiencia_score"], "Peso (%)":weights["eficiencia"], "Contribución actual (puntos)":row["eficiencia_score"]*weights["eficiencia"]},
            {"Criterio":"Equidad", "Puntaje línea de base":row["equidad_score"], "Puntaje actual":row["equidad_score"], "Peso (%)":weights["equidad"], "Contribución actual (puntos)":row["equidad_score"]*weights["equidad"]},
            {"Criterio":"Territorio", "Puntaje línea de base":row["territorio_score"], "Puntaje actual":row["territorio_score"], "Peso (%)":weights["territorio"], "Contribución actual (puntos)":row["territorio_score"]*weights["territorio"]},
        ])
        st.dataframe(criterion_rows, use_container_width=True, hide_index=True,
                     column_config={
                         "Puntaje línea de base": st.column_config.NumberColumn(format="%.3f"),
                         "Puntaje actual": st.column_config.NumberColumn(format="%.3f"),
                         "Peso (%)": st.column_config.NumberColumn(format="%.0f%%"),
                         "Contribución actual (puntos)": st.column_config.NumberColumn(format="%.2f"),
                     })
        c1,c2,c3 = st.columns(3)
        c1.metric("Puntaje total línea de base", f"{row['puntaje_base_diplan']:.2f}")
        c2.metric("Puntaje total actual", f"{row['puntaje_actual']:.2f}")
        c3.metric("Cambio en puntaje", f"{row['puntaje_actual']-row['puntaje_base_diplan']:+.2f}")
        st.markdown("### Cambio en el ranking")
        rc = rank_comparison_for_local(current, scope, row, profile)
        if not rc.empty:
            rc["Variación"] = rc["Variación"].map(lambda x: f"{int(x):+d}" if pd.notna(x) else "—")
            st.dataframe(rc, use_container_width=True, hide_index=True)
            st.caption("Variación positiva = el local mejora posiciones respecto de la línea de base; negativa = retrocede.")

# Ranking
with TABS[7]:
    st.subheader("Ranking: línea de base vs. ranking actualizado")
    level=st.radio("Ámbito del ranking",["Mi ámbito","Región","Provincia","Distrito"],horizontal=True)
    rscope=current[current["region"]==region].copy()
    baseline_col="ranking_base_regional"
    if level=="Mi ámbito":
        rscope=scope.copy()
        # ranking base recalculado dentro del ámbito DRE/UGEL para comparación homogénea
        b=rank_scope(rscope,"puntaje_base_diplan","seguridad_critica_base","ranking_base_scope")
        baseline_map=b.set_index("cod_local")["ranking_base_scope"]
        rscope["Ranking base"]=rscope["cod_local"].map(baseline_map)
    elif level=="Región":
        rscope["Ranking base"]=rscope["ranking_base_regional"]
    elif level=="Provincia":
        provinces=sorted(rscope["provincia"].dropna().unique())
        prov=st.selectbox("Provincia",provinces)
        rscope=rscope[rscope["provincia"]==prov].copy(); rscope["Ranking base"]=rscope["ranking_base_provincial"]
    else:
        provinces=sorted(rscope["provincia"].dropna().unique()); prov=st.selectbox("Provincia",provinces,key="rankprov")
        d0=rscope[rscope["provincia"]==prov]; districts=sorted(d0["distrito"].dropna().unique()); dist=st.selectbox("Distrito",districts)
        rscope=d0[d0["distrito"]==dist].copy(); rscope["Ranking base"]=rscope["ranking_base_distrital"]
    ranked=rank_scope(rscope,"puntaje_actual","seguridad_critica","Ranking actual")
    ranked["Variación"] = ranked["Ranking base"].astype(float) - ranked["Ranking actual"].astype(float)
    cols=["Ranking base","Ranking actual","Variación","cod_local","nombre_local","provincia","distrito","ugel","s1_actualizado","seguridad_score","eficiencia_score","equidad_score","territorio_score","puntaje_base_diplan","puntaje_actual"]
    view=ranked[cols].rename(columns={"s1_actualizado":"S1 actualizado","puntaje_base_diplan":"Puntaje base","puntaje_actual":"Puntaje actual"})
    st.caption("Variación positiva = el local mejora posiciones respecto de la línea de base.")
    st.dataframe(view,use_container_width=True,hide_index=True,height=560)
    st.download_button("Descargar ranking CSV",view.to_csv(index=False).encode("utf-8-sig"),"ranking_priorizacion.csv","text/csv")

# Metodología + carga masiva
with TABS[8]:
    st.subheader("Metodología y administración de actualizaciones")
    st.markdown("""
    <div class='card'>
    <b>Estructura:</b> Seguridad, Eficiencia, Equidad y Territorio se expresan en una escala 0–1. Los pesos internos de los subcriterios son fijos. La DRE define los pesos de los cuatro criterios dentro de los rangos permitidos y con suma igual a 100%.<br><br>
    <b>Línea de base:</b> todos los locales comienzan con brecha, puntajes y ranking precargados por DIPLAN. Seguridad 2–5, Eficiencia, Equidad y Territorio son de consulta. Únicamente Seguridad 1 es actualizable por la DRE/UGEL.<br><br>
    <b>Eficiencia:</b> se utiliza Eficiencia 1a en locales urbanos y Eficiencia 2 en locales rurales. La brecha estimada ya está precargada y no se edita en el aplicativo. Eficiencia 1 (nivel de utilización de infraestructura) no forma parte del aplicativo.
    </div>
    """,unsafe_allow_html=True)
    st.markdown("### Carga masiva de actualizaciones de Seguridad 1")
    if TEMPLATE_FILE.exists():
        st.download_button("Descargar plantilla S1",TEMPLATE_FILE.read_bytes(),TEMPLATE_FILE.name,"text/csv")
    upload=st.file_uploader("Cargar CSV/XLSX de actualizaciones S1",type=["csv","xlsx"])
    if upload is not None:
        try:
            udf=pd.read_csv(upload,dtype={"cod_local":str}) if upload.name.lower().endswith(".csv") else pd.read_excel(upload,dtype={"cod_local":str})
            n_s1=unknown=0
            known=set(base["cod_local"])
            for _,r in udf.iterrows():
                code=str(r.get("cod_local","")).replace(".0","").strip()
                if code not in known:
                    unknown+=1
                    continue
                s1rec={k:(None if pd.isna(r.get(k)) else r.get(k)) for k in ["area_demoler_m2","area_techada_m2","material_sistema","norma_sismica_categoria","estado_conservacion","fuente_s1","observaciones_s1"]}
                s1rec={"area_demoler_m2":s1rec["area_demoler_m2"],"area_techada_m2":s1rec["area_techada_m2"],"material_sistema":str(s1rec["material_sistema"] or ""),"norma_sismica_categoria":str(s1rec["norma_sismica_categoria"] or ""),"estado_conservacion":str(s1rec["estado_conservacion"] or ""),"fuente":str(s1rec["fuente_s1"] or ""),"observaciones":str(s1rec["observaciones_s1"] or ""),"fecha_actualizacion":str(date.today())}
                if calculate_s1(s1rec,cfg)["s1"] is not None:
                    st.session_state.s1_updates[code]=s1rec
                    n_s1+=1
            st.success(f"Carga procesada: {n_s1} actualizaciones S1, {unknown} códigos no encontrados.")
        except Exception as exc:
            st.error(f"No se pudo procesar el archivo: {exc}")
    if st.session_state.s1_updates:
        rows=[]
        for code in sorted(st.session_state.s1_updates):
            r={"cod_local":code}
            r.update({f"s1_{k}":v for k,v in st.session_state.s1_updates.get(code,{}).items()})
            rows.append(r)
        st.download_button("Descargar avances registrados",pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),"avances_priorizacion.csv","text/csv")
