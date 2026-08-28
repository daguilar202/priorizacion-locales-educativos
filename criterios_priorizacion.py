# -*- coding: utf-8 -*-
"""Motor compartido de puntajes para Streamlit y pruebas."""
from __future__ import annotations

import math
from typing import Any


def num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def clip01(value: float | None) -> float | None:
    if value is None:
        return None
    return min(1.0, max(0.0, value))


def calculate_safety(record: dict[str, Any], cfg: dict) -> dict[str, Any]:
    """Calcula Seguridad 1-5 y el índice agregado con pesos técnicos fijos."""
    recodes = cfg["security_recodes"]
    sw = cfg["subcriterion_weights"]["seguridad"]

    # S1: ISE. Si hay RD válido, se combina con OB mediante máximo.
    area_demo = num(record.get("area_demoler_m2"))
    area_total = num(record.get("area_techada_m2"))
    rd = None
    if area_demo is not None and area_total is not None and area_total > 0 and area_demo >= 0:
        rd = clip01(area_demo / area_total)

    mat = recodes["material"].get(str(record.get("material_sistema", "")), None)
    norm = recodes["normativa"].get(str(record.get("norma_sismica_categoria", "")), None)
    cons = recodes["conservacion"].get(str(record.get("estado_conservacion", "")), None)
    ob_components = [x for x in (mat, norm, cons) if x is not None]
    ob = max(ob_components) if ob_components else None
    s1_components = [x for x in (rd, ob) if x is not None]
    s1 = max(s1_components) if s1_components else None

    # S2: zona sísmica. Mapeo configurable provisional si no se cargan factores Z exactos.
    s2 = recodes["zona_sismica"].get(str(record.get("zona_sismica", "")), None)

    # S3: movimientos en masa = max(susceptibilidad, ocurrencia)
    sus = recodes["nivel_ordinal"].get(str(record.get("mm_susceptibilidad", "")), None)
    occ = recodes["mm_ocurrencia"].get(str(record.get("mm_ocurrencia", "")), None)
    vals = [x for x in (sus, occ) if x is not None]
    s3 = max(vals) if vals else None

    # S4: inundación = max(peligro modelado, recurrencia normalizada)
    flood = recodes["nivel_ordinal"].get(str(record.get("inundacion_peligro", "")), None)
    recurrence = clip01(num(record.get("inundacion_recurrencia_norm")))
    vals = [x for x in (flood, recurrence) if x is not None]
    s4 = max(vals) if vals else None

    # S5: heladas/friaje
    s5 = recodes["heladas_friaje"].get(str(record.get("heladas_friaje", "")), None)

    scores = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
    complete = all(v is not None for v in scores.values())
    aggregate = None
    if complete:
        aggregate = sum(scores[k] * sw[k] for k in scores)

    critical_threshold = float(cfg["security_rule"]["critical_s1_threshold"])
    critical = s1 is not None and s1 >= critical_threshold

    return {
        **scores,
        "rd": rd,
        "ob": ob,
        "seguridad_score": aggregate,
        "seguridad_completa": complete,
        "seguridad_critica": critical,
    }


def validate_criterion_weights(weights: dict[str, float], cfg: dict) -> tuple[bool, str]:
    for key, definition in cfg["criterion_weights"].items():
        value = float(weights.get(key, 0))
        lo, hi = float(definition["min"]), float(definition["max"])
        if not (lo <= value <= hi):
            return False, f"{key.title()} debe estar entre {lo:g}% y {hi:g}%."
    total = sum(float(v) for v in weights.values())
    if abs(total - 100.0) > 1e-8:
        return False, f"Los pesos deben sumar 100%. Actualmente suman {total:.1f}%."
    return True, "OK"


def total_score(row: dict[str, Any], weights: dict[str, float]) -> float | None:
    safety = num(row.get("seguridad_score"))
    eff = num(row.get("eficiencia_score"))
    eq = num(row.get("equidad_score"))
    ter = num(row.get("territorio_score"))
    if any(v is None for v in (safety, eff, eq, ter)):
        return None
    return 100.0 * (
        safety * weights["seguridad"] / 100
        + eff * weights["eficiencia"] / 100
        + eq * weights["equidad"] / 100
        + ter * weights["territorio"] / 100
    )
