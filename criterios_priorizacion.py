# -*- coding: utf-8 -*-
"""Motor de puntajes del aplicativo de priorización de locales educativos."""
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


def calculate_s1(record: dict[str, Any], cfg: dict) -> dict[str, float | None]:
    """Calcula S1 = máx(RD, OB) con los componentes de la ficha técnica."""
    recodes = cfg["security_recodes"]
    area_demo = num(record.get("area_demoler_m2"))
    area_total = num(record.get("area_techada_m2"))
    rd = None
    if area_demo is not None and area_total is not None and area_total > 0 and area_demo >= 0:
        rd = clip01(area_demo / area_total)

    mat = recodes["material"].get(str(record.get("material_sistema", "")), None)
    norma = recodes["normativa"].get(str(record.get("norma_sismica_categoria", "")), None)
    cons = recodes["conservacion"].get(str(record.get("estado_conservacion", "")), None)
    ob_vals = [x for x in (mat, norma, cons) if x is not None]
    ob = max(ob_vals) if ob_vals else None
    vals = [x for x in (rd, ob) if x is not None]
    s1 = max(vals) if vals else None
    return {"s1": s1, "rd": rd, "ob": ob}


def aggregate_safety(s1: Any, s2: Any, s3: Any, s4: Any, s5: Any, cfg: dict) -> dict[str, Any]:
    scores = {"s1": num(s1), "s2": num(s2), "s3": num(s3), "s4": num(s4), "s5": num(s5)}
    complete = all(v is not None for v in scores.values())
    aggregate = None
    if complete:
        w = cfg["subcriterion_weights"]["seguridad"]
        aggregate = sum(scores[k] * w[k] for k in scores)
    threshold = float(cfg["security_rule"]["critical_s1_threshold"])
    critical = scores["s1"] is not None and scores["s1"] >= threshold
    return {**scores, "seguridad_score": aggregate, "seguridad_completa": complete, "seguridad_critica": critical}


def validate_criterion_weights(weights: dict[str, float], cfg: dict) -> tuple[bool, str]:
    for key, definition in cfg["criterion_weights"].items():
        value = float(weights.get(key, 0))
        lo, hi = float(definition["min"]), float(definition["max"])
        if not (lo <= value <= hi):
            return False, f"{definition['label']} debe estar entre {lo:g}% y {hi:g}%."
    total = sum(float(v) for v in weights.values())
    if abs(total - 100.0) > 1e-8:
        return False, f"Los pesos deben sumar 100%. Actualmente suman {total:.1f}%."
    return True, "OK"


def total_score(row: dict[str, Any], weights: dict[str, float]) -> float | None:
    vals = {
        "seguridad": num(row.get("seguridad_score")),
        "eficiencia": num(row.get("eficiencia_score")),
        "equidad": num(row.get("equidad_score")),
        "territorio": num(row.get("territorio_score")),
    }
    if any(v is None for v in vals.values()):
        return None
    return 100.0 * sum(vals[k] * float(weights[k]) / 100.0 for k in vals)
