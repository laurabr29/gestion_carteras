"""
Análisis Multifactor de Empresas Españolas — Dos versiones comparadas
Analista Cuantitativo — Análisis Fundamental
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/laura/Desktop/gestion_carteras/"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN COMÚN
# ─────────────────────────────────────────────────────────────────────────────

sector_map = {
    "REP.MC": "Energía",    "ELE.MC": "Energía",    "ENG.MC": "Energía",
    "IBE.MC": "Energía",    "NTGY.MC": "Energía",   "SLR.MC": "Energía",
    "ANE.MC": "Energía",    "SOLARPACK.MC": "Energía", "CFG.MC": "Energía",
    "SAN.MC": "Financiero", "BBVA.MC": "Financiero","CABK.MC": "Financiero",
    "SAB.MC": "Financiero", "BKT.MC": "Financiero", "MAP.MC": "Financiero",
    "UNI.MC": "Financiero", "GCO.MC": "Financiero", "MORA.MC": "Financiero",
    "RENTA4.MC": "Financiero", "AYALA.MC": "Financiero",
    "MRL.MC": "Inmobiliario","COL.MC": "Inmobiliario","AEDAS.MC": "Inmobiliario",
    "NEINOR.MC": "Inmobiliario","VIA.MC": "Inmobiliario","CASH.MC": "Inmobiliario",
    "TEF.MC": "Telecomunicaciones","CLNX.MC": "Telecomunicaciones",
    "AMS.MC": "Tecnología", "IDR.MC": "Tecnología", "AGIL.MC": "Tecnología",
    "IATEC.MC": "Tecnología","GEST.MC": "Tecnología","EDR.MC": "Tecnología",
    "GRF.MC": "Salud",      "ALM.MC": "Salud",      "ROVI.MC": "Salud",
    "PHM.MC": "Salud",      "FDR.MC": "Salud",
    "ITX.MC": "Consumo",    "PUIG.MC": "Consumo",   "VID.MC": "Consumo",
    "MDF.MC": "Consumo",    "DIGI.MC": "Consumo",
    "ACS.MC": "Industrial", "FER.MC": "Industrial", "OHLA.MC": "Industrial",
    "CAF.MC": "Industrial", "TALGO.MC": "Industrial","TRE.MC": "Industrial",
    "ACR.MC": "Industrial", "SCYR.MC": "Industrial", "IBS.MC": "Industrial",
    "APAM.MC": "Industrial",
    "IAG.MC": "Turismo",    "AIR.MC": "Turismo",    "MEL.MC": "Turismo",
    "NH.MC": "Turismo",
    "EBRO.MC": "Alimentación","VIS.MC": "Alimentación","DEOLEO.MC": "Alimentación",
    "ALNT.MC": "Alimentación",
    "MTS.MC": "Materiales", "ACX.MC": "Materiales", "CIE.MC": "Materiales",
    "ENCE.MC": "Materiales",
    "A3M.MC": "Medios",     "TL5.MC": "Medios",
    "AENA.MC": "Transporte","RED.MC": "Transporte", "LOG.MC": "Transporte",
    "ANA.MC": "Transporte", "COR.MC": "Transporte", "SPS.MC": "Transporte",
    "FCC.MC": "Servicios",
}

nombre_map = {
    "ITX.MC": "Inditex",        "SAN.MC": "Santander",      "IBE.MC": "Iberdrola",
    "BBVA.MC": "BBVA",           "CABK.MC": "CaixaBank",     "FER.MC": "Ferrovial",
    "AENA.MC": "AENA",           "ELE.MC": "Endesa",         "MTS.MC": "ArcelorMittal",
    "ACS.MC": "ACS",             "NTGY.MC": "Naturgy",       "AMS.MC": "Amadeus",
    "REP.MC": "Repsol",          "TEF.MC": "Telefónica",     "CLNX.MC": "Cellnex",
    "IAG.MC": "IAG (Iberia)",    "SAB.MC": "Sabadell",       "BKT.MC": "Bankinter",
    "ANA.MC": "Acciona",         "MAP.MC": "Mapfre",         "IDR.MC": "Indra",
    "PUIG.MC": "Puig Brands",    "RED.MC": "Redeia",         "MRL.MC": "Merlin Properties",
    "UNI.MC": "Unicaja",         "ANE.MC": "Acciona Energías","GRF.MC": "Grifols",
    "ROVI.MC": "Rovi",           "LOG.MC": "Logista",        "FDR.MC": "Fluidra",
    "ENG.MC": "Enagas",          "SCYR.MC": "Sacyr",         "COL.MC": "Colonial",
    "ACX.MC": "Acerinox",        "SLR.MC": "Solaria",
    "CAF.MC": "CAF",             "MEL.MC": "Meliá Hotels",   "AIR.MC": "Airbus",
    "A3M.MC": "Atresmedia",      "TL5.MC": "Mediaset España",
    "PHM.MC": "Pharmamar",       "ALM.MC": "Almirall",       "AEDAS.MC": "Aedas Homes",
    "NEINOR.MC": "Neinor Homes", "VID.MC": "Vidrala",        "VIS.MC": "Viscofan",
    "EBRO.MC": "Ebro Foods",     "CIE.MC": "CIE Automotive",
    "TALGO.MC": "Talgo",         "OHLA.MC": "OHLA",          "ACR.MC": "Acrinox",
    "DIGI.MC": "Digi",           "CASH.MC": "Castellana Properties",
    "TRE.MC": "Técnicas Reunidas","GEST.MC": "Gigas Hosting",
}

METRICAS = {
    "valoracion": {
        "EV_EBITDA":       {"lower_better": True,  "peso": 0.35},
        "price_to_book":   {"lower_better": True,  "peso": 0.25},
        "EV_revenue":      {"lower_better": True,  "peso": 0.25},
        "PEG":             {"lower_better": True,  "peso": 0.15},
    },
    "rentabilidad": {
        "ROE":             {"lower_better": False, "peso": 0.30},
        "ROIC":            {"lower_better": False, "peso": 0.30},
        "ROA":             {"lower_better": False, "peso": 0.20},
        "net_margin":      {"lower_better": False, "peso": 0.20},
    },
    "crecimiento": {
        "one_year_return": {"lower_better": False, "peso": 0.60},
        "PEG":             {"lower_better": True,  "peso": 0.40},
    },
    "calidad": {
        "fcf_margin":       {"lower_better": False, "peso": 0.40},
        "operating_margin": {"lower_better": False, "peso": 0.35},
        "gross_margin":     {"lower_better": False, "peso": 0.25},
    },
    "solidez": {
        "ROA":             {"lower_better": False, "peso": 0.50},
        "price_to_cash":   {"lower_better": True,  "peso": 0.50},
    },
}

PESOS_CATEGORIA = {
    "calidad": 0.30, "rentabilidad": 0.25, "valoracion": 0.20,
    "crecimiento": 0.15, "solidez": 0.10,
}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE ANÁLISIS
# ─────────────────────────────────────────────────────────────────────────────

def parse_pct_or_num(val):
    if pd.isna(val) or val == "—":
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace('\xa0', '').replace('\u2212', '-').replace(',', '.')
    if s.endswith('%'):
        try: return float(s[:-1]) / 100
        except: return np.nan
    try: return float(s)
    except: return np.nan

def extract_ticker_xlsx(simbolo):
    ibex35_tickers = [
        "ITX","SAN","IBE","BBVA","CABK","FER","AENA","ELE","MTS","ACS",
        "NTGY","AMS","REP","TEF","CLNX","IAG","SAB","BKT","ANA","MAP",
        "IDR","PUIG","RED","MRL","UNI","ANE","GRF","ROVI","LOG","FDR",
        "ENG","SCYR","COL","ACX","SLR","FCC","ACR",
    ]
    s = str(simbolo)
    for t in sorted(ibex35_tickers, key=len, reverse=True):
        if s.startswith(t):
            return t + ".MC"
    m = re.match(r'^([A-Z0-9]+?)(?=[A-Z][a-z])', s)
    if not m:
        m = re.match(r'^([A-Z0-9]+)', s)
    return (m.group(1) + ".MC") if m else s

def cargar_ibex35_xlsx():
    raw = pd.read_excel(BASE + "ibex35.xlsx")
    raw = raw[raw["Símbolo"].notna()].copy()
    raw = raw[~raw["Símbolo"].isin(["D", "DREIT"])].copy()
    rows = []
    for _, row in raw.iterrows():
        ticker = extract_ticker_xlsx(row["Símbolo"])
        rows.append({
            "empresa":         ticker,
            "ROA":             parse_pct_or_num(row.get("ROA")),
            "ROE":             parse_pct_or_num(row.get("ROE")),
            "ROIC":            parse_pct_or_num(row.get("ROIC")),
            "gross_margin":    parse_pct_or_num(row.get("Margen bruto")),
            "operating_margin":parse_pct_or_num(row.get("Margen explotación")),
            "net_margin":      parse_pct_or_num(row.get("Margen neto")),
            "fcf_margin":      parse_pct_or_num(row.get("Margen de flujo de efectivo disponible")),
            "price_to_book":   parse_pct_or_num(row.get("P/B")),
            "price_to_sales":  parse_pct_or_num(row.get("Precio/ventas (P/S)")),
            "PEG":             parse_pct_or_num(row.get("PEG")),
            "EV_EBITDA":       parse_pct_or_num(row.get("EV/EBITDA")),
            "EV_revenue":      parse_pct_or_num(row.get("EV/ingresos")),
            "one_year_return": parse_pct_or_num(row.get("Rendimiento de la capitalización de mercado\xa0%")),
            "indice":          "Ibex 35",
        })
    return pd.DataFrame(rows)

def asignar_sector(row):
    ticker = row["ticker"]
    if ticker in sector_map:
        return sector_map[ticker]
    indice = row.get("indice", "")
    if indice == "BME Growth":    return "Pequeña Cap / Growth"
    if indice == "Ibex Small Cap": return "Pequeña Cap"
    if indice == "Ibex Medium Cap":return "Mediana Cap"
    return "Gran Cap"

def get_nombre(ticker):
    return nombre_map.get(ticker, ticker.replace(".MC", "").replace("_", " "))

def normalizar_serie(serie, lower_better=True):
    ranks = serie.rank(pct=True, na_option="keep")
    return (1 - ranks) if lower_better else ranks

def calcular_score_categoria(df_grupo, metricas_cat):
    score = pd.Series(0.0, index=df_grupo.index)
    peso_total = 0.0
    for col, cfg in metricas_cat.items():
        if col not in df_grupo.columns:
            continue
        serie = df_grupo[col]
        if serie.notna().sum() < 2:
            continue
        norm = normalizar_serie(serie, cfg["lower_better"])
        score += norm.fillna(0) * cfg["peso"]
        peso_total += cfg["peso"]
    return (score / peso_total) if peso_total > 0 else score

def run_analysis(df_raw, label):
    """Limpieza + scoring + top10 sobre un DataFrame ya combinado."""
    df = df_raw.copy()
    if "empresa" in df.columns:
        df.rename(columns={"empresa": "ticker"}, inplace=True)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["non_null_count"] = df.notna().sum(axis=1)
    df = df.sort_values("non_null_count", ascending=False)
    df = df.drop_duplicates(subset="ticker", keep="first")
    df = df.drop(columns=["non_null_count"]).reset_index(drop=True)
    for col in [c for c in df.columns if c not in ["ticker", "indice"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sector"] = df.apply(asignar_sector, axis=1)
    df["nombre"] = df["ticker"].apply(get_nombre)

    resultados = []
    for sector, grupo in df.groupby("sector"):
        grupo = grupo.copy()
        scores_cat = {cat: calcular_score_categoria(grupo, m) for cat, m in METRICAS.items()}
        grupo["score_total"]       = sum(scores_cat[c] * p for c, p in PESOS_CATEGORIA.items())
        grupo["score_valoracion"]  = scores_cat["valoracion"]
        grupo["score_rentabilidad"]= scores_cat["rentabilidad"]
        grupo["score_crecimiento"] = scores_cat["crecimiento"]
        grupo["score_calidad"]     = scores_cat["calidad"]
        grupo["score_solidez"]     = scores_cat["solidez"]
        resultados.append(grupo)

    df_scored = pd.concat(resultados).reset_index(drop=True)
    for col in ["score_total","score_valoracion","score_rentabilidad",
                "score_crecimiento","score_calidad","score_solidez"]:
        df_scored[col] = (df_scored[col] * 100).round(1)

    metricas_clave = ["ROE","ROA","net_margin","EV_EBITDA","operating_margin",
                      "price_to_book","fcf_margin","ROIC","one_year_return","gross_margin"]
    df_scored["metricas_disponibles"] = df_scored[metricas_clave].notna().sum(axis=1)
    df_filtered = df_scored[df_scored["metricas_disponibles"] >= 3].copy()
    top10 = df_filtered.nlargest(10, "score_total").reset_index(drop=True)
    top10.index = top10.index + 1

    print(f"\n[{label}] Empresas únicas: {len(df_scored)} | Sectores: {df_scored['sector'].nunique()}")
    print(f"[{label}] Top 10:")
    for i, row in top10.iterrows():
        print(f"  {i:2}. {row['nombre']:25} | {row['sector']:20} | {row['score_total']:5.1f}")

    return top10, df_scored

# ─────────────────────────────────────────────────────────────────────────────
# CARGAR Y EJECUTAR AMBOS ANÁLISIS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("CARGANDO DATOS...")

df_bme    = pd.read_csv(BASE + "bme_growth_dataset.csv");   df_bme["indice"]    = "BME Growth"
df_small  = pd.read_csv(BASE + "ibex_small_cap_dataset (1).csv"); df_small["indice"]  = "Ibex Small Cap"
df_medium = pd.read_csv(BASE + "ibex_medium_cap_dataset.csv");    df_medium["indice"] = "Ibex Medium Cap"

# Análisis v1: con ibex35.xlsx
df_ibex_v1 = cargar_ibex35_xlsx()
df_v1 = pd.concat([df_bme, df_small, df_medium, df_ibex_v1], ignore_index=True)
top10_v1, scored_v1 = run_analysis(df_v1, "V1 — ibex35.xlsx")

# Análisis v2: con ibex35_dataset.csv
df_ibex_v2 = pd.read_csv(BASE + "ibex35_dataset.csv"); df_ibex_v2["indice"] = "Ibex 35"
df_v2 = pd.concat([df_bme, df_small, df_medium, df_ibex_v2], ignore_index=True)
top10_v2, scored_v2 = run_analysis(df_v2, "V2 — ibex35_dataset.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES HTML
# ─────────────────────────────────────────────────────────────────────────────

def fmt(val, decimals=2, pct=False):
    if pd.isna(val): return "—"
    if pct: return f"{val*100:.{decimals}f}%"
    return f"{val:.{decimals}f}"

def score_color(score):
    if score >= 70: return "#10b981"
    if score >= 50: return "#f59e0b"
    return "#ef4444"

def score_bg(score):
    if score >= 70: return "bg-emerald-500"
    if score >= 50: return "bg-amber-500"
    return "bg-red-500"

colors_radar = [
    "rgba(99,102,241,0.7)","rgba(16,185,129,0.7)","rgba(245,158,11,0.7)",
    "rgba(239,68,68,0.7)", "rgba(59,130,246,0.7)","rgba(168,85,247,0.7)",
    "rgba(236,72,153,0.7)","rgba(20,184,166,0.7)","rgba(251,146,60,0.7)",
    "rgba(132,204,22,0.7)",
]

def build_chart_data(top10, suffix):
    labels_js = json.dumps(top10["nombre"].tolist())
    scores_js  = json.dumps(top10["score_total"].tolist())
    bar_colors = json.dumps([score_color(s) for s in top10["score_total"].tolist()])

    radar_datasets = []
    for idx, (_, row) in enumerate(top10.iterrows()):
        radar_datasets.append({
            "label": row["nombre"],
            "data": [
                float(row["score_calidad"]      or 0),
                float(row["score_rentabilidad"] or 0),
                float(row["score_valoracion"]   or 0),
                float(row["score_crecimiento"]  or 0),
                float(row["score_solidez"]      or 0),
            ],
            "borderColor": colors_radar[idx],
            "backgroundColor": colors_radar[idx].replace("0.7","0.15"),
            "borderWidth": 2, "pointRadius": 4,
        })
    radar_js = json.dumps({"labels": ["Calidad","Rentabilidad","Valoración","Crecimiento","Solidez"],
                           "datasets": radar_datasets})
    scatter_data = []
    for _, row in top10.iterrows():
        scatter_data.append({"x": float(row["score_calidad"] or 0),
                              "y": float(row["score_valoracion"] or 0),
                              "label": row["nombre"]})
    scatter_js = json.dumps(scatter_data)
    return labels_js, scores_js, bar_colors, radar_js, scatter_js

def generar_tarjeta(rank, row):
    sc = row["score_total"]
    color = score_color(sc)
    bg_cls = score_bg(sc)
    sub = "".join([
        f'<div class="text-center"><div class="text-xs text-gray-500">{lbl}</div>'
        f'<div class="text-xs font-bold text-gray-700">{fmt(val, decimals=0)}</div></div>'
        for lbl, val in [("Cal", row["score_calidad"]),("Rent", row["score_rentabilidad"]),
                         ("Val", row["score_valoracion"]),("Cre", row["score_crecimiento"]),
                         ("Sol", row["score_solidez"])]
    ])
    ratios = [
        ("ROE",       row.get("ROE"),             "indigo",  True),
        ("ROIC",      row.get("ROIC"),             "purple",  True),
        ("ROA",       row.get("ROA"),              "blue",    True),
        ("EV/EBITDA", row.get("EV_EBITDA"),        "emerald", False, "x"),
        ("P/B",       row.get("price_to_book"),    "amber",   False, "x"),
        ("Mg. Neto",  row.get("net_margin"),       "rose",    True),
        ("Mg. Op.",   row.get("operating_margin"), "teal",    True),
        ("FCF Mg.",   row.get("fcf_margin"),       "cyan",    True),
        ("Ret. 1Y",   row.get("one_year_return"),  "violet",  True),
    ]
    ratio_html = ""
    for r in ratios:
        lbl, val, clr = r[0], r[1], r[2]
        is_pct = r[3] if len(r) > 3 else True
        suffix2 = r[4] if len(r) > 4 else ""
        v = fmt(val, pct=is_pct) + suffix2
        ratio_html += (f'<div class="text-center p-2 bg-{clr}-50 rounded-lg">'
                       f'<div class="text-xs text-gray-500">{lbl}</div>'
                       f'<div class="text-sm font-bold text-{clr}-700">{v}</div></div>')
    return f"""
    <div class="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100 hover:shadow-xl transition-shadow duration-300">
      <div class="bg-gradient-to-r from-indigo-600 to-purple-600 p-4 relative">
        <div class="absolute top-3 right-3 w-10 h-10 rounded-full {bg_cls} flex items-center justify-center text-white font-bold text-sm">#{rank}</div>
        <div class="text-white">
          <div class="text-xs font-medium opacity-75 uppercase tracking-wider">{row['indice']}</div>
          <div class="text-2xl font-bold mt-0.5">{row['nombre']}</div>
          <div class="text-sm opacity-80 mt-0.5">{row['ticker']} · {row['sector']}</div>
        </div>
      </div>
      <div class="px-4 py-3 bg-gray-50 border-b border-gray-100">
        <div class="flex items-center justify-between mb-1.5">
          <span class="text-sm font-semibold text-gray-700">Score Total</span>
          <span class="text-2xl font-bold" style="color:{color}">{sc:.1f}</span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-2">
          <div class="h-2 rounded-full" style="width:{min(sc,100)}%;background:{color}"></div>
        </div>
        <div class="grid grid-cols-5 gap-1 mt-2">{sub}</div>
      </div>
      <div class="p-4 grid grid-cols-3 gap-3">{ratio_html}</div>
    </div>"""

def generar_explicacion(rank, row):
    sc = row["score_total"]
    color = score_color(sc)
    strengths = []
    checks = [
        ("ROE",             row.get("ROE"),             lambda v: v > 0.15, lambda v: f"ROE del {v*100:.1f}%"),
        ("ROIC",            row.get("ROIC"),            lambda v: v > 0.10, lambda v: f"ROIC del {v*100:.1f}%"),
        ("net_margin",      row.get("net_margin"),      lambda v: v > 0.10, lambda v: f"margen neto del {v*100:.1f}%"),
        ("fcf_margin",      row.get("fcf_margin"),      lambda v: v > 0.08, lambda v: f"FCF margin del {v*100:.1f}%"),
        ("EV_EBITDA",       row.get("EV_EBITDA"),       lambda v: v < 10,   lambda v: f"EV/EBITDA de {v:.1f}x"),
        ("operating_margin",row.get("operating_margin"),lambda v: v > 0.10, lambda v: f"margen operativo del {v*100:.1f}%"),
        ("one_year_return", row.get("one_year_return"), lambda v: v > 0.15, lambda v: f"retorno 1Y de +{v*100:.1f}%"),
        ("price_to_book",   row.get("price_to_book"),   lambda v: v < 2.0,  lambda v: f"P/B de {v:.2f}x"),
    ]
    for _, val, cond, desc in checks:
        if not pd.isna(val) and cond(val):
            strengths.append(desc(val))

    ev_val = row.get("EV_EBITDA"); pb_val = row.get("price_to_book")
    is_value = (not pd.isna(ev_val) and ev_val < 8) or (not pd.isna(pb_val) and pb_val < 1.5)
    value_badge = '<span class="inline-block bg-amber-100 text-amber-800 text-xs font-semibold px-2 py-0.5 rounded-full ml-2">⭐ Value Opportunity</span>' if is_value else ""
    strengths_text = "; ".join(strengths) if strengths else "posicionamiento sectorial favorable dentro de su grupo de pares."

    sub_scores = "".join([
        f'<div class="text-center bg-gray-50 rounded-lg p-2"><div class="text-xs text-gray-400">{lbl}</div>'
        f'<div class="text-sm font-bold" style="color:{score_color(float(val or 0))}">{fmt(val, decimals=0)}</div></div>'
        for lbl, val in [("Cal", row["score_calidad"]),("Rent", row["score_rentabilidad"]),
                         ("Val", row["score_valoracion"]),("Cre", row["score_crecimiento"]),
                         ("Sol", row["score_solidez"])]
    ])
    return f"""
    <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-shadow">
      <div class="flex items-start gap-4">
        <div class="w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center text-white font-bold text-sm" style="background:{color}">#{rank}</div>
        <div class="flex-1">
          <div class="flex items-center flex-wrap gap-2 mb-2">
            <h3 class="text-lg font-bold text-gray-800">{row['nombre']}</h3>
            <span class="text-sm text-gray-400">{row['ticker']}</span>
            <span class="bg-indigo-100 text-indigo-700 text-xs font-medium px-2 py-0.5 rounded-full">{row['sector']}</span>
            {value_badge}
          </div>
          <p class="text-gray-600 text-sm leading-relaxed">
            <strong>{row['nombre']}</strong> obtiene un score de <strong style="color:{color}">{sc:.1f}/100</strong>.
            Sus fortalezas incluyen {strengths_text}
          </p>
          <div class="mt-3 grid grid-cols-5 gap-2">{sub_scores}</div>
        </div>
      </div>
    </div>"""

def build_tab_content(top10, scored, suffix, fuentes):
    """Genera el HTML interior de una pestaña completa."""

    labels_js, scores_js, bar_colors_js, radar_js, scatter_js = build_chart_data(top10, suffix)

    # Fuentes badge
    fuentes_html = "".join(
        f'<span class="inline-flex items-center gap-1.5 bg-gray-100 text-gray-600 text-xs font-medium px-3 py-1.5 rounded-full">'
        f'<svg class="w-3 h-3 text-indigo-400" fill="currentColor" viewBox="0 0 20 20"><path d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"/></svg>'
        f'{f}</span>'
        for f in fuentes
    )

    # Podio top 3
    podio = ""
    for i, row in top10.head(3).iterrows():
        medal = ["🥇","🥈","🥉"][i-1]
        grad  = ["from-amber-400 to-yellow-300","from-slate-400 to-slate-300","from-orange-400 to-amber-300"][i-1]
        podio += f"""
      <div class="bg-gradient-to-br {grad} rounded-2xl p-5 text-white shadow-lg relative overflow-hidden">
        <div class="absolute -right-4 -top-4 text-7xl opacity-20">{medal}</div>
        <div class="text-3xl mb-1">{medal}</div>
        <div class="text-xl font-black">{row['nombre']}</div>
        <div class="text-sm opacity-80">{row['ticker']} · {row['sector']}</div>
        <div class="mt-3 text-3xl font-black">{row['score_total']:.1f}<span class="text-base font-normal opacity-70">/100</span></div>
      </div>"""

    # Lista 4-10
    lista = ""
    for i, row in top10.iloc[3:].iterrows():
        sc = row["score_total"]; color = score_color(sc)
        lista += f"""
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 px-5 py-3 flex items-center gap-4 hover:shadow-md transition-shadow">
        <div class="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-sm font-bold text-gray-500">#{i}</div>
        <div class="flex-1"><span class="font-semibold text-gray-800">{row['nombre']}</span>
          <span class="text-sm text-gray-400 ml-2">{row['ticker']}</span></div>
        <span class="text-sm text-gray-400 hidden md:block">{row['sector']}</span>
        <span class="text-sm text-gray-300">·</span>
        <span class="text-sm text-gray-400 hidden md:block">{row['indice']}</span>
        <div class="flex items-center gap-2 ml-4">
          <div class="w-24 bg-gray-100 rounded-full h-2">
            <div class="h-2 rounded-full" style="width:{min(sc,100)}%;background:{color}"></div>
          </div>
          <span class="font-bold text-base" style="color:{color}">{sc:.1f}</span>
        </div>
      </div>"""

    # Tabla
    filas = ""
    for i, row in top10.iterrows():
        sc = row["score_total"]; color = score_color(sc)
        filas += f"""
      <tr class="hover:bg-gray-50 transition-colors">
        <td class="px-4 py-3 text-center font-bold text-gray-500">#{i}</td>
        <td class="px-4 py-3"><div class="font-semibold text-gray-800">{row['nombre']}</div>
          <div class="text-xs text-gray-400">{row['ticker']}</div></td>
        <td class="px-4 py-3 text-sm text-gray-600">{row['sector']}</td>
        <td class="px-4 py-3 text-sm text-gray-500">{row['indice']}</td>
        <td class="px-4 py-3 text-center"><span class="font-bold text-lg" style="color:{color}">{sc:.1f}</span></td>
        <td class="px-4 py-3 text-center text-sm">{fmt(row.get('ROE'), pct=True)}</td>
        <td class="px-4 py-3 text-center text-sm">{fmt(row.get('ROIC'), pct=True)}</td>
        <td class="px-4 py-3 text-center text-sm">{fmt(row.get('ROA'), pct=True)}</td>
        <td class="px-4 py-3 text-center text-sm">{fmt(row.get('EV_EBITDA'), decimals=1)}x</td>
        <td class="px-4 py-3 text-center text-sm">{fmt(row.get('price_to_book'), decimals=2)}x</td>
        <td class="px-4 py-3 text-center text-sm">{fmt(row.get('net_margin'), pct=True)}</td>
        <td class="px-4 py-3 text-center text-sm">{fmt(row.get('operating_margin'), pct=True)}</td>
        <td class="px-4 py-3 text-center text-sm">{fmt(row.get('fcf_margin'), pct=True)}</td>
        <td class="px-4 py-3 text-center text-sm">{fmt(row.get('one_year_return'), pct=True)}</td>
      </tr>"""

    tarjetas = "\n".join(generar_tarjeta(i, row) for i, row in top10.iterrows())
    explicaciones = "\n".join(generar_explicacion(i, row) for i, row in top10.iterrows())

    return f"""
  <!-- ── FUENTES ───────────────────────────────────────────── -->
  <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
    <div class="flex items-center gap-2 mb-3">
      <svg class="w-4 h-4 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
      <span class="text-sm font-bold text-gray-700">Bases de datos utilizadas</span>
      <span class="ml-2 text-xs text-gray-400">{len(scored)} empresas · {scored['sector'].nunique()} sectores</span>
    </div>
    <div class="flex flex-wrap gap-2">{fuentes_html}</div>
  </div>

  <!-- ── METODOLOGÍA ──────────────────────────────────────── -->
  <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
    <p class="text-xs font-bold text-gray-600 mb-3">Pesos del modelo multifactor</p>
    <div class="grid grid-cols-5 gap-2">
      {"".join([
        f'<div class="text-center p-2 bg-gradient-to-b from-indigo-50 to-white rounded-xl border border-indigo-100">'
        f'<div class="text-xl font-black text-indigo-600">{p*100:.0f}%</div>'
        f'<div class="text-xs font-semibold text-gray-600 mt-0.5">{n}</div></div>'
        for n,p in [("Calidad",0.30),("Rentabilidad",0.25),("Valoración",0.20),("Crecimiento",0.15),("Solidez",0.10)]
      ])}
    </div>
  </div>

  <!-- ── RANKING ───────────────────────────────────────────── -->
  <div>
    <h3 class="text-lg font-bold text-gray-800 mb-4">Ranking Top 10</h3>
    <div class="grid grid-cols-3 gap-4 mb-4">{podio}</div>
    <div class="space-y-2">{lista}</div>
  </div>

  <!-- ── GRÁFICOS ──────────────────────────────────────────── -->
  <div>
    <h3 class="text-lg font-bold text-gray-800 mb-4">Visualizaciones</h3>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
        <p class="text-sm font-bold text-gray-600 mb-3">Score Total</p>
        <div style="height:260px"><canvas id="barChart_{suffix}"></canvas></div>
      </div>
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
        <p class="text-sm font-bold text-gray-600 mb-3">Calidad vs Valoración</p>
        <div style="height:260px"><canvas id="scatterChart_{suffix}"></canvas></div>
      </div>
    </div>
    <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-5 mt-5">
      <p class="text-sm font-bold text-gray-600 mb-3">Radar Multifactor</p>
      <div class="mx-auto" style="height:360px;max-width:580px"><canvas id="radarChart_{suffix}"></canvas></div>
    </div>
  </div>

  <!-- ── TABLA ─────────────────────────────────────────────── -->
  <div>
    <h3 class="text-lg font-bold text-gray-800 mb-4">Tabla Comparativa</h3>
    <div class="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">#</th>
              <th class="px-3 py-3 text-left text-xs font-semibold uppercase tracking-wider">Empresa</th>
              <th class="px-3 py-3 text-left text-xs font-semibold uppercase tracking-wider">Sector</th>
              <th class="px-3 py-3 text-left text-xs font-semibold uppercase tracking-wider">Índice</th>
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">Score</th>
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">ROE</th>
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">ROIC</th>
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">ROA</th>
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">EV/EBITDA</th>
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">P/B</th>
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">Mg.Neto</th>
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">Mg.Op.</th>
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">FCF Mg.</th>
              <th class="px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider">Ret.1Y</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-50">{filas}</tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ── TARJETAS ──────────────────────────────────────────── -->
  <div>
    <h3 class="text-lg font-bold text-gray-800 mb-4">Fichas de Empresa</h3>
    <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4">{tarjetas}</div>
  </div>

  <!-- ── EXPLICACIONES ─────────────────────────────────────── -->
  <div>
    <h3 class="text-lg font-bold text-gray-800 mb-4">¿Por qué han sido seleccionadas?</h3>
    <div class="space-y-3">{explicaciones}</div>
  </div>

<script>
(function(){{
  const labels_{suffix}    = {labels_js};
  const scores_{suffix}    = {scores_js};
  const barColors_{suffix} = {bar_colors_js};
  const radarCfg_{suffix}  = {radar_js};
  const scatterRaw_{suffix}= {scatter_js};

  new Chart(document.getElementById('barChart_{suffix}'), {{
    type: 'bar',
    data: {{ labels: labels_{suffix}, datasets: [{{ label: 'Score', data: scores_{suffix},
      backgroundColor: barColors_{suffix}, borderRadius: 8, borderSkipped: false }}] }},
    options: {{ responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: ctx => ` Score: ${{ctx.raw.toFixed(1)}}` }} }} }},
      scales: {{ y: {{ min:0, max:100, grid:{{ color:'#f1f5f9' }}, ticks:{{ font:{{ size:10 }} }} }},
                 x: {{ grid:{{ display:false }}, ticks:{{ font:{{ size:9 }} }} }} }}
    }}
  }});

  new Chart(document.getElementById('radarChart_{suffix}'), {{
    type: 'radar', data: radarCfg_{suffix},
    options: {{ responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position:'right', labels:{{ font:{{ size:9 }}, boxWidth:10 }} }} }},
      scales: {{ r: {{ min:0, max:100, grid:{{ color:'#e2e8f0' }},
        pointLabels:{{ font:{{ size:10, weight:'600' }} }}, ticks:{{ stepSize:25, font:{{ size:8 }} }} }} }}
    }}
  }});

  new Chart(document.getElementById('scatterChart_{suffix}'), {{
    type: 'scatter',
    data: {{ datasets: [{{ label:'Empresas',
      data: scatterRaw_{suffix}.map(d => ({{x:d.x, y:d.y}})),
      backgroundColor: barColors_{suffix}, pointRadius:8, pointHoverRadius:10 }}] }},
    options: {{ responsive: true, maintainAspectRatio: false,
      plugins: {{ legend:{{ display:false }},
        tooltip: {{ callbacks: {{ label: ctx => {{
          const d = scatterRaw_{suffix}[ctx.dataIndex];
          return ` ${{d.label}} | Cal: ${{d.x.toFixed(1)}} | Val: ${{d.y.toFixed(1)}}`;
        }} }} }} }},
      scales: {{
        x: {{ title:{{ display:true, text:'Score Calidad', font:{{ size:10 }} }}, min:0, max:100, grid:{{ color:'#f1f5f9' }} }},
        y: {{ title:{{ display:true, text:'Score Valoración', font:{{ size:10 }} }}, min:0, max:100, grid:{{ color:'#f1f5f9' }} }}
      }}
    }}
  }});
}})();
</script>"""

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUIR HTML CON DOS PESTAÑAS
# ─────────────────────────────────────────────────────────────────────────────

print("\nGenerando dashboard HTML con dos pestañas...")

fuentes_v1 = ["bme_growth_dataset.csv", "ibex_small_cap_dataset (1).csv",
               "ibex_medium_cap_dataset.csv", "ibex35.xlsx"]
fuentes_v2 = ["bme_growth_dataset.csv", "ibex_small_cap_dataset (1).csv",
               "ibex_medium_cap_dataset.csv", "ibex35_dataset.csv"]

tab_v1 = build_tab_content(top10_v1, scored_v1, "v1", fuentes_v1)
tab_v2 = build_tab_content(top10_v2, scored_v2, "v2", fuentes_v2)

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Análisis Multifactor — Mercado Bursátil Español</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    body {{ font-family:'Inter',sans-serif; }}
    .gradient-text {{
      background: linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    }}
    .tab-panel {{ display:none; }}
    .tab-panel.active {{ display:block; }}
    .tab-btn {{ transition: all .2s; }}
    .tab-btn.active {{
      background: linear-gradient(135deg,#6366f1,#8b5cf6);
      color: white; box-shadow: 0 4px 14px rgba(99,102,241,.35);
    }}
    table {{ border-collapse:separate; border-spacing:0; }}
    ::-webkit-scrollbar {{ width:6px; height:6px; }}
    ::-webkit-scrollbar-track {{ background:#f1f5f9; }}
    ::-webkit-scrollbar-thumb {{ background:#c7d2fe; border-radius:3px; }}
  </style>
</head>
<body class="bg-gradient-to-br from-slate-50 to-indigo-50 min-h-screen">

<!-- HEADER -->
<header class="bg-white border-b border-gray-100 sticky top-0 z-30 shadow-sm">
  <div class="max-w-screen-xl mx-auto px-6 py-4 flex items-center justify-between">
    <div>
      <h1 class="text-xl font-bold gradient-text">Análisis Multifactor Bursátil</h1>
      <p class="text-xs text-gray-400 mt-0.5">Mercado Español · Comparativa de dos versiones del análisis</p>
    </div>
    <div class="flex gap-2 text-xs">
      <span class="bg-indigo-50 text-indigo-600 px-3 py-1.5 rounded-full font-medium">Modelo multifactor</span>
      <span class="bg-purple-50 text-purple-600 px-3 py-1.5 rounded-full font-medium">Score 0–100</span>
    </div>
  </div>
</header>

<!-- TABS NAV -->
<div class="max-w-screen-xl mx-auto px-6 pt-6">
  <div class="flex gap-3 mb-6">
    <button onclick="switchTab('v1')" id="btn-v1"
      class="tab-btn active flex-1 py-3 px-6 rounded-2xl text-sm font-semibold text-gray-600 bg-white shadow-sm border border-gray-100">
      <div class="flex items-center justify-center gap-2">
        <span class="w-2 h-2 rounded-full bg-indigo-400"></span>
        Versión 1 — ibex35.xlsx
      </div>
      <div class="text-xs font-normal mt-0.5 opacity-75">Datos con parsing desde Excel</div>
    </button>
    <button onclick="switchTab('v2')" id="btn-v2"
      class="tab-btn flex-1 py-3 px-6 rounded-2xl text-sm font-semibold text-gray-600 bg-white shadow-sm border border-gray-100">
      <div class="flex items-center justify-center gap-2">
        <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
        Versión 2 — ibex35_dataset.csv
      </div>
      <div class="text-xs font-normal mt-0.5 opacity-75">Datos en formato CSV homogéneo</div>
    </button>
  </div>
</div>

<!-- TAB V1 -->
<div id="panel-v1" class="tab-panel active">
  <main class="max-w-screen-xl mx-auto px-6 pb-10 space-y-8">
    {tab_v1}
    <footer class="text-center text-xs text-gray-400 py-6 border-t border-gray-100">
      <p>Análisis V1 — ibex35.xlsx · Solo con fines informativos · No constituye asesoramiento financiero</p>
    </footer>
  </main>
</div>

<!-- TAB V2 -->
<div id="panel-v2" class="tab-panel">
  <main class="max-w-screen-xl mx-auto px-6 pb-10 space-y-8">
    {tab_v2}
    <footer class="text-center text-xs text-gray-400 py-6 border-t border-gray-100">
      <p>Análisis V2 — ibex35_dataset.csv · Solo con fines informativos · No constituye asesoramiento financiero</p>
    </footer>
  </main>
</div>

<script>
function switchTab(tab) {{
  ['v1','v2'].forEach(t => {{
    document.getElementById('panel-' + t).classList.toggle('active', t === tab);
    document.getElementById('btn-' + t).classList.toggle('active', t === tab);
  }});
}}
</script>

</body>
</html>"""

output_path = BASE + "analysis_dashboard.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ Dashboard con 2 pestañas generado: {output_path}")
print("=" * 60)
