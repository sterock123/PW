#import iniziali#
import math
from pathlib import Path
from datetime import date, timedelta

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from flask_caching import Cache



PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=False,
    external_scripts=[PLOTLY_CDN],
    serve_locally=True,
)
app.title = "Dashboard Resa Uva"


cache = Cache(app.server, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 900})


AREA_FISSA   = "Provincia di Reggio Calabria"
FUSO_ORARIO  = "Europe/Rome"


#funzione info accanto tab principale
def info_i(testo: str, aria_label: str = "Info"):
    return html.Span(
        [html.Span("i", className="i-badge")],
        className="has-tip",
        tabIndex=0,
        **{"data-tip": testo, "aria-label": aria_label}
    )

#helper KPI/UI
def scheda_kpi(titolo, value_id):
    return html.Div(
        [html.P(titolo, className="kpi-title"),
         html.Div(id=value_id, className="kpi-value")],
        className="kpi-card"
    )

#controlli
def formatta_euro(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = 0.0
    segno = "-" if v < 0 else ""
    s = f"{abs(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{segno}€{s}"

def formatta_intero_it(n) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return "0"

#range mensili
INTERVALLO_TEMP_IDEALE_C = {
    1:(5,9),  2:(6,10),  3:(9,14), 4:(10,17),
    5:(16,22), 6:(20,25), 7:(21,27), 8:(22,27),
    9:(18,23), 10:(14,19), 11:(9,14), 12:(6,10)
}
PIOGGIA_IDEALE_MM = {
    1:70, 2:60, 3:60, 4:60, 5:50, 6:40,
    7:30, 8:30, 9:40, 10:60, 11:70, 12:80
}

#funzioni resa mensile
def aggregato_mensile(df: pd.DataFrame) -> pd.DataFrame:
    """Ritorna per (anno, mese): T media, Pioggia totale, UR media."""
    if df.empty:
        return pd.DataFrame(columns=['year','month','t','r','h'])
    d = df.copy()
    d['temperature_2m_mean'] = pd.to_numeric(d.get('temperature_2m_mean'), errors='coerce').interpolate().bfill().ffill()
    d['precipitation_sum']   = pd.to_numeric(d.get('precipitation_sum'),   errors='coerce').fillna(0.0)
    d['relative_humidity_2m_mean'] = pd.to_numeric(d.get('relative_humidity_2m_mean'), errors='coerce').interpolate().bfill().ffill()
    return (d.groupby(['year','month'], as_index=False)
              .agg(t=('temperature_2m_mean','mean'),
                   r=('precipitation_sum','sum'),
                   h=('relative_humidity_2m_mean','mean')))

def calcola_perdita_mese(temp_c, rain_mm, ur_pct, mese: int) -> float:
    """
    Funzione euristica per “perdita mese” (0–20%).
    Penalizza eccessi termici/umidità e insufficienza/abuso di pioggia.
    """
    t = float(temp_c or 0.0); r = float(rain_mm or 0.0); h = float(ur_pct or 0.0)
    perdita = 0.0
    if mese in [12,1,2]:
        if t < 5: perdita += 5
        elif t < 10: perdita += 3
        elif t > 12: perdita += 2
        if r < 80: perdita += 3
        elif r > 150: perdita += 1
    elif mese in [3,4,5]:
        if t < 12: perdita += 3
        elif t > 20: perdita += 2
        if r < 50: perdita += 4
        elif r > 120: perdita += 1
    elif mese in [6,7,8]:
        if t > 35: perdita += 5
        elif t < 25: perdita += 2
        if r < 30: perdita += 3
        elif r > 60: perdita += 1
    else:
        if t < 15: perdita += 2
        elif t > 25: perdita += 2
        if r < 40: perdita += 3
        elif r > 80: perdita += 1
    # Effetti UR
    if mese in [5,6,8,9] and h >= 85.0: perdita += 2
    if mese in [6,7,8] and h <= 35.0: perdita += (2 if t > 32 else 1)
    return float(min(perdita, 20.0))

def calcola_perdita_anno(anno: int, df: pd.DataFrame) -> float:
    """Applica le perdite mensili composte (gen→ott) e ritorna la perdita complessiva % [0–100]."""
    if df.empty:
        return 0.0
    m = aggregato_mensile(df)
    sotto = m[m['year'] == int(anno)]
    if sotto.empty:
        return 0.0
    def _uno(g):
        t = float(pd.to_numeric(g.t, errors='coerce') if pd.notna(g.t) else 0.0)
        r = float(pd.to_numeric(g.r, errors='coerce') if pd.notna(g.r) else 0.0)
        h = float(pd.to_numeric(g.h, errors='coerce') if pd.notna(g.h) else 0.0)
        return min(calcola_perdita_mese(t, r, h, int(g.month)), 20.0)
    fattori = 1 - sotto.apply(lambda g: _uno(g)/100.0, axis=1)
    resa_rel = float(fattori.prod()) if len(fattori) else 1.0
    return float(min(max((1 - resa_rel) * 100.0, 0.0), 100.0))

#pulizia del db
def valida_e_pulisci(df_finale: pd.DataFrame) -> pd.DataFrame:
    df = df_finale.copy()
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df["year"] = pd.to_numeric(df.get("year"), errors='coerce').astype('Int64')
    df["month"] = pd.to_numeric(df.get("month"), errors='coerce').fillna(1).astype(int)
    for c in ['temperature_2m_mean','temperature_2m_max','temperature_2m_min',
              'precipitation_sum','relative_humidity_2m_mean','sunshine_hours']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    prefer = ["date","year","month","temperature_2m_mean","temperature_2m_max","temperature_2m_min",
              "precipitation_sum","relative_humidity_2m_mean","sunshine_hours","time"]
    cols = [c for c in prefer if c in df.columns]
    return df[cols].sort_values("date").reset_index(drop=True)

#misure medie deviazione std
_TEMP_MEDIA_MESE = {
    1:10.0, 2:11.0, 3:13.0, 4:16.5, 5:20.5, 6:24.5,
    7:27.0, 8:27.5, 9:24.0, 10:20.0, 11:15.5, 12:12.0
}

_TEMP_STD_MESE = {m: (2.0 if m in (7,8) else 2.5) for m in range(1,13)}

_ESCURSIONE_MESE = {  # °C
    1:7.0, 2:7.0, 3:7.5, 4:8.0, 5:9.0, 6:10.0,
    7:12.0, 8:11.5, 9:9.5, 10:8.0, 11:7.5, 12:7.0
}

# Pioggia media mensile
_PIOGGIA_MESE_MM = {
    1:75, 2:65, 3:60, 4:55, 5:35, 6:20,
    7:10, 8:15, 9:40, 10:85, 11:95, 12:90
}

# Catena di Markov per occorrenza pioggia:
_PIOGGIA_P01 = {
    1:0.22, 2:0.20, 3:0.18, 4:0.16, 5:0.12, 6:0.08,
    7:0.05, 8:0.06, 9:0.14, 10:0.22, 11:0.26, 12:0.24
}
_PIOGGIA_P11 = {
    1:0.55, 2:0.52, 3:0.48, 4:0.45, 5:0.40, 6:0.35,
    7:0.30, 8:0.32, 9:0.45, 10:0.58, 11:0.62, 12:0.60
}

# umidità relativa media mensile (base)
_UR_MEDIA_MESE = {
    1:72, 2:70, 3:68, 4:66, 5:65, 6:60,
    7:58, 8:60, 9:65, 10:70, 11:72, 12:73
}


#funzione per min e max coerenti e markov
def _tmedia_in_tmin_tmax_coerenti(tmedia: float, mese: int, rng) -> tuple[float, float]:
    amp = _ESCURSIONE_MESE.get(mese, 8.0)
    tmin = tmedia - amp/2 + rng.normal(0, 0.6)
    tmax = tmedia + amp/2 + rng.normal(0, 0.6)
    return tmin, tmax

def _markov_rain_occurrence(ndays: int, p01: float, p11: float, rng) -> np.ndarray:
    occ = np.zeros(ndays, dtype=bool)
    p_wet = p01 / (1 - p11 + p01)
    state = rng.random() < p_wet
    for i in range(ndays):
        if state:
            occ[i] = True
            state = (rng.random() < p11) 
        else:
            occ[i] = False
            state = (rng.random() < p01)
    return occ

def _ammontare_pioggia(ngiorni: int, totale_mm: float, occ: np.ndarray, rng) -> np.ndarray:
    rain = np.zeros(ngiorni, dtype=float)
    nwet = int(occ.sum())
    if nwet == 0 or totale_mm <= 0:
        return rain
    shape = 1.6
    grezzi = rng.gamma(shape, 1.0, size=nwet)
    n_ext = int(max(0, round(0.15 * nwet)))
    if n_ext > 0:
        idx = rng.choice(nwet, size=n_ext, replace=False)
        grezzi[idx] *= rng.uniform(1.8, 3.0, size=n_ext)
    grezzi *= (totale_mm / max(grezzi.sum(), 1e-6))
    rain[np.where(occ)[0]] = grezzi
    return rain

def _ur_da_T_e_pioggia(tmedia, tmax, pioggia, ur_base, rng):
    ur = (ur_base
          + (pioggia > 0) * rng.normal(7, 2)
          - np.maximum(0, tmedia - (np.mean(tmedia) + 3)) * rng.uniform(0.25, 0.5, size=len(tmedia))
          - np.maximum(0, tmax - 32) * rng.uniform(0.15, 0.35, size=len(tmedia)))
    return np.clip(ur, 35, 98)

#funzione ore di sole
def _ore_di_sole(pioggia, ur, rng):
    base = 9.0 - (ur - 60) * 0.07 - (pioggia > 0) * rng.uniform(2.5, 5.0, size=len(ur))
    sole = base + rng.normal(0, 1.3, size=len(ur))
    return np.clip(sole, 0, 13)

def _inietta_ondate_caldo(tmean, tmax, mese, rng):
    nd = len(tmean)
    mask = np.zeros(nd, dtype=bool)
    if mese not in (6,7,8,9) or nd == 0:
        return mask, tmean, tmax
    n_ep = rng.integers(0, 3)
    for _ in range(n_ep):
        durata = int(rng.integers(3, 7))
        start = int(rng.integers(0, max(1, nd - durata)))
        seg = slice(start, start + durata)
        mask[seg] = True
        delta_max = rng.uniform(3.5, 7.0)
        delta_med = rng.uniform(1.2, 2.5)
        tmax[seg] += delta_max
        tmean[seg] += delta_med
    return mask, tmean, tmax

#funzione toogle per layout
def _applica_mods_ai_profili(fatt_pioggia: float,
                             fatt_temp: float,
                             fatt_var_temp: float,
                             fatt_eventi: float):

    
    PIOGGIA_MM = {m: max(0.0, mm * float(fatt_pioggia)) for m, mm in _PIOGGIA_MESE_MM.items()}

    
    delta_C = (float(fatt_temp) - 1.0) * 15.0
    TEMP_MEDIA = {m: _TEMP_MEDIA_MESE[m] + delta_C for m in _TEMP_MEDIA_MESE}

    #varianza temperatura
    TEMP_STD = {m: max(0.1, std * float(fatt_var_temp)) for m, std in _TEMP_STD_MESE.items()}

    #eveti intensi: fattore da propagare nella generazione delle code
    EVENTI_MULT = float(fatt_eventi)

    return TEMP_MEDIA, TEMP_STD, PIOGGIA_MM, EVENTI_MULT

#funzione meteo fin
def simula_meteo(anno_da: int = 2015,
                 anno_a: int | None = None,
                 seed: int | None = 40,
                 includi_future: bool = True,
                 giorni_future: int = 7,
                 fatt_pioggia: float = 1.0,
                 fatt_temp: float = 1.0,
                 fatt_var_temp: float = 1.0,
                 fatt_eventi: float = 1.0) -> pd.DataFrame:

    TEMP_MEDIA_MESE, TEMP_STD_MESE, PIOGGIA_MESE_MM, EVENTI_MULT = _applica_mods_ai_profili(
        fatt_pioggia, fatt_temp, fatt_var_temp, fatt_eventi
    )
    #controllo
    if anno_a is None:
        anno_a = pd.Timestamp.today().year
    rng = np.random.default_rng(seed)
    oggi = pd.Timestamp.today(tz=pytz.timezone(FUSO_ORARIO)).normalize().tz_localize(None)

    righe = []
    phi = 0.60
    sigma_ar = 1.2 * float(fatt_var_temp)

    for anno in range(anno_da, anno_a + 1):
        for mese in range(1, 12+1):
            giorni = pd.date_range(f"{anno}-{mese:02d}-01", periods=1, freq="MS")
            giorni = pd.date_range(giorni[0], giorni[0] + pd.offsets.MonthEnd(1), freq="D")
            if anno == oggi.year and mese == oggi.month:
                giorni = giorni[giorni <= (oggi - pd.Timedelta(days=1))]
                if len(giorni) == 0:
                    continue
            nd = len(giorni)
            if nd == 0:
                continue

            
            base = TEMP_MEDIA_MESE[mese]
            drift = 0.02 * (anno - anno_da)
            anom = np.zeros(nd)
            eps = rng.normal(0, sigma_ar, size=nd)
            for i in range(1, nd):
                anom[i] = phi * anom[i-1] + eps[i]
            tmedia = base + drift + anom + rng.normal(0, TEMP_STD_MESE[mese], size=nd)

            tmin = np.empty(nd); tmax = np.empty(nd)
            for i in range(nd):
                tmin[i], tmax[i] = _tmedia_in_tmin_tmax_coerenti(float(tmedia[i]), mese, rng)


            p01 = _PIOGGIA_P01[mese]; p11 = _PIOGGIA_P11[mese]
            occ = _markov_rain_occurrence(nd, p01, p11, rng)


            rain = np.zeros(nd, dtype=float)
            nwet = int(occ.sum())
            if nwet > 0:
                tot_mm = PIOGGIA_MESE_MM[mese]
                shape = 1.6
                grezzi = rng.gamma(shape, 1.0, size=nwet)
                quota_ext = np.clip(0.15 * EVENTI_MULT, 0.05, 0.40)
                n_ext = int(max(0, round(quota_ext * nwet)))
                if n_ext > 0:
                    idx = rng.choice(nwet, size=n_ext, replace=False)
                    tail_mult = rng.uniform(1.8, 3.0, size=n_ext) * EVENTI_MULT
                    grezzi[idx] *= tail_mult
                grezzi *= (tot_mm / max(grezzi.sum(), 1e-6))
                rain[np.where(occ)[0]] = grezzi


            mask_hw, tmedia, tmax = _inietta_ondate_caldo(tmedia.copy(), tmax.copy(), mese, rng)
            rain[mask_hw] = 0.0


            ur_base = _UR_MEDIA_MESE[mese]
            ur = _ur_da_T_e_pioggia(tmedia, tmax, rain, ur_base, rng)
            sole = _ore_di_sole(rain, ur, rng)
            

            for i, g in enumerate(giorni):
                righe.append({
                    "date": g, "year": g.year, "month": g.month,
                    "temperature_2m_mean": float(tmedia[i]),
                    "temperature_2m_max": float(tmax[i]),
                    "temperature_2m_min": float(tmin[i]),
                    "precipitation_sum":  float(rain[i]),
                    "relative_humidity_2m_mean": float(ur[i]),
                    "sunshine_hours": float(sole[i]),
                    "time": g
                })


    if includi_future and giorni_future > 0:
        inizio_f = oggi
        fine_f = oggi + pd.Timedelta(days=giorni_future)
        idx_future = pd.date_range(start=inizio_f, end=fine_f, freq="D")
        for g in idx_future:
            m = g.month
            base = TEMP_MEDIA_MESE[m]; drift = 0.02 * (g.year - anno_da)
            tm = float(np.random.default_rng(seed + g.day if seed is not None else None)
                       .normal(base + drift, TEMP_STD_MESE[m] * 1.2))
            tmin_f, tmax_f = _tmedia_in_tmin_tmax_coerenti(tm, m, rng)
            p01 = _PIOGGIA_P01[m]; p11 = _PIOGGIA_P11[m]
            pwet = p01 / (1 - p11 + p01)
            wet = (rng.random() < pwet)
            if wet:
                base_gamma = rng.gamma(1.8, 6.0)
                r = float(base_gamma * EVENTI_MULT)
            else:
                r = 0.0
            ur_b = _UR_MEDIA_MESE[m]
            ur_f = float(np.clip(ur_b + (r > 0)*rng.normal(7,2) - max(0, tm - (base + 3))*rng.uniform(0.25,0.5), 35, 98))
            sole_f = float(np.clip(9.0 - (ur_f - 60)*0.07 - (r>0)*rng.uniform(2.5,5.0) + rng.normal(0,1.5), 0, 13))
            
            righe.append({
                "date": g, "year": g.year, "month": g.month,
                "temperature_2m_mean": tm,
                "temperature_2m_max": float(tmax_f),
                "temperature_2m_min": float(tmin_f),
                "precipitation_sum":  r,
                "relative_humidity_2m_mean": ur_f,                
                "sunshine_hours": sole_f,
                "time": g
            })

    return pd.DataFrame(righe)


#scheda simulatore meteo
def layout_storico():
    return html.Div([
        # COLONNA SINISTRA
        html.Div([
            # Sezione simulazione (seed + rigenera)
            html.Div([
                html.Div([
                    html.Span("Dati simulati (Reggio Calabria)"),
                    info_i("Cambia il seed per rigenerare scenari diversi mantenendo coerenza stagionale.")
                ], className="section-title"),
                html.Label("Seed simulazione", className="subtitle"),
                dcc.Input(id='sim-seed', type='number', value=40, min=0, step=1, style={'width':'100%'}),
                html.Button("Rigenera dati", id='sim-rigenera', n_clicks=0,
                            style={'marginTop':'8px','width':'100%'}),
                dcc.Markdown(id='sim-stato',
                             style={'marginTop':6, 'fontSize':12, 'color':'#555'})
            ], className="card"),

            # controlli prsonalizzati di simulazione
        html.Div([
            html.Div([html.Span("Clima simulato"), info_i(
                "Preset rapidi + slider NORMALIZZATI (-1 → +1) per manipolare il dataset simulato:\n"
                "- Pioggia (−1: molto più arido · +1: molto più umido)\n"
                "- Temperatura (−1: più fresco · +1: più caldo)\n"
                "- Variabilità T (−1: meno variabile · +1: molto variabile)\n"
                "- Eventi intensi (−1: meno frequenti/deboli · +1: più frequenti/intensi)"
            )], className="section-title"),

            html.Label("Preset", className="subtitle"),
            dcc.RadioItems(
                id='sim-preset',
                options=[
                    {'label':' Normale',      'value':'norm'},
                    {'label':' Più arida',    'value':'arid'},
                    {'label':' Più umida',    'value':'humid'},
                    {'label':' Più calda',    'value':'warm'},
                    {'label':' Più fresca',   'value':'cool'},
                    {'label':' Estremi ↑',    'value':'extreme'},
                ],
                value='norm', labelStyle={'display':'inline-block','marginRight':'12px'}
            ),

            html.Label("Pioggia (−1 = arido · +1 = umido)", className="subtitle"),
            dcc.Slider(
                id='sim-fattore-pioggia',
                min=-1.0, max=1.0, step=0.05, value=0.0,
                marks={-1:'−1', -0.5:'−0.5', 0:'0', 0.5:'0.5', 1:'1'}
            ),

            html.Label("Temperatura (−1 = fresco · +1 = caldo)", className="subtitle"),
            dcc.Slider(
                id='sim-fattore-temperatura',
                min=-1.0, max=1.0, step=0.01, value=0.0,
                marks={-1:'−1', -0.5:'−0.5', 0:'0', 0.5:'0.5', 1:'1'}
            ),

            html.Label("Variabilità T (−1 = meno · +1 = più)", className="subtitle"),
            dcc.Slider(
                id='sim-variabilita-temp',
                min=-1.0, max=1.0, step=0.05, value=0.0,
                marks={-1:'−1', -0.5:'−0.5', 0:'0', 0.5:'0.5', 1:'1'}
            ),

            html.Label("Eventi intensi (−1 = meno · +1 = più)", className="subtitle"),
            dcc.Slider(
                id='sim-extremes-mult',
                min=-1.0, max=1.0, step=0.05, value=0.0,
                marks={-1:'−1', -0.5:'−0.5', 0:'0', 0.5:'0.5', 1:'1'}
            ),

            dcc.Markdown(
                "_Scala normalizzata: **−1** indica la condizione **meno**, **+1** la condizione **più**. "
                "Il valore **0** corrisponde allo scenario base._",
                style={'fontSize':12,'color':'#555','marginTop':'6px'}
            ),
        ], className="card"),


            #analisi
            html.Div([
                html.Div([
                    html.Span("Impostazioni analisi"),
                    info_i("Seleziona anni e granularità; imposta soglie KPI. "
                           "Puoi mostrare le condizioni ideali (solo vista mensile).")
                ], className="section-title"),

                html.Label("Anni da analizzare (multi-selezione)", className="subtitle"),
                dcc.Dropdown(id='storico-anni', options=[], value=[], multi=True, clearable=False),

                html.Label("Granularità grafici", className="subtitle"),
                dcc.RadioItems(
                    id='storico-granularita',
                    options=[{'label':' Mensile','value':'M'},{'label':' Giornaliera','value':'D'}],
                    value='M',
                    labelStyle={'display':'inline-block','marginRight':'14px'}
                ),

                html.Div("Linea/fascia condizioni ideali (solo mensile)", className="subtitle"),
                dcc.Checklist(
                    id='storico-mostra-ideale',
                    options=[{'label':' Mostra condizioni ideali','value':'on'}],
                    value=['on'],
                    style={'marginBottom':'6px'}
                ),

                html.Div(style={'height':'6px'}),

                html.Div("Soglie KPI", className="subtitle"),
                html.Div([
                    html.Div([
                        html.Div("Giorno piovoso ≥ (mm)"),
                        dcc.Input(id='soglia-pioggia-mm', type='number', value=1.0, min=0, step=0.5,
                                  style={'width':'100%'})
                    ], style={'flex':'1 1 33%'}),

                    html.Div([
                        html.Div("Pioggia intensa ≥ (mm)"),
                        dcc.Input(id='soglia-intensa-mm', type='number', value=20.0, min=0, step=1,
                                  style={'width':'100%'})
                    ], style={'flex':'1 1 33%', 'padding':'0 8px'}),

                    html.Div([
                        html.Div("Giorno molto caldo ≥ (°C)"),
                        dcc.Input(id='soglia-caldo-c', type='number', value=35.0, min=0, step=0.5,
                                  style={'width':'100%'})
                    ], style={'flex':'1 1 33%'}),
                ], style={'display':'flex','gap':'0','marginBottom':'6px'}),

                html.Div([
                    html.Div([
                        html.Div("Umidità relativa (UR) elevata ≥ (%)"),
                        dcc.Input(id='soglia-ur', type='number', value=85.0, min=0, max=100, step=1,
                                  style={'width':'100%'})
                    ], style={'flex':'1 1 33%'}),
                    html.Div([
                        dcc.Checklist(
                            id='storico-mostra-boxplot',
                            options=[{'label':' Mostra boxplot mensili','value':'on'}],
                            value=[]
                        )
                    ], style={'flex':'1 1 66%','paddingLeft':'8px'})
                ], style={'display':'flex','gap':'0'}),

                html.Div(
                    "Le soglie sono personalizzabili; i KPI sommano i giorni sugli anni selezionati.",
                    id='storico-note-soglie',
                    style={'fontSize':12,'color':'#555','marginTop':'6px'}
                )
            ], className="card"),

            
        ], className="col-left"),

        # COLONNA DESTRA
        html.Div([
            # KPI storici
            html.Div([
                html.Div([
                    html.Span("KPI"),
                    info_i("Conteggi totali su anni/periodi selezionati. "
                           "'Giorni molto caldi' usa Tmax se disponibile.")
                ], className="section-title"),
                html.Div([
                    scheda_kpi("Giorni piovosi", 'kpi-giorni-piovosi'),
                    scheda_kpi("Piogge intense", 'kpi-giorni-intensi'),
                    scheda_kpi("Giorni molto caldi (Tmax ≥ soglia)", 'kpi-giorni-caldi'),
                    scheda_kpi("Giorni dannosi (hot ∪ heavy)", 'kpi-giorni-dannosi'),
                    scheda_kpi("Giorni con umidità ≥ soglia", 'kpi-giorni-umidi'),
                    scheda_kpi("Umidità media (ago–set)", 'kpi-ur-ago-set'),
                ], style={'display':'grid','gridTemplateColumns':'repeat(2, minmax(150px, 1fr))','gap':'10px'})
            ], className="card"),
            # estremi
            html.Div([
                html.Div([
                    html.Span("Eventi estremi & sequenze"),
                    info_i("Ondate di caldo (Tmax ≥ soglia, ≥3 gg), longest dry spell "
                           "(giorni consecutivi con pioggia < soglia), # giorni con pioggia intensa.")
                ], className="section-title"),
                html.Div(id="storico-estremi", className="mini-kpis")
            ], className="card"),
            #resa stimata grafico
            html.Div([
                html.Div([
                    html.Span("Resa stimata (cumulata) per mese"),
                    info_i("100% a gennaio; ogni mese applica (1 − perdita_mese). "
                           "Linea mostrata da gennaio a ottobre.")
                ], className="section-title"),
                dcc.Loading(dcc.Graph(id='storico-resa-mensile'), type='dot')
            ], className="card"),

            #grafico temp
            html.Div([
                html.Div([
                    html.Span("Temperature — confronto per anno"),
                    info_i("Linee per ogni anno selezionato. In vista mensile può mostrare fascia/linea ideale.")
                ], className="section-title"),
                dcc.Loading(dcc.Graph(id='storico-linea-temp' ), type='dot')
            ], className="card"),

            #grafico pioggia
            html.Div([
                html.Div([
                    html.Span("Pioggia — confronto per anno"),
                    info_i("Linee per ogni anno selezionato. In vista mensile può mostrare linea di pioggia ideale.")
                ], className="section-title"),
                dcc.Loading(dcc.Graph(id='storico-linea-pioggia'), type='dot')
            ], className="card"),

            

            #anni simili
            html.Div([
                html.Div([
                    html.Span("Annate più simili (ago–set)"),
                    info_i("Confronto su T media, Pioggia totale e UR media di agosto–settembre; "
                           "mostra le 3 annate più vicine alla selezionata.")
                ], className="section-title"),
                html.Ul(id="storico-simili", style={"margin":"0","paddingLeft":"18px"})
            ], className="card"),

            #opzionale boxplot
            html.Div(id='boxplot-card', style={'display':'none'}, children=[
                html.Div([
                    html.Span("Boxplot mensili (storico) con overlay anni selezionati"),
                    info_i("Distribuzione storica mensile con sovrapposizione delle medie degli anni scelti "
                           "e (opzionale) linea ideale.")
                ], className="section-title"),
                dcc.Loading(dcc.Graph(id='storico-boxplot'), type='dot')
            ])
        ], className="col-right")
    ], className="wrap")

#scheda economics
def layout_economico():
    return html.Div([
        # COLONNA SINISTRA
        html.Div([
            #quant e perdite
            html.Div([
                html.Div([
                    html.Span("Quantità & perdite"),
                    info_i("Inserisci ettari e quintali/ha. Seleziona l'anno per stimare le perdite meteo (simulate).")
                ], className="section-title"),
                html.Label("Anno (per perdite)", className="subtitle"),
                dcc.Dropdown(id='eco-anno-perdite', options=[], value=None, clearable=False),
                html.Label("Ettari", className="subtitle"),
                dcc.Input(id='eco-ettari', type='number', value=10, min=0, step=0.1, style={'width':'100%'}),
                html.Label("Quintali/ettaro", className="subtitle"),
                dcc.Input(id='eco-qpe', type='number', value=50, min=0, step=1, style={'width':'100%'}),
                dcc.Markdown(id='eco-kg-sorgente-md',
                             style={'marginTop':6, 'fontSize':12, 'color':'#555'})
            ], className="card"),

            #tipologia raccolta
            html.Div([
                html.Div([
                    html.Span("Modalità raccolta"),
                    info_i("Scegli manuale o meccanica. Parametri e costi si aggiornano.")
                ], className="section-title"),
                dcc.RadioItems(
                    id='eco-mode',
                    options=[{'label':' Manuale', 'value':'manuale'},
                             {'label':' Meccanica', 'value':'meccanica'}],
                    value='manuale',
                    labelStyle={'display':'inline-block','marginRight':'12px'}
                ),
                html.Hr(style={'margin':'8px 0'}),

                #manuale
                html.Div(id='eco-parametri-manuale', children=[
                    html.Label("Raccoglitori (persone)"),
                    dcc.Input(id='eco-pickers', type='number', value=20, min=0, step=1,
                              style={'width':'100%','textAlign':'center'}),
                    html.Label("Ore/giorno per raccoglitore", className="subtitle"),
                    dcc.Input(id='eco-hours', type='number', value=8, min=0, step=0.5, style={'width':'100%'}),
                    html.Label("Produttività per raccoglitore (kg/ora)", className="subtitle"),
                    dcc.Slider(id='eco-prod_kg_h', min=40, max=150, step=5, value=80,
                               marks={40:'40',60:'60',80:'80',100:'100',120:'120',150:'150'}),
                    html.Label("Costo orario per addetto (€) — lordo azienda", className="subtitle"),
                    dcc.Input(id='eco-wage', type='number', value=12.0, min=0, step=0.5, style={'width':'100%'}),
                    html.Label("Costo kit/DPI per lavoratore (una tantum, €)", className="subtitle"),
                    dcc.Input(id='eco-setup_worker', type='number', value=15.0, min=0, step=1, style={'width':'100%'}),
                ], style={'display':'block'}),

                #meccanica
                html.Div(id='eco-parametri-meccanica', children=[
                    html.Label("Velocità vendemmiatrice (t/ora)", className="subtitle"),
                    dcc.Slider(id='eco-mach_tph', min=3, max=15, step=0.5, value=8,
                               marks={3:'3',5:'5',8:'8',10:'10',12:'12',15:'15'}),
                    html.Label("Ore/giorno macchina", className="subtitle"),
                    dcc.Input(id='eco-mach_hours', type='number', value=8, min=0, step=0.5, style={'width':'100%'}),
                    html.Label("Costo macchina (€/ora)", className="subtitle"),
                    dcc.Input(id='eco-mach_cost_h', type='number', value=250, min=0, step=10, style={'width':'100%'}),
                    html.Label("Squadra a supporto (persone)", className="subtitle"),
                    dcc.Input(id='eco-crew', type='number', value=4, min=0, step=1,
                              style={'width':'100%','textAlign':'center'}),
                    html.Label("Costo orario supporto (€) — lordo azienda", className="subtitle"),
                    dcc.Input(id='eco-crew_wage', type='number', value=16.0, min=0, step=0.5, style={'width':'100%'}),
                    html.Label("Malus resa vendemmia meccanica (%)", className="subtitle"),
                    dcc.Slider(id='eco-malus_mech_pct', min=0, max=10, step=0.5, value=3,
                               marks={0:'0%',2:'2%',4:'4%',6:'6%',8:'8%',10:'10%'}),
                    dcc.Markdown(id='eco-malus-caption',
                                 style={'fontSize':12,'color':'#555','marginTop':'6px'})
                ], style={'display':'none'})
            ], className="card"),

            #resa uva e prezzi
            html.Div([
                html.Div([
                    html.Span("Trasformazione & prezzi"),
                    info_i("Resa cantina e prezzi: determinano bottiglie, casse e ricavi.")
                ], className="section-title"),
                html.Label("Resa cantina (%)", className="subtitle"),
                dcc.Slider(id='eco-resa_cantina', min=50, max=85, step=1, value=70,
                           marks={50:'50%',60:'60%',70:'70%',80:'80%',85:'85%'}),
                html.Label("Kg per bottiglia (0.75 L ≈ 0.75 kg)", className="subtitle"),
                dcc.Input(id='eco-kg_per_bott', type='number', value=0.75, min=0.6, max=0.8, step=0.01,
                          style={'width':'100%'}),
                html.Label("Bottiglie per cassa", className="subtitle"),
                dcc.Input(id='eco-bott_per_case', type='number', value=6, min=1, step=1, style={'width':'100%'}),
                html.Label("Percentuale vendibile (%)", className="subtitle"),
                dcc.Input(id='eco-sellable_pct', type='number', value=90, min=70, max=100, step=1,
                          style={'width':'100%'}),
                html.Label("Prezzo per cassa (€)", className="subtitle"),
                dcc.Input(id='eco-prezzo_cassa', type='number', value=24.0, min=0, step=0.5,
                          style={'width':'100%'}),
            ], className="card"),

            #costi
            html.Div([
                html.Div([
                    html.Span("Impostazioni costi, imbottigliamento, materie prime e cantina"),
                    info_i("Gestione cantina, costi fissi, imbottigliamento, produttività, materie prime e costi generici.")
                ], className="section-title"),

                html.Div([
                    html.Label("Fisso stagionale cantina (€)", className="subtitle"),
                        dcc.Input(id='cant-fisso_stagionale', type='number', value=500.0, min=0, step=10,
                                style={'width':'100%'}),
                    html.Label("Gestione cantina", className="subtitle"),
                        dcc.RadioItems(
                            id='cant-gestione',
                            options=[{'label':' Conto proprio','value':'cp'},
                                    {'label':' Conto terzi','value':'ct'}],
                            value='cp',
                            labelStyle={'display':'inline-block','marginRight':'12px'}
                        ),
                    
                    html.Div([
                        
                        
                        html.Label("Imbottigliamento (€/bottiglia) — Conto proprio", className="subtitle"),
                        dcc.Input(id='cant-costo_bott_cp', type='number', value=0.25, min=0, step=0.01,
                                style={'width':'100%'}),
                        html.Label("Produttività imbottigliatrice (bottiglie/ora)", className="subtitle"),
                        dcc.Input(id='cant-bph', type='number', value=1200, min=1, step=1,
                                style={'width':'100%'}),
                        html.Label("Costi per bottiglia (€)", className="subtitle"),
                        html.Div("Bottiglia vuota"),
                        dcc.Input(id='eco-cost_bottle', type='number', value=0.25, min=0, step=0.01,
                                style={'width':'100%'}),
                        html.Div("Tappo"),
                        dcc.Input(id='eco-cost_cork', type='number', value=0.05, min=0, step=0.01,
                                style={'width':'100%'}),
                    ], className="inner-col"),

                    
                    html.Div([
                        html.Label("Imbottigliamento (€/bottiglia) — Conto terzi", className="subtitle"),
                        dcc.Input(id='cant-costo_bott_ct', type='number', value=0.40, min=0, step=0.01,
                                style={'width':'100%'}),
                        html.Label("Ore/giorno imbottigliamento", className="subtitle"),
                        dcc.Input(id='cant-hours_day', type='number', value=6, min=1, step=0.5,
                                style={'width':'100%'}),
                        html.Label("Capsula"),
                        dcc.Input(id='eco-cost_capsule', type='number', value=0.03, min=0, step=0.01,
                                style={'width':'100%'}),
                        html.Label("Etichetta"),
                        dcc.Input(id='eco-cost_label', type='number', value=0.02, min=0, step=0.01,
                                style={'width':'100%'}),
                        html.Label("Cartone (€/cassa)", className="subtitle"),
                        dcc.Input(id='eco-cost_carton', type='number', value=0.40, min=0, step=0.01,
                                style={'width':'100%'}),
                        html.Label("Altri costi imbottigliamento (€/h)", className="subtitle"),
                        dcc.Input(id='eco-other', type='number', value=25.0, min=0, step=1,
                                style={'width':'100%'}),
                        html.Div("(applicati alle ore di imbottigliamento: ore_tot = bottiglie_in_cassa / bph)",
                                style={'fontSize':11,'color':'#666','marginTop':'6px'}),

                    ], className="inner-col"),
                ], style={'display':'grid',
                        'gridTemplateColumns':'1fr 1fr',
                        'gap':'20px'}),

            ], className="card")

        ], className="col-left"),

        # COLONNA DESTRA
        html.Div([
            html.Div([
                #kpi
                html.Div([
                    html.Span("KPI sintetici"),
                    info_i("Riepilogo rapido quantità, tempi e risultati economici per i due scenari.")
                ], className="section-title"),
                html.Div([
                    html.Div([
                        scheda_kpi("Kg netti (base)", 'kpi-kg-netti'),
                        scheda_kpi("Bottiglie in cassa", 'kpi-bott-in-cassa'),
                        scheda_kpi("Casse vendibili", 'kpi-casse'),
                    ], style={'display':'grid','gridTemplateRows':'repeat(3,1fr)','gap':'10px'}),
                    html.Div([
                        scheda_kpi("Giorni vendemmia (Manu)", 'kpi-giorni-manu'),
                        scheda_kpi("Costo vendemmia (Manu)", 'kpi-costo-vendemmia-manu'),
                        scheda_kpi("Giorni imbottigliamento (Manu)", 'kpi-giorni-imb-manu'),
                        scheda_kpi("Costo imbott.+cantina (Manu)", 'kpi-costo-imb-manu'),
                        scheda_kpi("Ricavi (Manu)", 'kpi-ricavi-manu'),
                        scheda_kpi("Margine (Manu)", 'kpi-marg-manu'),
                    ], style={'display':'grid','gridTemplateRows':'repeat(6,auto)','gap':'10px'}),
                    html.Div([
                        scheda_kpi("Giorni vendemmia (Mech)", 'kpi-giorni-mech'),
                        scheda_kpi("Costo vendemmia (Mech)", 'kpi-costo-vendemmia-mech'),
                        scheda_kpi("Giorni imbottigliamento (Mech)", 'kpi-giorni-imb-mech'),
                        scheda_kpi("Costo imbott.+cantina (Mech)", 'kpi-costo-imb-mech'),
                        scheda_kpi("Ricavi (Mech)", 'kpi-ricavi-mech'),
                        scheda_kpi("Margine (Mech)", 'kpi-marg-mech'),
                    ], style={'display':'grid','gridTemplateRows':'repeat(6,auto)','gap':'10px'}),
                ], className="kpi-cols")
            ], className="card"),

            html.Div([
                #previsioni meteo
                html.Div([
                    html.Span("Meteo prossimi giorni"),
                    info_i("Forecast 3–7 giorni (simulato): temperatura media e pioggia giornaliera.")
                ], className="section-title"),
                html.Label("Orizzonte (giorni)", className="subtitle"),
                dcc.Slider(id='fc-orizzonte-giorni', min=3, max=7, step=1, value=7, marks={3:'3',5:'5',7:'7'}),
                dcc.Loading(dcc.Graph(id='eco-fig-forecast'), type='dot'),
            ], className="card"),

            #confronto manuvsmech
            html.Div([                
                html.Div([
                    html.Span("Confronto economico scenari"),
                    info_i("Ricavi, costi e margine a confronto per manuale vs meccanica.")
                ], className="section-title"),
                dcc.Loading(dcc.Graph(id='eco-fig-riepilogo'), type='dot')
            ], className="card"),

            html.Div([
                html.Div([
                    html.Span("Composizione costi — scenario selezionato"),
                    info_i("Waterfall dei contributi di costo e margine sullo scenario attivo.")
                ], className="section-title"),
                dcc.Loading(dcc.Graph(id='eco-fig-cascata'), type='dot')
            ], className="card"),

            html.Div([
                html.Div([
                    html.Span("Sensibilità margine vs prezzo per cassa — scenario selezionato"),
                    info_i("Curva margine al variare del prezzo; mostra anche il prezzo di breakeven.")
                ], className="section-title"),
                dcc.Loading(dcc.Graph(id='eco-fig-sens-prezzo'), type='dot')
            ], className="card"),
        ], className="col-right")
    ], className="wrap")


#layout
app.layout = html.Div([
    dcc.Store(id='store-dati', data=None),

    html.Header(
        html.Div([
            html.Div([
                html.Span("🍇", className="app-badge"),
                html.Span("Dashboard Resa Uva", className="app-title"),
            ], className="app-brand"),
            html.Div(f"{AREA_FISSA}", id="header-area", className="app-subtitle")
        ], className="app-header-inner"),
        className="app-header"
    ),

    dcc.Tabs(id='tabs', value='tab-storico', children=[
        dcc.Tab(label='Analisi Storica', value='tab-storico'),
        dcc.Tab(label='Piano economico e scenari', value='tab-economico'),
    ]),

    html.Div(id='tab-storico-content', children=layout_storico()),
    html.Div(id='tab-economico-content', children=layout_economico(), style={'display':'none'}),
], id='app-root', className='theme-app')

#callback
@app.callback(
    Output('tab-economico-content','style'),
    Output('tab-storico-content','style'),
    Input('tabs','value')
)
def mostra_tab(tab):
    if tab == 'tab-storico':
        return {'display':'none'}, {'display':'block'}
    return {'display':'block'}, {'display':'none'}

def _mappa_norm_a_fattori(s_pioggia: float, s_temp: float, s_varT: float, s_ext: float):
    
    s_pioggia = float(s_pioggia or 0.0)
    s_temp    = float(s_temp    or 0.0)
    s_varT    = float(s_varT    or 0.0)
    s_ext     = float(s_ext     or 0.0)

    def interp(v, vmin, vmax):  # v in −1..+1
        v = max(-1.0, min(1.0, float(v)))
        return vmin + (v + 1.0) * (vmax - vmin) / 2.0

    fatt_pioggia = interp(s_pioggia, 0.5, 1.5)
    fatt_temp    = interp(s_temp,    0.9, 1.1)
    fatt_varT    = interp(s_varT,    0.5, 1.8)
    fatt_eventi  = interp(s_ext,     0.5, 2.0)
    return fatt_pioggia, fatt_temp, fatt_varT, fatt_eventi



@app.callback(
    Output('sim-fattore-pioggia', 'value'),
    Output('sim-fattore-temperatura', 'value'),
    Output('sim-variabilita-temp', 'value'),
    Output('sim-extremes-mult', 'value'),
    Input('sim-preset', 'value'),
    prevent_initial_call=False
)
def sync_slider_con_preset(preset):

    preset = preset or 'norm'

    #normale
    s_pioggia = 0.0  
    s_temp    = 0.0
    s_varT    = 0.0
    s_ext     = 0.0
    #tipologia selezionalta
    if preset == 'arid':
        s_pioggia = -0.6        
    elif preset == 'humid':
        s_pioggia = +0.6      
    elif preset == 'warm':
        s_temp    = +0.6       
    elif preset == 'cool':
        s_temp    = -0.4       
    elif preset == 'extreme':
        s_ext     = +0.6        
        s_varT    = +0.3        

    return s_pioggia, s_temp, s_varT, s_ext



@app.callback(
    Output('store-dati', 'data'),
    Output('sim-stato', 'children'),
    
    Input('sim-seed', 'value'),
    Input('sim-rigenera', 'n_clicks'),
    
    Input('sim-preset', 'value'),              
    Input('sim-fattore-pioggia', 'value'),
    Input('sim-fattore-temperatura', 'value'),
    Input('sim-variabilita-temp', 'value'),
    Input('sim-extremes-mult', 'value'),
    prevent_initial_call=False
)
def genera_o_rigenera_dati(seed_val, _n_clicks,
                           preset, s_pioggia, s_temp, s_varT, s_ext):
    #controllo seed
    try:
        seed = int(seed_val) if seed_val is not None else 40
    except Exception:
        seed = 40

    
    s_pioggia = float(s_pioggia or 0.0)
    s_temp    = float(s_temp    or 0.0)
    s_varT    = float(s_varT    or 0.0)
    s_ext     = float(s_ext     or 0.0)

    
    f_pioggia, f_temp, f_varT, f_ext = _mappa_norm_a_fattori(s_pioggia, s_temp, s_varT, s_ext)

    
    chiave = f"sim_df_seed_{seed}_p{f_pioggia:.2f}_t{f_temp:.2f}_v{f_varT:.2f}_e{f_ext:.2f}"

    #cache velocizz
    df = cache.get(chiave)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
    
        df_raw = simula_meteo(
            anno_da=2015, seed=seed,
            includi_future=True, giorni_future=7,
            fatt_pioggia=f_pioggia,
            fatt_temp=f_temp,
            fatt_var_temp=f_varT,
            fatt_eventi=f_ext
        )
        df = valida_e_pulisci(df_raw)
        cache.set(chiave, df, timeout=900)

    
    data_json = df.to_json(date_format='iso', orient='split')

    
    preset = preset or 'norm'
    if not df.empty:
        dmin = pd.to_datetime(df['date']).min().date()
        dmax = pd.to_datetime(df['date']).max().date()
        righe = len(df)
        anni = sorted(pd.Series(df['year']).dropna().astype(int).unique().tolist())
        testo = (
            f"Dati simulati — seed **{seed}** · preset **{preset}** · "
            f"S: pioggia **{s_pioggia:+.2f}**, temp **{s_temp:+.2f}**, varT **{s_varT:+.2f}**, estremi **{s_ext:+.2f}** · "
            f"Fattori: pioggia **{f_pioggia:.2f}**, temp **{f_temp:.2f}**, varT **{f_varT:.2f}**, estremi **{f_ext:.2f}** · "
            f"{righe:,} righe · **{dmin} → {dmax}** · anni: {', '.join(map(str, anni))}"
        ).replace(",", ".")
    else:
        testo = f"Dati simulati — seed **{seed}** · preset **{preset}** (dataset vuoto?)"

    return data_json, testo


#callback 2
@app.callback(
    Output('storico-anni','options'),
    Output('storico-anni','value'),
    Output('eco-anno-perdite','options'),
    Output('eco-anno-perdite','value'),
    Input('store-dati','data')
)
def popola_anni_per_entrambi(data_json):
    df = pd.read_json(data_json, orient='split') if data_json else pd.DataFrame()
    if df.empty or 'year' not in df.columns:
        return [], [], [], None
    anni = sorted(pd.Series(df['year']).dropna().astype(int).unique().tolist())
    anno_last = int(anni[-1]) if anni else None
    opzioni = [{'label': str(y), 'value': y} for y in anni]
    return opzioni, ([anno_last] if anno_last else []), opzioni, anno_last


#new df?
def df_da_store(data_json: str) -> pd.DataFrame:
    if not data_json:
        return pd.DataFrame(columns=[
            'date','year','month','temperature_2m_mean','temperature_2m_max',
            'precipitation_sum','relative_humidity_2m_mean'
        ])
    df = pd.read_json(data_json, orient='split')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df




#interattività callback
@app.callback(Output('boxplot-card','style'), Input('storico-mostra-boxplot','value'))
def toggle_boxplot(mostra_list):
    return {'display':'block'} if 'on' in (mostra_list or []) else {'display':'none'}


#KPI
@app.callback(
    Output('kpi-giorni-piovosi','children'),
    Output('kpi-giorni-intensi','children'),
    Output('kpi-giorni-caldi','children'),
    Output('kpi-giorni-dannosi','children'),
    Output('kpi-giorni-umidi','children'),
    Output('kpi-ur-ago-set','children'),
    Input('storico-anni','value'),
    Input('soglia-pioggia-mm','value'),
    Input('soglia-intensa-mm','value'),
    Input('soglia-caldo-c','value'),
    Input('soglia-ur','value'),
    State('store-dati','data')
)
def aggiorna_kpi(anni_sel, soglia_pioggia, soglia_intensa, soglia_caldo, soglia_ur, data_json):
    df = df_da_store(data_json)
    anni = [int(y) for y in (anni_sel or [])]
    if not anni or df.empty:
        return ("—","—","—","—","—","—")

    df = df[df['year'].isin(anni)].copy()

    soglia_pioggia = float(soglia_pioggia or 1.0)
    soglia_intensa = float(soglia_intensa or 20.0)
    soglia_caldo   = float(soglia_caldo   or 35.0)
    soglia_ur      = float(soglia_ur      or 85.0)

    df['precipitation_sum'] = pd.to_numeric(df['precipitation_sum'], errors='coerce').fillna(0.0)
    df['relative_humidity_2m_mean'] = pd.to_numeric(df['relative_humidity_2m_mean'], errors='coerce') \
                                        .fillna(method='ffill').fillna(method='bfill')

    temp_col = 'temperature_2m_max' if 'temperature_2m_max' in df.columns else 'temperature_2m_mean'
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce').fillna(method='ffill').fillna(method='bfill')

    giorni_piovosi = int((df['precipitation_sum'] >= soglia_pioggia).sum())
    giorni_intensi = int((df['precipitation_sum'] >= soglia_intensa).sum())
    giorni_caldi   = int((df[temp_col]            >= soglia_caldo  ).sum())
    giorni_dannosi = int(((df[temp_col] >= soglia_caldo) | (df['precipitation_sum'] >= soglia_intensa)).sum())
    giorni_umidi   = int((df['relative_humidity_2m_mean'] >= soglia_ur).sum())

    ur_agoset = df[df['month'].between(8,9)]['relative_humidity_2m_mean'].mean()
    ur_txt = (f"{ur_agoset:.0f}%" if pd.notnull(ur_agoset) else "—")
    return (f"{giorni_piovosi}", f"{giorni_intensi}", f"{giorni_caldi}", f"{giorni_dannosi}", f"{giorni_umidi}", ur_txt)


#temp +boxplot
@app.callback(
    Output('storico-linea-temp','figure'),
    Output('storico-linea-pioggia','figure'),
    Output('storico-boxplot','figure'),
    Input('storico-anni','value'),
    Input('storico-granularita','value'),
    Input('storico-mostra-ideale','value'),
    State('store-dati','data')
)
def aggiorna_linee_e_boxplot(anni_sel, granularita, mostra_ideale, data_json):
    df_tutti = df_da_store(data_json)
    anni = [int(y) for y in (anni_sel or [])]
    if df_tutti.empty or not anni:
        vuoto = go.Figure(layout=dict(title="Seleziona almeno un anno"))
        return vuoto, vuoto, vuoto

    df = df_tutti[df_tutti['year'].isin(anni)].copy()
    mostra_ideale_on = ('on' in (mostra_ideale or [])) and (granularita == 'M')

    #grafici a linee 
    if granularita == 'D':
        fig_temp = go.Figure(); fig_pioggia = go.Figure()
        for y in sorted(anni):
            sub = df[df['year']==y].sort_values('date')
            fig_temp.add_trace(go.Scatter(x=sub['date'], y=sub['temperature_2m_mean'], mode='lines', name=str(y)))
            fig_pioggia.add_trace(go.Scatter(x=sub['date'], y=sub['precipitation_sum'], mode='lines', name=str(y)))
        fig_temp.update_layout(
            title="Temperatura media giornaliera — confronto anni",
            xaxis_title="Data", 
            yaxis_title="°C",
            hovermode='x unified',
            legend=dict(
                orientation='v',   
                x=1, y=1,        
                xanchor='right',
                yanchor='bottom'
            )
        )

        fig_pioggia.update_layout(title="Pioggia giornaliera — confronto anni", xaxis_title="Data", yaxis_title="mm",
                                  hovermode='x unified', legend=dict(orientation='v',x=1, y=1, xanchor='right',yanchor='bottom'))
    else:
        fig_temp = go.Figure(); fig_pioggia = go.Figure()
        mesi = list(range(1,13))
        for y in sorted(anni):
            sub = df[df['year']==y]
            t_m = sub.groupby('month')['temperature_2m_mean'].mean()
            r_m = sub.groupby('month')['precipitation_sum'].sum()
            fig_temp.add_trace(go.Scatter(x=t_m.index, y=t_m.values, mode='lines+markers', name=str(y)))
            fig_pioggia.add_trace(go.Scatter(x=r_m.index, y=r_m.values, mode='lines+markers', name=str(y)))

        if mostra_ideale_on:
            ideale_min = [INTERVALLO_TEMP_IDEALE_C[m][0] for m in mesi]
            ideale_max = [INTERVALLO_TEMP_IDEALE_C[m][1] for m in mesi]
            ideale_mid = [(lo+hi)/2 for lo,hi in zip(ideale_min, ideale_max)]
            fig_temp.add_trace(go.Scatter(x=mesi, y=ideale_min, mode='lines', line=dict(width=0),
                                          name='Range ideale T° (min)', showlegend=False, hoverinfo='skip'))
            fig_temp.add_trace(go.Scatter(x=mesi, y=ideale_max, mode='lines', line=dict(width=0),
                                          fill='tonexty', fillcolor='rgba(16,185,129,0.10)',
                                          name='Range ideale T°',
                                          hovertemplate="Mese %{x}<br>Range ideale: %{y:.1f}°C<extra></extra>"))
            fig_temp.add_trace(go.Scatter(x=mesi, y=ideale_mid, mode='lines', line=dict(width=2, dash='dot'),
                                          name='T° ideale (media)',
                                          hovertemplate="Mese %{x}<br>T° ideale: %{y:.1f}°C<extra></extra>"))
            pioggia_ideale = [PIOGGIA_IDEALE_MM[m] for m in mesi]
            fig_pioggia.add_trace(go.Scatter(x=mesi, y=pioggia_ideale, mode='lines', line=dict(width=2, dash='dot'),
                                             name='Pioggia ideale',
                                             hovertemplate="Mese %{x}<br>Pioggia ideale: %{y:.0f} mm<extra></extra>"))

        fig_temp.update_layout(
            title="Temperatura media mensile — confronto anni",
            xaxis_title="Mese",
            yaxis_title="°C",
            hovermode='x unified',
            legend=dict(
                orientation='v',  
                x=1, y=1,        
                xanchor='right',
                yanchor='bottom'
            )
        )
        fig_pioggia.update_layout(title="Pioggia totale mensile — confronto anni", xaxis_title="Mese", yaxis_title="mm",
                                  hovermode='x unified', legend=dict(orientation='v',x=1, y=1, xanchor='right',yanchor='bottom'))

    #box opzionale
    box = make_subplots(rows=1, cols=2, subplot_titles=("T media — boxplot storico","Pioggia giornaliera — boxplot storico"))
    box.add_trace(go.Box(x=df_tutti['month'], y=df_tutti['temperature_2m_mean'],
                         name='Storico T°', boxmean='sd', showlegend=False), 1,1)
    box.add_trace(go.Box(x=df_tutti['month'], y=df_tutti['precipitation_sum'],
                         name='Storico pioggia', boxmean='sd', showlegend=False), 1,2)
    for y in sorted(anni):
        d = df_tutti[df_tutti['year']==y]
        t_m = d.groupby('month')['temperature_2m_mean'].mean()
        r_m = d.groupby('month')['precipitation_sum'].mean()
        box.add_trace(go.Scatter(x=t_m.index, y=t_m.values, mode='lines+markers', name=f"T° {y}"), 1,1)
        box.add_trace(go.Scatter(x=r_m.index, y=r_m.values, mode='lines+markers', name=f"Pioggia {y}"), 1,2)

    if mostra_ideale_on:
        mesi = list(range(1,13))
        ideale_mid = [(INTERVALLO_TEMP_IDEALE_C[m][0]+INTERVALLO_TEMP_IDEALE_C[m][1])/2 for m in mesi]
        pioggia_ideale = [PIOGGIA_IDEALE_MM[m] for m in mesi]
        box.add_trace(go.Scatter(x=mesi, y=ideale_mid, mode='lines', line=dict(dash='dot', width=2),
                                 name='T° ideale', showlegend=True), 1,1)
        box.add_trace(go.Scatter(x=mesi, y=pioggia_ideale, mode='lines', line=dict(dash='dot', width=2),
                                 name='Pioggia ideale', showlegend=True), 1,2)

    box.update_xaxes(title_text="Mese", row=1, col=1); box.update_yaxes(title_text="°C", row=1, col=1)
    box.update_xaxes(title_text="Mese", row=1, col=2); box.update_yaxes(title_text="mm/giorno", row=1, col=2)
    box.update_layout(title="Boxplot mensili (storico) con overlay anni selezionati")
    return fig_temp, fig_pioggia, box


#resa mensile
@app.callback(
    Output('storico-resa-mensile','figure'),
    Input('storico-anni','value'),
    State('store-dati','data')
)
def resa_mensile_cumulata(anni_sel, data_json):
    df_tutti = df_da_store(data_json)
    anni = [int(y) for y in (anni_sel or [])]

    if df_tutti.empty or not anni:
        return go.Figure(layout=dict(
            title="Resa stimata (cumulata) per mese",
            annotations=[dict(text="Seleziona almeno un anno", x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=14, color="#666"))]
        ))

    m = aggregato_mensile(df_tutti)

    fig = go.Figure()
    for y in sorted(anni):
        sub = m[m['year'] == y].set_index('month')[['t','r','h']]

        
        idx_mesi = pd.Index(range(1, 11), name='month')
        sub = sub.reindex(idx_mesi)

        
        sub['t'] = sub['t'].interpolate().fillna(method='bfill').fillna(method='ffill')
        sub['h'] = sub['h'].interpolate().fillna(method='bfill').fillna(method='ffill')
        sub['r'] = sub['r'].fillna(0.0)

        #perdita menisle
        perdite = []
        for mese, riga in sub.iterrows():
            p = calcola_perdita_mese(riga['t'], riga['r'], riga['h'], int(mese))
            perdite.append(float(min(max(p, 0.0), 20.0)))
        sub['perdita_mese'] = perdite
        sub['fattore'] = 1.0 - (sub['perdita_mese'] / 100.0)
        resa = 100.0 * sub['fattore'].cumprod()

        fig.add_trace(go.Scatter(
            x=sub.index.values, y=resa.values,
            mode='lines+markers', name=str(y),
            hovertemplate="Mese %{x}<br>Resa cumulata: %{y:.1f}%<extra></extra>"
        ))

    
    fig.add_vrect(x0=7.5, x1=9.5, fillcolor="rgba(124,58,237,0.06)",
                  line_width=0, layer="below", annotation_text="Ago–Set", annotation_position="top left") #evidenza fra agosto e settembre

    fig.update_layout(
        title="Resa stimata (cumulata) per mese<br><sup>100% a inizio anno → penalità mensili composte (fino a ottobre)</sup>",
        xaxis_title="Mese", yaxis_title="Resa cumulata (%)",
        hovermode="x unified",
        legend=dict(orientation='h', y=1.04, x=1, xanchor='right'),
        margin=dict(l=60, r=20, t=80, b=50)
    )
    fig.update_xaxes(tickmode='array', tickvals=list(range(1,11)),
                     showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(range=[55, 100], showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    return fig


#helper
def _run_true_piu_lunga(mask: pd.Series) -> int:
    if mask.empty:
        return 0
    m = mask.fillna(False)
    runs = m.astype(int).groupby((~m).cumsum()).sum()
    return int(runs.max() or 0)


#estremi
@app.callback(
    Output("storico-estremi","children"),
    Input("storico-anni","value"),
    Input("soglia-pioggia-mm","value"),
    Input("soglia-intensa-mm","value"),
    Input("soglia-caldo-c","value"),
    State("store-dati","data")
)
def render_estremi(anni_sel, soglia_pioggia, soglia_intensa, soglia_caldo, data_json):
    df = df_da_store(data_json)
    anni = [int(y) for y in (anni_sel or [])]

    def card(lbl, val="—", sub=""):
        return html.Div([html.Div(lbl, className="mini-label"),
                         html.Div(val, className="mini-value"),
                         html.Div([html.Span(sub, className="mini-sub")], className="mini-sub")],
                        className="mini-kpi-card")

    if df.empty or not anni:
        return [card("Ondate di caldo (≥3 gg)"), card("Longest dry spell"), card("Piogge intense")]

    d = df[df["year"].isin(anni)].copy()
    temp_col = 'temperature_2m_max' if 'temperature_2m_max' in d.columns else 'temperature_2m_mean'
    t = pd.to_numeric(d[temp_col], errors="coerce")
    r = pd.to_numeric(d["precipitation_sum"], errors="coerce").fillna(0.0)

    soglia_caldo   = float(soglia_caldo or 35.0)
    soglia_pioggia = float(soglia_pioggia or 1.0)
    soglia_intensa = float(soglia_intensa or 20.0)

    hot_mask = t >= soglia_caldo
    episodi = 0
    giorni_hot_ep = 0
    if not hot_mask.empty:
        gruppi = (hot_mask != hot_mask.shift()).cumsum()
        lunghezze = hot_mask.groupby(gruppi).sum()
        episodi = int((lunghezze >= 3).sum())
        giorni_hot_ep = int(lunghezze[lunghezze >= 3].sum())

    dry_mask = r < soglia_pioggia
    longest_dry = _run_true_piu_lunga(dry_mask)
    giorni_intensi = int((r >= soglia_intensa).sum())

    return [
        card("Ondate di caldo (≥3 gg)", f"{episodi} episodi", f"{giorni_hot_ep} gg totali ≥ {soglia_caldo:.0f}°C"),
        card("Longest dry spell", f"{longest_dry} gg", f"pioggia < {soglia_pioggia:.1f} mm"),
        card("Piogge intense", f"{giorni_intensi} gg", f"≥ {soglia_intensa:.0f} mm/g"),
    ]


@app.callback(
    Output("storico-simili","children"),
    Input("storico-anni","value"),
    State("store-dati","data")
)
def annate_simili(anni_sel, data_json):
    df = df_da_store(data_json)
    anni = [int(y) for y in (anni_sel or [])]
    if df.empty or not anni:
        return [html.Li("Seleziona almeno un anno")]

    anno_riferimento = max(anni)
    mask_as = df["month"].between(8, 9)
    per_anno = (df[mask_as].groupby("year", as_index=False)
                .agg(t=("temperature_2m_mean","mean"),
                     r=("precipitation_sum","sum"),
                     h=("relative_humidity_2m_mean","mean"))).dropna()

    if anno_riferimento not in per_anno["year"].values or len(per_anno) < 2:
        return [html.Li("Dati insufficienti per il match")]

    Z = per_anno.set_index("year").copy()
    for c in ["t","r","h"]:
        std = Z[c].std(ddof=0) or 1.0
        Z[c] = (Z[c] - Z[c].mean()) / std

    target = Z.loc[anno_riferimento, ["t","r","h"]]
    dist = ((Z[["t","r","h"]] - target) ** 2).sum(axis=1).pow(0.5).sort_values()
    top = [y for y in dist.index if y != anno_riferimento][:3]

    elementi = []
    for y in top:
        riga = per_anno[per_anno["year"] == y].iloc[0]
        elementi.append(html.Li(f"{int(y)} — T {riga.t:.1f}°C, R {riga.r:.0f} mm, UR {riga.h:.0f}%"))
    return elementi

#callback economics 
@app.callback(
    Output('eco-parametri-manuale','style'),
    Output('eco-parametri-meccanica','style'),
    Output('eco-malus-caption','children'),
    Input('eco-mode','value'),
    Input('eco-malus_mech_pct','value')
)
def switch_modalita(mode, malus):
    malus = float(malus or 0.0)
    if mode == 'meccanica':
        cap = f"_Nota: la resa di vendemmia meccanica applica un **malus del {malus:.1f}%** rispetto alla manuale._"
        return {'display':'none'}, {'display':'block'}, cap
    else:
        return {'display':'block'}, {'display':'none'}, "_Nessun malus di resa applicato in manuale._"



@app.callback(
    Output('eco-kg-sorgente-md','children'),
    Input('eco-ettari','value'),
    Input('eco-qpe','value'),
    Input('eco-anno-perdite','value'),
    State('store-dati','data')
)
def mostra_sorgente_kg(ettari, qpe, anno_perdite, data_json):
    df = df_da_store(data_json)
    ettari = float(ettari or 0.0); qpe = float(qpe or 0.0)
    kg_lordi = max(0.0, ettari * qpe * 1000.0)
    anno = int(anno_perdite or 0)
    txt_lordi = f"{kg_lordi:,.0f}".replace(",", ".")
    if df[df['year'] == anno].empty:
        return f"Valore inserito: **{txt_lordi} kg** (nessun dato perdite per {anno})"
    perdita_tot = calcola_perdita_anno(anno, df)
    kg_net = kg_lordi * (1.0 - perdita_tot/100.0)
    txt_net   = f"{kg_net:,.0f}".replace(",", ".")
    return (f"Valore inserito: **{txt_lordi} kg** · Perdite stimate {anno} (composte, T°/pioggia/UR): "
            f"**{perdita_tot:.1f}%** → **Resa effettiva: {txt_net} kg**")


#interattività tutto economics
@app.callback(
    Output('kpi-kg-netti','children'),
    Output('kpi-bott-in-cassa','children'),
    Output('kpi-casse','children'),

    Output('kpi-giorni-manu','children'),
    Output('kpi-costo-vendemmia-manu','children'),
    Output('kpi-giorni-imb-manu','children'),
    Output('kpi-costo-imb-manu','children'),
    Output('kpi-ricavi-manu','children'),
    Output('kpi-marg-manu','children'),

    Output('kpi-giorni-mech','children'),
    Output('kpi-costo-vendemmia-mech','children'),
    Output('kpi-giorni-imb-mech','children'),
    Output('kpi-costo-imb-mech','children'),
    Output('kpi-ricavi-mech','children'),
    Output('kpi-marg-mech','children'),

    Output('eco-fig-riepilogo','figure'),
    Output('eco-fig-cascata','figure'),
    Output('eco-fig-sens-prezzo','figure'),

    Input('eco-ettari','value'), Input('eco-qpe','value'), Input('eco-anno-perdite','value'),
    Input('eco-resa_cantina','value'), Input('eco-kg_per_bott','value'),
    Input('eco-bott_per_case','value'), Input('eco-prezzo_cassa','value'), Input('eco-sellable_pct','value'),
    #man
    Input('eco-pickers','value'), Input('eco-hours','value'), Input('eco-prod_kg_h','value'),
    Input('eco-wage','value'), Input('eco-setup_worker','value'),
    #mec
    Input('eco-mach_tph','value'), Input('eco-mach_hours','value'), Input('eco-mach_cost_h','value'),
    Input('eco-crew','value'), Input('eco-crew_wage','value'), Input('eco-malus_mech_pct','value'),
    #materie
    Input('eco-cost_bottle','value'), Input('eco-cost_cork','value'),
    Input('eco-cost_capsule','value'), Input('eco-cost_label','value'), Input('eco-cost_carton','value'),
    Input('eco-other','value'),
    #cantina
    Input('cant-gestione','value'), Input('cant-fisso_stagionale','value'),
    Input('cant-costo_bott_cp','value'), Input('cant-costo_bott_ct','value'),
    Input('cant-bph','value'), Input('cant-hours_day','value'),
    Input('eco-mode','value'),
    


    State('store-dati','data')
)
#calcolo economics totale
def calcola_scenari(ettari, qpe, anno_perdite,
                    resa_cantina_pct, kg_per_bott, bott_per_case, prezzo_cassa, sellable_pct,
                    pickers, hours, prod_kg_h_base, wage, setup_worker_cost,
                    mach_tph, mach_hours, mach_cost_h, crew, crew_wage, malus_mech_pct,
                    cost_bottle, cost_cork, cost_capsule, cost_label, cost_carton,
                    other_costs,
                    cant_mode, cant_fisso, cant_cp, cant_ct, bph, ore_imb_giorno,
                    mode_selezionato,
                    data_json):

    df = df_da_store(data_json)
    anno_perdite = int(anno_perdite or 0)
    perdita_pct = calcola_perdita_anno(anno_perdite, df) if anno_perdite else 0.0

    #quantità
    other_rate = float(other_costs or 0.0)
    resa_cantina = (float(resa_cantina_pct or 70.0) / 100.0)
    kg_per_bott = float(kg_per_bott or 0.75)
    bott_per_case = int(bott_per_case or 6)
    prezzo_cassa = float(prezzo_cassa or 0.0)
    sellable_pct = float(sellable_pct or 0.0)

    ettari = float(ettari or 0.0); qpe = float(qpe or 0.0)
    kg_lordi = max(0.0, ettari * qpe * 1000.0)
    kg_tot_base = kg_lordi * (1 - perdita_pct/100.0)

    malus_mech = float(malus_mech_pct or 0.0) / 100.0
    kg_tot_mech = kg_tot_base * (1.0 - malus_mech)

    
    def confeziona_e_ricavi(kg_tot):
        vino_eff_kg = kg_tot * resa_cantina
        bottiglie_teoriche = vino_eff_kg / kg_per_bott if kg_per_bott > 0 else 0
        bottiglie_vendibili = bottiglie_teoriche * (sellable_pct/100.0)
        casse_int = int(bottiglie_vendibili // bott_per_case) if bott_per_case > 0 else 0
        bott_in_cassa = casse_int * bott_per_case
        ricavi = casse_int * prezzo_cassa
        return bott_in_cassa, casse_int, ricavi

    def costo_materiali(bott_in_cassa, casse_int):
        cb=float(cost_bottle or 0.0); cc=float(cost_cork or 0.0); ccap=float(cost_capsule or 0.0); cl=float(cost_label or 0.0); cct=float(cost_carton or 0.0)
        costo_per_bott = cb + cc + ccap + cl
        return bott_in_cassa * costo_per_bott + casse_int * cct

    
    cant_fisso = float(cant_fisso or 0.0)
    costo_per_bott_imb = float(cant_cp if (cant_mode == 'cp') else cant_ct)
    bph = float(bph or 1.0)
    ore_imb_giorno = float(ore_imb_giorno or 6.0)

    def cantina_giorni_e_costi(bott_in_cassa):
        if bph <= 0 or ore_imb_giorno <= 0:
            ore_tot = 0.0
            giorni = 0
        else:
            ore_tot = bott_in_cassa / bph 
            giorni = math.ceil(ore_tot / ore_imb_giorno)

        costo_var = bott_in_cassa * costo_per_bott_imb
        
        costo_overhead = other_rate * ore_tot
        costo_tot = cant_fisso + costo_var + costo_overhead
        return giorni, costo_tot, ore_tot



    # MANUALE
    bott_manu, casse_manu, ricavi_manu = confeziona_e_ricavi(kg_tot_base)
    mat_cost_manu = costo_materiali(bott_manu, casse_manu)
    giorni_imb_manu, costo_imb_manu_raw, ore_imb_manu = cantina_giorni_e_costi(bott_manu)
    ore_fatturate_manu = giorni_imb_manu * ore_imb_giorno 
    costo_imb_manu = costo_imb_manu_raw + other_rate * max(0.0, ore_fatturate_manu - ore_imb_manu)


    pickers = int(pickers or 0); hours = float(hours or 0.0)
    prod_kg_h_base = float(prod_kg_h_base or 80.0); wage = float(wage or 0.0)
    setup_worker_cost = float(setup_worker_cost or 0.0)
    capacita_giornaliera_manu = prod_kg_h_base * hours * max(pickers, 0)
    giorni_manu_cont = (kg_tot_base / capacita_giornaliera_manu) if capacita_giornaliera_manu > 0 else float('inf')
    giorni_manu_eff = math.ceil(giorni_manu_cont) if math.isfinite(giorni_manu_cont) else 0
    ore_tot_manu = giorni_manu_eff * hours * pickers
    costo_lavoro_manu = ore_tot_manu * wage
    costo_setup_manu = setup_worker_cost * pickers
    altri_manu = 0.0
    costo_vendemmia_manu = costo_lavoro_manu + costo_setup_manu + altri_manu
    costo_tot_manu = costo_vendemmia_manu + mat_cost_manu + costo_imb_manu
    margine_manu   = ricavi_manu - costo_tot_manu

    # MECCANICA
    bott_mech, casse_mech, ricavi_mech = confeziona_e_ricavi(kg_tot_mech)
    mat_cost_mech = costo_materiali(bott_mech, casse_mech)
    giorni_imb_mech, costo_imb_mech_raw, ore_imb_mech = cantina_giorni_e_costi(bott_mech)
    ore_fatturate_mech = giorni_imb_mech * ore_imb_giorno  
    costo_imb_mech = costo_imb_mech_raw + other_rate * max(0.0, ore_fatturate_mech - ore_imb_mech)

    mach_tph = float(mach_tph or 8.0); mach_hours = float(mach_hours or 8.0)
    mach_cost_h = float(mach_cost_h or 250.0); crew = int(crew or 0); crew_wage = float(crew_wage or 16.0)
    capacita_giornaliera_mech = mach_tph * 1000.0 * mach_hours
    giorni_mech_cont = (kg_tot_mech / capacita_giornaliera_mech) if capacita_giornaliera_mech > 0 else float('inf')
    giorni_mech_eff = math.ceil(giorni_mech_cont) if math.isfinite(giorni_mech_cont) else 0
    ore_tot_mech_macchina = giorni_mech_eff * mach_hours
    ore_tot_mech_supporto = ore_tot_mech_macchina * crew
    costo_macchina = mach_cost_h * ore_tot_mech_macchina
    costo_supporto = crew_wage * ore_tot_mech_supporto
    altri_mech = 0.0
    costo_vendemmia_mech = costo_macchina + costo_supporto + altri_mech
    costo_tot_mech = costo_vendemmia_mech + mat_cost_mech + costo_imb_mech
    margine_mech   = ricavi_mech - costo_tot_mech

    # kpi manuale
    kpi_kg = f"{kg_tot_base:,.0f}".replace(",", ".") + " kg"
    kpi_bott_in_cassa = formatta_intero_it(bott_manu)
    kpi_casse = formatta_intero_it(casse_manu)

    kpi_g_manu = f"{giorni_manu_eff} gg" if math.isfinite(giorni_manu_cont) else "—"
    kpi_costo_vendemmia_manu = formatta_euro(costo_vendemmia_manu)
    kpi_g_imb_manu = f"{giorni_imb_manu} gg"
    kpi_costo_imb_manu = formatta_euro(costo_imb_manu)
    kpi_ricavi_manu = formatta_euro(ricavi_manu)
    kpi_marg_manu = formatta_euro(margine_manu)

    kpi_g_mech = f"{giorni_mech_eff} gg" if math.isfinite(giorni_mech_cont) else "—"
    kpi_costo_vendemmia_mech = formatta_euro(costo_vendemmia_mech)
    kpi_g_imb_mech = f"{giorni_imb_mech} gg"
    kpi_costo_imb_mech = formatta_euro(costo_imb_mech)
    kpi_ricavi_mech = formatta_euro(ricavi_mech)
    kpi_marg_mech = formatta_euro(margine_mech)

    
    etichette = ["Ricavi", "Costi totali", "Margine"]
    manu_vals = [ricavi_manu, costo_tot_manu, margine_manu]
    mech_vals = [ricavi_mech, costo_tot_mech, margine_mech]
    col_manu = "rgba(59,130,246,0.85)"  
    col_mech = "rgba(124,58,237,0.85)" 

    fig_riep = go.Figure()
    fig_riep.add_trace(go.Bar(name="Manuale", x=etichette, y=manu_vals,
                              marker_color=col_manu,
                              text=[formatta_euro(v) for v in manu_vals],
                              textposition="outside", textfont=dict(size=12), cliponaxis=False,
                              hovertemplate="%{x}<br>Manuale: %{y:,.0f} €<extra></extra>"))
    fig_riep.add_trace(go.Bar(name="Meccanica", x=etichette, y=mech_vals,
                              marker_color=col_mech,
                              text=[formatta_euro(v) for v in mech_vals],
                              textposition="outside", textfont=dict(size=12), cliponaxis=False,
                              hovertemplate="%{x}<br>Meccanica: %{y:,.0f} €<extra></extra>"))

    delta_marg = margine_mech - margine_manu
    delta_txt  = f"Δ margine Mech vs Manu: {formatta_euro(delta_marg)}"
    delta_color = "#065f46" if delta_marg > 0 else ("#b91c1c" if delta_marg < 0 else "#374151")
    fig_riep.add_annotation(xref="paper", yref="paper", x=1.0, y=1.14, xanchor="right", yanchor="top",
                            text=delta_txt, showarrow=False,
                            font=dict(color=delta_color, size=13),
                            bgcolor="rgba(0,0,0,0.04)", bordercolor="rgba(0,0,0,0.10)",
                            borderwidth=1, borderpad=4)
    fig_riep.add_annotation(xref="paper", yref="paper", x=0, y=1.11, showarrow=False,
                            text=(f"<span style='font-weight:600'>Casse</span> — "
                                  f"Manuale: {casse_manu:,} · Meccanica: {casse_mech:,}").replace(",", "."),
                            font=dict(size=12))
    ymax = max(manu_vals + mech_vals + [0])
    fig_riep.update_layout(
        title="Confronto economico scenari<br><sup>Ricavi · Costi totali · Margine</sup>",
        barmode="group", bargap=0.25,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=50, r=20, t=90, b=40),
        yaxis=dict(title="€", showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                   zeroline=True, zerolinecolor="rgba(0,0,0,0.15)",
                   range=[0, ymax*1.25 if ymax else 1])
    )

    
    if mode_selezionato == 'manuale':
        steps = [
            dict(type='absolute', name='Ricavi', y=ricavi_manu),
            dict(type='relative', name='Vendemmia', y=-costo_vendemmia_manu),
            dict(type='relative', name='Materie',  y=-mat_cost_manu),
            dict(type='relative', name='Cantina+Imb.', y=-costo_imb_manu),
            dict(type='total',    name='Margine', y=margine_manu)
        ]
        costi_sel = costo_tot_manu
        casse_sel = casse_manu
    else:
        steps = [
            dict(type='absolute', name='Ricavi', y=ricavi_mech),
            dict(type='relative', name='Vendemmia', y=-costo_vendemmia_mech),
            dict(type='relative', name='Materie',  y=-mat_cost_mech),
            dict(type='relative', name='Cantina+Imb.', y=-costo_imb_mech),
            dict(type='total',    name='Margine', y=margine_mech)
        ]
        costi_sel = costo_tot_mech
        casse_sel = casse_mech

    labels_w   = [s['name'] for s in steps]
    measures_w = [s['type'] for s in steps]
    values_w   = [s['y'] for s in steps]
    ricavi_val  = float(values_w[0] if labels_w and labels_w[0]=="Ricavi" else 0.0)
    margine_val = float(values_w[-1] if labels_w and labels_w[-1]=="Margine" else 0.0)

    def _pct_su_ricavi(v: float) -> str:
        if ricavi_val <= 0:
            return ""
        return f" ({abs(v)/ricavi_val*100:.0f}%)"

    testi_barre = []
    for lab, v in zip(labels_w, values_w):
        if lab == "Ricavi":
            testi_barre.append(formatta_euro(v))
        elif lab == "Margine":
            pct = f" ({(margine_val/ricavi_val*100):.0f}%)" if ricavi_val > 0 else ""
            testi_barre.append(f"{formatta_euro(v)}{pct}")
        else:
            testi_barre.append(f"{formatta_euro(v)}{_pct_su_ricavi(v)}")

    fig_cascata = go.Figure(go.Waterfall(
        x=labels_w, y=values_w, measure=measures_w,
        text=testi_barre, textposition="outside", textfont=dict(size=12), cliponaxis=False,
        connector=dict(line=dict(dash="dot", color="rgba(0,0,0,0.35)"))),
    )
    fig_cascata.update_traces(
        increasing=dict(marker=dict(color="rgba(16,185,129,0.90)")),   
        decreasing=dict(marker=dict(color="rgba(220,53,69,0.90)")),    
        totals=dict(marker=dict(color="rgba(59,130,246,0.90)"))        
    )
    fig_cascata.add_annotation(xref="paper", yref="paper", x=1.0, y=1.12, xanchor="right", yanchor="top",
                               text=(f"Margine: {formatta_euro(margine_val)}" +
                                     (f" ({margine_val/ricavi_val*100:.0f}% dei ricavi)" if ricavi_val>0 else "")),
                               showarrow=False, font=dict(size=13),
                               bgcolor="rgba(0,0,0,0.04)", bordercolor="rgba(0,0,0,0.10)", borderwidth=1, borderpad=4)
    ymax_val = max([abs(v) for v in values_w] + [ricavi_val, margine_val, 1])
    fig_cascata.update_layout(
        title="Composizione costi — scenario selezionato<br><sup>Ricavi · (−) Costi · Margine</sup>",
        showlegend=False,
        margin=dict(l=60, r=20, t=85, b=40),
        yaxis=dict(title="€", showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                   zeroline=True, zerolinecolor="rgba(0,0,0,0.15)",
                   range=[min(0, min(values_w)*1.15), ymax_val*1.25])
    )

    #breakeven
    breakeven = (costi_sel / casse_sel) if casse_sel > 0 else None
    p_curr = float(prezzo_cassa or 0.0)
    if p_curr > 0:
        p_min = max(0, p_curr * 0.5); p_max = p_curr * 1.5
    else:
        base = breakeven or 20
        p_min = max(0, base * 0.5); p_max = base * 1.5
    steps = 40
    prezzi = [p_min + (p_max - p_min) * i/steps for i in range(steps + 1)]
    curva_marg = [casse_sel * p - costi_sel for p in prezzi]

    y_min = min(curva_marg + [0]); y_max = max(curva_marg + [0])
    pad = max(1.0, 0.08 * (y_max - y_min if y_max != y_min else (abs(y_max) + 1)))
    y0 = y_min - pad; y1 = y_max + pad

    shapes = []
    if breakeven is not None:
        bx = max(p_min, min(p_max, breakeven))
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=p_min, x1=bx, y0=0, y1=1,
                           fillcolor="rgba(220,53,69,0.08)", line=dict(width=0)))
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=bx, x1=p_max, y0=0, y1=1,
                           fillcolor="rgba(16,185,129,0.08)", line=dict(width=0)))

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=prezzi, y=curva_marg, mode='lines', name='Margine totale',
                                  hovertemplate="Prezzo: €%{x:.2f}<br>Margine: €%{y:,.0f}<extra></extra>",
                                  line=dict(width=3)))
    fig_sens.add_hline(y=0, line_dash='dot', line_width=1)
    if breakeven is not None:
        fig_sens.add_vline(x=breakeven, line_dash='dot', line_width=1.5,
                           annotation_text=f"Breakeven €{breakeven:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
                           annotation_position="top right")
    if p_curr > 0 and casse_sel > 0:
        m0 = casse_sel * p_curr - costi_sel
        fig_sens.add_trace(go.Scatter(x=[p_curr], y=[m0], mode='markers', name='Prezzo attuale',
                                      marker=dict(size=9, symbol='circle'),
                                      hovertemplate="Prezzo attuale: €%{x:.2f}<br>Margine: €%{y:,.0f}<extra></extra>"))
        fig_sens.add_annotation(x=p_curr, y=m0, xanchor='left', yanchor='bottom',
                                text=f"€{p_curr:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
                                showarrow=True, arrowhead=2, arrowsize=1, ax=24, ay=-18)

    fig_sens.add_annotation(xref="paper", yref="paper", x=0, y=1.07, showarrow=False,
                            text=f"<span style='font-weight:600'>Pendenza = {casse_sel:,} casse</span> · "
                                 f"Margine(p) = {casse_sel:,} × p − €{costi_sel:,.0f}".replace(",", "."))
    fig_sens.update_layout(
        title="Sensibilità margine vs prezzo per cassa<br><sup>(scenario selezionato)</sup>",
        xaxis_title='Prezzo €/cassa', yaxis_title='Margine totale (€)',
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right'),
        shapes=shapes,
        margin=dict(l=60, r=20, t=72, b=50)
    )
    fig_sens.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig_sens.update_yaxes(range=[y0, y1], showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    return (
        kpi_kg, kpi_bott_in_cassa, kpi_casse,
        kpi_g_manu, kpi_costo_vendemmia_manu, kpi_g_imb_manu, kpi_costo_imb_manu, kpi_ricavi_manu, kpi_marg_manu,
        kpi_g_mech, kpi_costo_vendemmia_mech, kpi_g_imb_mech, kpi_costo_imb_mech, kpi_ricavi_mech, kpi_marg_mech,
        fig_riep, fig_cascata, fig_sens
    )


#
@app.callback(
    Output('eco-fig-forecast','figure'),
    Input('fc-orizzonte-giorni','value'),
    Input('store-dati','data')
)
def figura_forecast(orizzonte_giorni, data_json):
    df = df_da_store(data_json)
    if df.empty:
        return go.Figure(layout=dict(title="Nessun dato disponibile"))

    n = int(orizzonte_giorni or 7)
    n = max(3, min(n, 7))

    tz = pytz.timezone(FUSO_ORARIO)
    oggi_locale = pd.Timestamp.now(tz=tz).normalize().tz_localize(None)

    s = pd.to_datetime(df['date'], errors='coerce')
    fine = oggi_locale + pd.Timedelta(days=n)
    dfw = df[(s >= oggi_locale) & (s < fine)].copy().sort_values('date')

    if dfw.empty:
        return go.Figure(layout=dict(title="Nessun forecast disponibile nel periodo selezionato"))

    t = pd.to_numeric(dfw['temperature_2m_mean'], errors='coerce').fillna(0)
    r = pd.to_numeric(dfw['precipitation_sum'], errors='coerce').fillna(0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=dfw['date'], y=r, name="Pioggia (mm)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=dfw['date'], y=t, mode='lines+markers', name="Temperatura (°C)"), secondary_y=True)
    fig.update_yaxes(title_text="Pioggia (mm)", secondary_y=False)
    fig.update_yaxes(title_text="Temperatura (°C)", secondary_y=True)
    fig.update_layout(title=f"Forecast prossimi {n} giorni (simulato)", hovermode="x unified",
                      legend=dict(orientation='h', y=1.02, x=1, xanchor='right'))
    return fig


#avvio app finale
if __name__ == '__main__':
    app.run(debug=False, port=8070)
