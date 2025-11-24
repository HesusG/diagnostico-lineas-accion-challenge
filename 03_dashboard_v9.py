"""
Dashboard SERVQUAL v9 - Fundación Teletón
==========================================
- Mapa con tab "Normalizado" (ponderado por volumen)
- Hover con NPS, Satisfacción, Calidad, SERVQUAL
- Nota de advertencia sobre rankings
- Áreas de oportunidad mejorada (compacta, colores)

Ejecutar: streamlit run 03_dashboard_v9.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import statsmodels.api as sm
import json
import urllib.request
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SERVQUAL Teletón", page_icon="💜", layout="wide")

# =============================================================================
# PALETA Y ESCALAS
# =============================================================================
PALETA = {
    'morado_primario': '#5A0077', 'morado_profundo': '#3B0050', 'morado_claro': '#A45DB4',
    'amarillo': '#F9C400', 'naranja': '#FF7A21', 'turquesa': '#009EC6',
    'rojo': '#D9351A', 'blanco': '#FAFAFA', 'gris_claro': '#F2F2F2',
    'gris_medio': '#9E9E9E', 'gris_apagado': '#BDBDBD', 'gris_oscuro': '#424242',
}

ESCALA_NPS = ['#FFF4C2', '#FFE88F', '#FFD054', '#FFB028', '#FF7A21', '#D9351A']
ESCALA_PURPURA = ['#F3E6F7', '#D9B6E3', '#B987CE', '#8C4DAE', '#5A0077', '#3B0050']
ESCALA_DIVERGENTE = ['#007F8A', '#22A7A6', '#83D3CD', '#E7E0F0', '#A45DB4', '#5A0077']
ESCALA_VOLUMEN = ['#E8F5E9', '#A5D6A7', '#66BB6A', '#43A047', '#2E7D32', '#1B5E20']  # Verde para volumen
CATEGORICA_BRAND = ['#5A0077', '#F9C400', '#FF7A21', '#009EC6', '#D43F8D', '#3F51B5']
CATEGORICA_MONO = ['#5A0077', '#7B1FA2', '#9C27B0', '#AB47BC', '#BA68C8', '#CE93D8']

# GeoJSON de México
@st.cache_data
def load_mexico_geojson():
    url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode())
    except:
        return None

MEXICO_GEOJSON = load_mexico_geojson()

ESTADO_GEOJSON_MAP = {
    'Aguascalientes': 'Aguascalientes', 'Baja California': 'Baja California',
    'Baja California Sur': 'Baja California Sur', 'Campeche': 'Campeche',
    'Chiapas': 'Chiapas', 'Chihuahua': 'Chihuahua', 'Ciudad de México': 'Distrito Federal',
    'Coahuila': 'Coahuila de Zaragoza', 'Colima': 'Colima', 'Durango': 'Durango',
    'Estado de México': 'México', 'Guanajuato': 'Guanajuato', 'Guerrero': 'Guerrero',
    'Hidalgo': 'Hidalgo', 'Jalisco': 'Jalisco', 'Michoacán': 'Michoacán de Ocampo',
    'Morelos': 'Morelos', 'Nayarit': 'Nayarit', 'Nuevo León': 'Nuevo León',
    'Oaxaca': 'Oaxaca', 'Puebla': 'Puebla', 'Querétaro': 'Querétaro',
    'Quintana Roo': 'Quintana Roo', 'San Luis Potosí': 'San Luis Potosí',
    'Sinaloa': 'Sinaloa', 'Sonora': 'Sonora', 'Tabasco': 'Tabasco',
    'Tamaulipas': 'Tamaulipas', 'Tlaxcala': 'Tlaxcala', 'Veracruz': 'Veracruz de Ignacio de la Llave',
    'Yucatán': 'Yucatán', 'Zacatecas': 'Zacatecas'
}

# CSS
st.markdown(f"""
<style>
    .main {{background-color: {PALETA['blanco']} !important;}}
    .kpi-card {{
        background: linear-gradient(135deg, {PALETA['gris_claro']} 0%, {PALETA['blanco']} 100%);
        border: 1px solid #e0e0e0; border-radius: 12px; padding: 20px; text-align: center;
        box-shadow: 0 2px 8px rgba(90,0,119,0.1);
    }}
    .kpi-value {{font-size: 2.5rem; font-weight: 700; color: {PALETA['morado_primario']}; margin: 10px 0;}}
    .kpi-label {{font-size: 0.9rem; color: {PALETA['gris_oscuro']}; text-transform: uppercase;}}
    .section-title {{
        font-size: 1.3rem; font-weight: 600; color: {PALETA['morado_profundo']};
        margin: 30px 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid {PALETA['morado_claro']};
    }}
    .insight-card {{
        background-color: {PALETA['gris_claro']}; border-radius: 8px;
        padding: 15px; margin: 10px 0; border-left: 4px solid {PALETA['morado_primario']};
    }}
    .stat-box {{
        background-color: {PALETA['gris_claro']}; border: 1px solid #e0e0e0;
        border-radius: 6px; padding: 12px; font-family: monospace; font-size: 0.85rem;
    }}
    .warning-box {{
        background-color: #fff3cd; border: 1px solid #ffc107;
        border-radius: 8px; padding: 15px; margin: 15px 0; color: #856404;
    }}
    .note-box {{
        background-color: #e3f2fd; border: 1px solid #90caf9;
        border-radius: 6px; padding: 10px; margin: 10px 0; color: #1565c0; font-size: 0.85rem;
    }}
    .rank-item {{
        background: white; border: 1px solid #eee; border-radius: 6px;
        padding: 8px 12px; margin: 4px 0; font-size: 0.9rem;
    }}
    .metrics-section {{
        margin-top: 30px; padding-top: 20px;
    }}
    .opp-card {{
        background: white; border-radius: 8px; padding: 12px;
        border-left: 4px solid {PALETA['morado_primario']}; margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .opp-item {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 6px 0; border-bottom: 1px solid #f0f0f0;
    }}
    .opp-item:last-child {{border-bottom: none;}}
</style>
""", unsafe_allow_html=True)

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv('data/teleton_enriched.csv', encoding='utf-8-sig')
    df['Giro_display'] = df['Giro'].replace({
        'Teletón (Grupos internos de la Fundación)': 'Teletón',
        'Gubernamental': 'Gobierno'
    })
    df['estado_geojson'] = df['Estado_limpio'].map(ESTADO_GEOJSON_MAP)
    return df

df = load_data()
vars_servqual = ['AT_1', 'AT_2', 'FI_1', 'FI_2', 'FI_3', 'R_1', 'R_2', 'R_3', 'E_1', 'E_2', 'E_3', 'E_4']
vars_scores = ['score_tangibles', 'score_fiabilidad', 'score_responsiveness', 'score_empatia']

# =============================================================================
# HEADER
# =============================================================================
col_title, col_toggle = st.columns([4, 1])
with col_title:
    st.markdown("# 💜 Dashboard de Calidad de Servicio")
    st.markdown("**Fundación Teletón** | Modelo SERVQUAL")
with col_toggle:
    st.markdown("<br>", unsafe_allow_html=True)
    color_mode = st.toggle("🎨 Brand", value=False)

COLORES_CAT = CATEGORICA_BRAND if color_mode else CATEGORICA_MONO
ESCALA_MAPA = ESCALA_NPS if color_mode else ESCALA_PURPURA

st.markdown("---")

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("## 🎛️ Filtros")
giro_sel = st.sidebar.selectbox("Organización", ['Todos'] + sorted(df['Giro_display'].dropna().unique().tolist()))
estado_sel = st.sidebar.selectbox("Estado", ['Todos'] + sorted(df['Estado_limpio'].dropna().unique().tolist()))
region_sel = st.sidebar.selectbox("Región", ['Todos'] + sorted(df['region_simplificada'].dropna().unique().tolist()))
antiguedad_sel = st.sidebar.selectbox("Antigüedad", ['Todos', 'Nuevo', 'Establecido', 'Veterano'])

df_f = df.copy()
if giro_sel != 'Todos': df_f = df_f[df_f['Giro_display'] == giro_sel]
if estado_sel != 'Todos': df_f = df_f[df_f['Estado_limpio'] == estado_sel]
if region_sel != 'Todos': df_f = df_f[df_f['region_simplificada'] == region_sel]
if antiguedad_sel != 'Todos': df_f = df_f[df_f['antiguedad_grupo'] == antiguedad_sel]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 {len(df_f):,}** de {len(df):,}")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2 = st.tabs(["📈 Visión Operativa", "🔬 Análisis Estadístico"])

with tab1:
    # =========================================================================
    # KPIs
    # =========================================================================
    st.markdown('<p class="section-title">📊 Indicadores Clave</p>', unsafe_allow_html=True)
    nps_counts = df_f['nps_categoria'].value_counts(normalize=True) * 100
    nps_score = nps_counts.get('Promotor', 0) - nps_counts.get('Detractor', 0)
    sat_pct = (df_f['D_1'].mean() / 10) * 100 if len(df_f) > 0 else 0
    cal_pct = (df_f['C_1'].mean() / 5) * 100 if len(df_f) > 0 else 0
    serv_pct = (df_f['score_servqual_total'].mean() / 5) * 100 if len(df_f) > 0 else 0
    info_pct = (df_f['INFO'].mean() / 10) * 100 if len(df_f) > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(f'<div class="kpi-card"><div style="font-size:1.5rem">🎯</div><div class="kpi-value">{nps_score:.0f}</div><div class="kpi-label">NPS</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div style="font-size:1.5rem">😊</div><div class="kpi-value">{sat_pct:.0f}%</div><div class="kpi-label">Satisfacción</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div style="font-size:1.5rem">⭐</div><div class="kpi-value">{cal_pct:.0f}%</div><div class="kpi-label">Calidad</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-card"><div style="font-size:1.5rem">📋</div><div class="kpi-value">{serv_pct:.0f}%</div><div class="kpi-label">SERVQUAL</div></div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="kpi-card"><div style="font-size:1.5rem">📰</div><div class="kpi-value">{info_pct:.0f}%</div><div class="kpi-label">Info</div></div>', unsafe_allow_html=True)

    # SECCIÓN RESTAURADA: ¿Qué significan estos indicadores?
    with st.expander("ℹ️ ¿Qué significan estos indicadores?"):
        st.markdown("""
        | Indicador | Descripción | Escala Original | Interpretación |
        |-----------|-------------|-----------------|----------------|
        | **NPS** | Net Promoter Score = %Promotores - %Detractores | -100 a +100 | >0 bueno, >50 excelente |
        | **Satisfacción** | Nivel de satisfacción general normalizado | 1-10 → 0-100% | >80% muy satisfecho |
        | **Calidad** | Percepción de calidad del servicio | 1-5 → 0-100% | >80% alta calidad |
        | **SERVQUAL** | Índice compuesto de calidad (Parasuraman et al.) | 1-5 → 0-100% | >80% excelente servicio |
        | **Info** | Qué tan informados se sienten los benefactores | 1-10 → 0-100% | >70% bien informados |
        """)

    # =========================================================================
    # RESUMEN DE MÉTRICAS (con padding, mediana, IQR)
    # =========================================================================
    st.markdown('<div class="metrics-section"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📋 Resumen de Métricas</p>', unsafe_allow_html=True)

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.markdown("**Indicadores de Resultado**")
        def calc_stats(series):
            return {
                'Media': series.mean(),
                'Mediana': series.median(),
                'Desv.Est.': series.std(),
                'IQR': f"{series.quantile(0.25):.1f}-{series.quantile(0.75):.1f}"
            }
        outcome_data = []
        for var, nombre, escala in [('D_1', 'Satisfacción', '1-10'), ('NPS', 'Recomendación', '1-10'),
                                     ('C_1', 'Calidad', '1-5'), ('INFO', 'Información', '1-10')]:
            stats = calc_stats(df_f[var].dropna())
            outcome_data.append({'Métrica': nombre, 'Media': f"{stats['Media']:.2f}", 'Mediana': f"{stats['Mediana']:.1f}",
                                'Desv.Est.': f"{stats['Desv.Est.']:.2f}", 'IQR (Q1-Q3)': stats['IQR'], 'Escala': escala})
        st.dataframe(pd.DataFrame(outcome_data), hide_index=True, use_container_width=True)

    with col_res2:
        st.markdown("**Dimensiones SERVQUAL** *(escala 1-5)*")
        serv_data = []
        for var, nombre in [('score_tangibles', 'Tangibles'), ('score_fiabilidad', 'Fiabilidad'),
                            ('score_responsiveness', 'Responsiveness'), ('score_empatia', 'Empatía')]:
            stats = calc_stats(df_f[var].dropna())
            serv_data.append({'Dimensión': nombre, 'Media': f"{stats['Media']:.2f}", 'Mediana': f"{stats['Mediana']:.2f}",
                             'Desv.Est.': f"{stats['Desv.Est.']:.2f}", 'IQR (Q1-Q3)': stats['IQR']})
        # Total (sin **)
        stats_total = calc_stats(df_f['score_servqual_total'].dropna())
        serv_data.append({'Dimensión': 'TOTAL', 'Media': f"{stats_total['Media']:.2f}", 'Mediana': f"{stats_total['Mediana']:.2f}",
                         'Desv.Est.': f"{stats_total['Desv.Est.']:.2f}", 'IQR (Q1-Q3)': stats_total['IQR']})
        st.dataframe(pd.DataFrame(serv_data), hide_index=True, use_container_width=True)

    # =========================================================================
    # MAPA CHOROPLETH CON FONDO
    # =========================================================================
    st.markdown('<p class="section-title">🗺️ Distribución Geográfica</p>', unsafe_allow_html=True)

    estado_stats = df_f.groupby('Estado_limpio').agg({
        'D_1': 'mean', 'C_1': 'mean', 'score_servqual_total': 'mean',
        'lat': 'first', 'long': 'first', 'estado_geojson': 'first', 'Estado_limpio': 'count'
    }).rename(columns={'Estado_limpio': 'n'}).reset_index()

    def calc_nps(estado):
        sub = df_f[df_f['Estado_limpio'] == estado]
        if len(sub) == 0: return 0
        c = sub['nps_categoria'].value_counts(normalize=True) * 100
        return c.get('Promotor', 0) - c.get('Detractor', 0)
    estado_stats['NPS_Score'] = [calc_nps(e) for e in estado_stats['Estado_limpio']]

    # Calcular NPS normalizado (ponderado por confianza basada en n)
    # Usamos: NPS_norm = NPS * factor_confianza, donde factor_confianza = 1 - 1/sqrt(n)
    # Esto penaliza estados con pocas respuestas
    n_total = estado_stats['n'].sum()
    estado_stats['confianza'] = 1 - 1 / np.sqrt(estado_stats['n'].clip(lower=1))
    estado_stats['NPS_normalizado'] = estado_stats['NPS_Score'] * estado_stats['confianza']
    estado_stats['Satisfaccion_norm'] = estado_stats['D_1'] * estado_stats['confianza']

    # 5 tabs: NPS, Satisfacción, Calidad, SERVQUAL, Normalizado
    map_tabs = st.tabs(["🎯 NPS", "😊 Satisfacción", "⭐ Calidad", "📋 SERVQUAL", "⚖️ Normalizado"])

    def crear_mapa_choropleth(data, col, titulo, custom_hover=True):
        """Crear mapa choropleth con fondo geográfico visible y hover completo"""
        if MEXICO_GEOJSON:
            # Preparar hover_data con todas las métricas
            if custom_hover:
                hover_template = (
                    "<b>%{customdata[0]}</b><br>" +
                    "NPS: %{customdata[1]:.0f}<br>" +
                    "Satisfacción: %{customdata[2]:.1f}<br>" +
                    "Calidad: %{customdata[3]:.1f}<br>" +
                    "SERVQUAL: %{customdata[4]:.2f}<br>" +
                    "Respuestas: %{customdata[5]}<extra></extra>"
                )
                customdata = np.column_stack([
                    data['Estado_limpio'],
                    data['NPS_Score'],
                    data['D_1'],
                    data['C_1'],
                    data['score_servqual_total'],
                    data['n']
                ])
            else:
                hover_template = None
                customdata = None

            fig = px.choropleth_mapbox(
                data,
                geojson=MEXICO_GEOJSON,
                locations='estado_geojson',
                featureidkey='properties.name',
                color=col,
                color_continuous_scale=ESCALA_MAPA,
                mapbox_style="carto-positron",
                center={"lat": 23.6345, "lon": -102.5528},
                zoom=4,
                opacity=0.7
            )
            if custom_hover:
                fig.update_traces(customdata=customdata, hovertemplate=hover_template)
            fig.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
                paper_bgcolor='white'
            )
        else:
            # Fallback sin GeoJSON
            fig = go.Figure(go.Scattergeo(
                lat=data['lat'], lon=data['long'], mode='markers',
                marker=dict(size=15, color=data[col], colorscale=ESCALA_MAPA, showscale=False),
                text=data['Estado_limpio'], hoverinfo='text'
            ))
            fig.update_geos(scope='north america', center=dict(lat=23.6, lon=-102.5), projection_scale=4)
            fig.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0))
        return fig

    def crear_mapa_volumen(data):
        """Mapa de volumen de respuestas (verde)"""
        if MEXICO_GEOJSON:
            hover_template = (
                "<b>%{customdata[0]}</b><br>" +
                "Respuestas: %{customdata[1]}<br>" +
                "NPS: %{customdata[2]:.0f}<br>" +
                "Confianza: %{customdata[3]:.0%}<extra></extra>"
            )
            customdata = np.column_stack([
                data['Estado_limpio'],
                data['n'],
                data['NPS_Score'],
                data['confianza']
            ])
            fig = px.choropleth_mapbox(
                data,
                geojson=MEXICO_GEOJSON,
                locations='estado_geojson',
                featureidkey='properties.name',
                color='n',
                color_continuous_scale=ESCALA_VOLUMEN,
                mapbox_style="carto-positron",
                center={"lat": 23.6345, "lon": -102.5528},
                zoom=4,
                opacity=0.7
            )
            fig.update_traces(customdata=customdata, hovertemplate=hover_template)
            fig.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_colorbar=dict(title="n"),
                paper_bgcolor='white'
            )
        else:
            fig = go.Figure()
        return fig

    def mostrar_ranking(data, col, show_n=False):
        """Ranking con cards estilizadas - sin (n=X) por defecto"""
        sorted_d = data.sort_values(col, ascending=False).head(8)
        for i, (_, r) in enumerate(sorted_d.iterrows(), 1):
            medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else f"{i}."
            color = PALETA['morado_primario'] if r[col] > data[col].median() else PALETA['gris_apagado']
            n_text = f'<small style="color:#999; margin-left:5px">(n={int(r["n"])})</small>' if show_n else ''
            st.markdown(f'''
            <div class="rank-item">
                <b>{medal}</b> {r["Estado_limpio"]}:
                <span style="color:{color}; font-weight:bold; float:right">{r[col]:.1f}</span>
                {n_text}
            </div>
            ''', unsafe_allow_html=True)

    def mostrar_ranking_normalizado(data):
        """Ranking con NPS normalizado mostrando n y confianza"""
        sorted_d = data.sort_values('NPS_normalizado', ascending=False).head(8)
        for i, (_, r) in enumerate(sorted_d.iterrows(), 1):
            medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else f"{i}."
            color = PALETA['morado_primario'] if r['NPS_normalizado'] > 0 else PALETA['gris_apagado']
            st.markdown(f'''
            <div class="rank-item">
                <b>{medal}</b> {r["Estado_limpio"]}
                <span style="color:{color}; font-weight:bold; float:right">{r["NPS_normalizado"]:.1f}</span><br>
                <small style="color:#666">NPS: {r["NPS_Score"]:.0f} × Conf: {r["confianza"]:.0%} (n={int(r["n"])})</small>
            </div>
            ''', unsafe_allow_html=True)

    with map_tabs[0]:
        c1, c2 = st.columns([2.5, 1])
        with c1: st.plotly_chart(crear_mapa_choropleth(estado_stats, 'NPS_Score', 'NPS'), use_container_width=True)
        with c2:
            st.markdown("**Top Estados**")
            mostrar_ranking(estado_stats, 'NPS_Score')
        # Nota de advertencia
        st.markdown('<div class="note-box">⚠️ <b>Nota:</b> Los rankings no están normalizados por volumen de respuestas. Estados con pocas respuestas pueden aparecer en el top sin ser estadísticamente representativos. Ver pestaña "Normalizado" para una vista ponderada.</div>', unsafe_allow_html=True)

    with map_tabs[1]:
        c1, c2 = st.columns([2.5, 1])
        with c1: st.plotly_chart(crear_mapa_choropleth(estado_stats, 'D_1', 'Satisfacción'), use_container_width=True)
        with c2:
            st.markdown("**Top Estados**")
            mostrar_ranking(estado_stats, 'D_1')
        st.markdown('<div class="note-box">⚠️ <b>Nota:</b> Los rankings no están normalizados por volumen de respuestas.</div>', unsafe_allow_html=True)

    with map_tabs[2]:
        c1, c2 = st.columns([2.5, 1])
        with c1: st.plotly_chart(crear_mapa_choropleth(estado_stats, 'C_1', 'Calidad'), use_container_width=True)
        with c2:
            st.markdown("**Top Estados**")
            mostrar_ranking(estado_stats, 'C_1')
        st.markdown('<div class="note-box">⚠️ <b>Nota:</b> Los rankings no están normalizados por volumen de respuestas.</div>', unsafe_allow_html=True)

    with map_tabs[3]:
        c1, c2 = st.columns([2.5, 1])
        with c1: st.plotly_chart(crear_mapa_choropleth(estado_stats, 'score_servqual_total', 'SERVQUAL'), use_container_width=True)
        with c2:
            st.markdown("**Top Estados**")
            mostrar_ranking(estado_stats, 'score_servqual_total')
        st.markdown('<div class="note-box">⚠️ <b>Nota:</b> Los rankings no están normalizados por volumen de respuestas.</div>', unsafe_allow_html=True)

    with map_tabs[4]:
        st.markdown("""
        <div class="insight-card">
            <b>⚖️ NPS Normalizado</b>: Pondera el NPS por un factor de confianza basado en el número de respuestas.<br>
            <code>NPS_norm = NPS × (1 - 1/√n)</code><br>
            Estados con más respuestas tienen mayor peso. Esto evita que estados con 1-2 respuestas dominen el ranking.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("**Mapa de Volumen de Respuestas**")
            st.plotly_chart(crear_mapa_volumen(estado_stats), use_container_width=True)
        with c2:
            st.markdown("**Top Estados (Normalizado)**")
            mostrar_ranking_normalizado(estado_stats)

        # Tabla comparativa
        st.markdown("**Comparación: Ranking Original vs Normalizado**")
        comparison = estado_stats[['Estado_limpio', 'NPS_Score', 'n', 'confianza', 'NPS_normalizado']].copy()
        comparison['Rank_Original'] = comparison['NPS_Score'].rank(ascending=False).astype(int)
        comparison['Rank_Normalizado'] = comparison['NPS_normalizado'].rank(ascending=False).astype(int)
        comparison['Cambio'] = comparison['Rank_Original'] - comparison['Rank_Normalizado']
        comparison = comparison.sort_values('Rank_Normalizado').head(10)
        comparison.columns = ['Estado', 'NPS', 'n', 'Confianza', 'NPS Norm.', 'Rank Orig.', 'Rank Norm.', 'Δ']
        comparison['Confianza'] = comparison['Confianza'].apply(lambda x: f"{x:.0%}")
        comparison['NPS Norm.'] = comparison['NPS Norm.'].apply(lambda x: f"{x:.1f}")
        comparison['Δ'] = comparison['Δ'].apply(lambda x: f"+{x}" if x > 0 else str(x) if x < 0 else "=")
        st.dataframe(comparison, hide_index=True, use_container_width=True)

    # =========================================================================
    # RADAR Y NPS
    # =========================================================================
    st.markdown('<p class="section-title">📊 Perfil de Calidad de Servicio</p>', unsafe_allow_html=True)

    # SECCIÓN RESTAURADA: ¿Qué es SERVQUAL?
    st.markdown("""
    <div class="insight-card">
        <b>¿Qué es SERVQUAL?</b> Es un modelo de medición de calidad de servicio desarrollado por Parasuraman et al. (1988).
        Mide 4 dimensiones: <b>Tangibles</b> (apariencia), <b>Fiabilidad</b> (cumplimiento), <b>Responsiveness</b> (rapidez) y <b>Empatía</b> (atención personalizada).
    </div>
    """, unsafe_allow_html=True)

    col_radar, col_nps = st.columns([1, 1])

    with col_radar:
        cats = ['Tangibles', 'Fiabilidad', 'Responsiveness', 'Empatía']
        vals = [df_f[f'score_{c.lower()}'].mean() for c in ['tangibles', 'fiabilidad', 'responsiveness', 'empatia']]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself',
                                            fillcolor='rgba(90,0,119,0.2)', line=dict(color=PALETA['morado_primario'], width=3), name='Actual'))
        fig_radar.add_trace(go.Scatterpolar(r=[4,4,4,4,4], theta=cats+[cats[0]],
                                            line=dict(color=PALETA['amarillo'], width=4, dash='dash'), name='Objetivo (4.0)'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[2.5, 5])),
                               showlegend=True, height=350, paper_bgcolor='white')
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_nps:
        n_prom = len(df_f[df_f['nps_categoria']=='Promotor'])
        n_pas = len(df_f[df_f['nps_categoria']=='Pasivo'])
        n_det = len(df_f[df_f['nps_categoria']=='Detractor'])
        total = n_prom + n_pas + n_det

        fig_nps = go.Figure(go.Pie(values=[n_det, n_pas, n_prom], labels=['Detractores', 'Pasivos', 'Promotores'],
                                   hole=0.6, marker_colors=[ESCALA_NPS[-1], ESCALA_NPS[2], ESCALA_NPS[0]],
                                   textinfo='percent', sort=False))
        fig_nps.update_layout(height=200, margin=dict(t=10,b=10),
                             annotations=[dict(text=f'NPS<br><b>{nps_score:.0f}</b>', x=0.5, y=0.5, font_size=16, showarrow=False)],
                             paper_bgcolor='white', showlegend=False)
        st.plotly_chart(fig_nps, use_container_width=True)

        nps_df = pd.DataFrame({'Cat': ['Detractores (1-6)', 'Pasivos (7-8)', 'Promotores (9-10)'],
                               'N': [n_det, n_pas, n_prom],
                               'Pct': [n_det/total*100 if total>0 else 0, n_pas/total*100 if total>0 else 0, n_prom/total*100 if total>0 else 0]})
        fig_bar = go.Figure(go.Bar(y=nps_df['Cat'], x=nps_df['Pct'], orientation='h',
                                   marker_color=[ESCALA_NPS[-1], ESCALA_NPS[2], ESCALA_NPS[0]],
                                   text=[f"{p:.0f}% (n={n})" for p,n in zip(nps_df['Pct'], nps_df['N'])], textposition='outside'))
        fig_bar.update_layout(height=140, xaxis_range=[0,100], plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=5,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # =========================================================================
    # ANÁLISIS POR SEGMENTOS
    # =========================================================================
    st.markdown('<p class="section-title">👥 Análisis por Segmentos</p>', unsafe_allow_html=True)

    seg_tabs = st.tabs(["🎯 NPS", "😊 Satisfacción", "⭐ Calidad", "📋 SERVQUAL"])
    metric_map = {'🎯 NPS': ('NPS', [1,10]), '😊 Satisfacción': ('D_1', [5,10]),
                  '⭐ Calidad': ('C_1', [2,5]), '📋 SERVQUAL': ('score_servqual_total', [2.5,5])}

    def crear_barras_seg(df_data, col, rango, group_col, horizontal=True):
        stats = df_data.groupby(group_col)[col].mean().sort_values(ascending=horizontal)
        media = df_data[col].mean()
        colores = [PALETA['morado_primario'] if v >= media else PALETA['gris_apagado'] for v in stats.values]
        if horizontal:
            fig = go.Figure(go.Bar(y=stats.index, x=stats.values, orientation='h', marker_color=colores,
                                   text=[f"{v:.1f}" for v in stats.values], textposition='outside'))
            fig.add_vline(x=media, line_dash="dash", line_color=PALETA['amarillo'], line_width=4)
            fig.update_layout(xaxis_range=rango)
        else:
            fig = go.Figure(go.Bar(x=stats.index, y=stats.values, marker_color=colores,
                                   text=[f"{v:.1f}" for v in stats.values], textposition='outside'))
            fig.add_hline(y=media, line_dash="dash", line_color=PALETA['amarillo'], line_width=4)
            fig.update_layout(yaxis_range=rango)
        fig.update_layout(height=260, plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=10,b=30))
        return fig, media

    for tab, (metric_col, rango) in zip(seg_tabs, metric_map.values()):
        with tab:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Por Organización**")
                fig, media = crear_barras_seg(df_f, metric_col, rango, 'Giro_display')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("**Por Antigüedad**")
                df_ant = df_f.dropna(subset=['antiguedad_grupo'])
                fig, _ = crear_barras_seg(df_ant, metric_col, rango, 'antiguedad_grupo', horizontal=False)
                st.plotly_chart(fig, use_container_width=True)
            with c3:
                st.markdown("**Por Región**")
                df_reg = df_f.dropna(subset=['region_simplificada'])
                fig, _ = crear_barras_seg(df_reg, metric_col, rango, 'region_simplificada')
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'<div class="insight-card">📊 Línea amarilla = media global ({df_f[metric_col].mean():.1f}). <b style="color:{PALETA["morado_primario"]}">Morado</b> = sobre media. <b style="color:{PALETA["gris_apagado"]}">Gris</b> = bajo media.</div>', unsafe_allow_html=True)

    # =========================================================================
    # VOLUMEN TEMPORAL
    # =========================================================================
    st.markdown('<p class="section-title">📅 Volumen de Respuestas en el Tiempo</p>', unsafe_allow_html=True)

    df_f['fecha'] = pd.to_datetime(df_f['fecha'])
    vol_diario = df_f.groupby('fecha').size().reset_index(name='respuestas')

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=vol_diario['fecha'], y=vol_diario['respuestas'],
                                  mode='lines+markers', line=dict(color=PALETA['morado_primario'], width=2),
                                  marker=dict(size=6), fill='tozeroy', fillcolor='rgba(90,0,119,0.1)'))
    fig_vol.update_layout(height=250, plot_bgcolor='white', paper_bgcolor='white',
                          xaxis_title='Fecha', yaxis_title='Respuestas', margin=dict(t=20,b=40))
    st.plotly_chart(fig_vol, use_container_width=True)

    # =========================================================================
    # ÁREA DE OPORTUNIDAD - REDISEÑADA
    # =========================================================================
    st.markdown('<p class="section-title">🎯 Áreas de Oportunidad</p>', unsafe_allow_html=True)

    items_names = {'AT_1': 'Apariencia', 'AT_2': 'Documentación', 'FI_1': 'Puntualidad',
                   'FI_2': 'Conocimiento', 'FI_3': 'Info Clara', 'R_1': 'Rapidez',
                   'R_2': 'Disposición', 'R_3': 'Flexibilidad', 'E_1': 'Comprensión',
                   'E_2': 'Tiempo', 'E_3': 'Preocupación', 'E_4': 'Personalización'}

    items_dimension = {'AT_1': 'Tangibles', 'AT_2': 'Tangibles', 'FI_1': 'Fiabilidad',
                       'FI_2': 'Fiabilidad', 'FI_3': 'Fiabilidad', 'R_1': 'Responsiveness',
                       'R_2': 'Responsiveness', 'R_3': 'Responsiveness', 'E_1': 'Empatía',
                       'E_2': 'Empatía', 'E_3': 'Empatía', 'E_4': 'Empatía'}

    item_stats = df_f[vars_servqual].agg(['mean', 'std']).T
    item_stats['nombre'] = item_stats.index.map(items_names)
    item_stats['dimension'] = item_stats.index.map(items_dimension)
    item_stats = item_stats.sort_values('mean')

    # Layout más compacto: 2 columnas
    col_opp1, col_opp2 = st.columns([1.2, 1])

    with col_opp1:
        # Top 5 items más bajos con barras horizontales compactas
        top5 = item_stats.head(5)

        # Colores por dimensión
        dim_colors = {'Tangibles': PALETA['turquesa'], 'Fiabilidad': PALETA['morado_primario'],
                      'Responsiveness': PALETA['naranja'], 'Empatía': PALETA['amarillo']}

        colors = [dim_colors.get(d, PALETA['gris_medio']) for d in top5['dimension']]

        fig_opp = go.Figure(go.Bar(
            y=top5['nombre'], x=top5['mean'], orientation='h',
            marker_color=colors,
            text=[f"{v:.2f}" for v in top5['mean']], textposition='outside'
        ))
        fig_opp.add_vline(x=df_f['score_servqual_total'].mean(), line_dash="dash",
                         line_color=PALETA['gris_medio'], line_width=2,
                         annotation_text=f"Media: {df_f['score_servqual_total'].mean():.2f}")
        fig_opp.update_layout(
            height=200,
            xaxis_range=[3, 5],
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=10, b=30, l=10, r=10),
            xaxis_title='Puntuación (1-5)'
        )
        st.plotly_chart(fig_opp, use_container_width=True)

    with col_opp2:
        # Recomendaciones en cards compactas
        st.markdown("**Recomendaciones de Acción**")

        for i, (idx, row) in enumerate(top5.head(3).iterrows()):
            icon = "🔴" if row['mean'] < 3.8 else "🟡" if row['mean'] < 4.0 else "🟢"
            gap = 4.0 - row['mean']  # Gap respecto a objetivo 4.0
            st.markdown(f"""
            <div class="opp-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span>{icon} <b>{row['nombre']}</b></span>
                    <span style="color:{PALETA['morado_primario']}; font-weight:bold">{row['mean']:.2f}</span>
                </div>
                <small style="color:#666">{row['dimension']} | Gap: {gap:.2f} pts</small>
            </div>
            """, unsafe_allow_html=True)

        # Leyenda de colores por dimensión
        st.markdown("<br>**Leyenda:**", unsafe_allow_html=True)
        legend_html = " ".join([f'<span style="color:{c}">●</span> {d}' for d, c in dim_colors.items()])
        st.markdown(f'<small>{legend_html}</small>', unsafe_allow_html=True)

    # Insight más compacto
    worst_item = item_stats.iloc[0]
    worst_dim = worst_item['dimension']
    st.markdown(f"""
    <div class="insight-card">
        💡 <b>Prioridad:</b> Mejorar <b>{worst_item['nombre']}</b> (dimensión {worst_dim}) con puntuación {worst_item['mean']:.2f}.
        La variabilidad (σ={worst_item['std']:.2f}) sugiere {"experiencias inconsistentes" if worst_item['std'] > 0.8 else "consistencia en la percepción"}.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: ANÁLISIS ESTADÍSTICO
# =============================================================================
with tab2:
    st.markdown('<div class="warning-box">⚠️ Los filtros <b>NO aplican</b> aquí. Se usa el dataset completo (n=274) para validez estadística.</div>', unsafe_allow_html=True)

    # Correlaciones
    st.markdown('<p class="section-title">🔗 Matriz de Correlaciones</p>', unsafe_allow_html=True)
    vars_corr = vars_scores + ['D_1', 'NPS', 'C_1']
    corr = df[vars_corr].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_m = corr.where(~mask)
    rename_c = {'score_tangibles': 'Tang.', 'score_fiabilidad': 'Fiab.', 'score_responsiveness': 'Resp.',
                'score_empatia': 'Emp.', 'D_1': 'Satisf.', 'NPS': 'NPS', 'C_1': 'Calidad'}
    corr_r = corr_m.rename(index=rename_c, columns=rename_c)
    fig_corr = px.imshow(corr_r, text_auto='.2f', color_continuous_scale=ESCALA_DIVERGENTE, zmin=-1, zmax=1)
    fig_corr.update_layout(height=380, paper_bgcolor='white')
    st.plotly_chart(fig_corr, use_container_width=True)

    # Chi-cuadrada
    st.markdown('<p class="section-title">🧪 Chi-Cuadrada</p>', unsafe_allow_html=True)

    st.markdown("### 1. NPS vs Organización")
    df_d = df.copy()
    df_d['Giro_display'] = df_d['Giro'].replace({'Teletón (Grupos internos de la Fundación)': 'Teletón'})
    cont1 = pd.crosstab(df_d['Giro_display'], df_d['nps_categoria'])
    chi2_1, p1, dof1, _ = chi2_contingency(cont1)
    cont1_pct = cont1.div(cont1.sum(axis=1), axis=0) * 100
    cols_o = [c for c in ['Detractor', 'Pasivo', 'Promotor'] if c in cont1_pct.columns]
    fig_chi = px.imshow(cont1_pct[cols_o], text_auto='.0f',
                       color_continuous_scale=[[0, ESCALA_NPS[-1]], [0.5, ESCALA_NPS[2]], [1, ESCALA_NPS[0]]],
                       aspect='auto')
    fig_chi.update_layout(height=280, coloraxis_showscale=False)
    st.plotly_chart(fig_chi, use_container_width=True)
    st.markdown(f'<div class="stat-box">χ²={chi2_1:.2f}, gl={dof1}, p={p1:.4f} → {"✅ Significativo" if p1<0.05 else "❌ No significativo"}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 2. NPS vs Antigüedad")
    df_c2 = df.dropna(subset=['antiguedad_grupo', 'nps_categoria'])
    cont2 = pd.crosstab(df_c2['antiguedad_grupo'], df_c2['nps_categoria'])
    chi2_2, p2, dof2, _ = chi2_contingency(cont2)
    cont2_pct = cont2.div(cont2.sum(axis=1), axis=0) * 100
    fig_chi2 = px.imshow(cont2_pct[cols_o].reindex(['Nuevo', 'Establecido', 'Veterano']), text_auto='.0f',
                        color_continuous_scale=[[0, ESCALA_NPS[-1]], [0.5, ESCALA_NPS[2]], [1, ESCALA_NPS[0]]],
                        aspect='auto')
    fig_chi2.update_layout(height=220, coloraxis_showscale=False)
    st.plotly_chart(fig_chi2, use_container_width=True)
    st.markdown(f'<div class="stat-box">χ²={chi2_2:.2f}, gl={dof2}, p={p2:.4f} → {"✅ Significativo" if p2<0.05 else "❌ No significativo"}</div>', unsafe_allow_html=True)

    # T-tests
    st.markdown('<p class="section-title">📊 Pruebas t</p>', unsafe_allow_html=True)

    def ttest(g1, g2, n1, n2, var):
        if len(g1)<5 or len(g2)<5: return None
        t, p = ttest_ind(g1, g2)
        ps = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
        d = (g1.mean() - g2.mean()) / ps if ps>0 else 0
        return {'n1':n1, 'n2':n2, 'm1':g1.mean(), 'm2':g2.mean(), 'c1':len(g1), 'c2':len(g2), 't':t, 'p':p, 'd':d, 'var':var}

    tests = [
        (df[df['nps_categoria']=='Promotor']['D_1'].dropna(), df[df['nps_categoria']=='Detractor']['D_1'].dropna(), 'Promotores', 'Detractores', 'Satisfacción'),
        (df[df['antiguedad_grupo']=='Nuevo']['score_servqual_total'].dropna(), df[df['antiguedad_grupo']=='Veterano']['score_servqual_total'].dropna(), 'Nuevos', 'Veteranos', 'SERVQUAL'),
        (df[df['Giro']=='Empresa']['D_1'].dropna(), df[df['Giro']=='Persona física']['D_1'].dropna(), 'Empresa', 'Persona física', 'Satisfacción'),
        (df[df['region_simplificada']=='Centro']['NPS'].dropna(), df[df['region_simplificada']!='Centro']['NPS'].dropna(), 'Centro', 'Otras', 'NPS')
    ]

    for i, (g1, g2, n1, n2, var) in enumerate(tests, 1):
        st.markdown(f"### {i}. {var}: {n1} vs {n2}")
        r = ttest(g1, g2, n1, n2, var)
        if r:
            df_box = pd.DataFrame({'Grupo': [n1]*len(g1)+[n2]*len(g2), 'Valor': list(g1)+list(g2)})
            fig = px.box(df_box, x='Grupo', y='Valor', color='Grupo', color_discrete_sequence=COLORES_CAT[:2])
            fig.update_layout(height=220, showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
            ef = "pequeño" if abs(r['d'])<0.5 else "mediano" if abs(r['d'])<0.8 else "grande"
            st.markdown(f'<div class="stat-box">{n1}: M={r["m1"]:.2f} (n={r["c1"]}) | {n2}: M={r["m2"]:.2f} (n={r["c2"]})<br>t={r["t"]:.3f}, p={r["p"]:.4f}, d={r["d"]:.2f} ({ef}) → {"✅" if r["p"]<0.05 else "❌"}</div>', unsafe_allow_html=True)
        st.markdown("---")

    # ANOVA
    st.markdown('<p class="section-title">📈 ANOVA</p>', unsafe_allow_html=True)
    ca1, ca2, ca3 = st.columns(3)

    with ca1:
        st.markdown("**Por Organización**")
        grupos = [df[df['Giro_display']==g]['D_1'].dropna() for g in df['Giro_display'].dropna().unique()]
        grupos = [g for g in grupos if len(g)>=3]
        if len(grupos)>=2:
            f, p = f_oneway(*grupos)
            st.markdown(f'<div class="stat-box">F={f:.2f}, p={p:.4f} {"✅" if p<0.05 else "❌"}</div>', unsafe_allow_html=True)
            fig = px.box(df.dropna(subset=['Giro_display','D_1']), x='Giro_display', y='D_1', color='Giro_display', color_discrete_sequence=COLORES_CAT)
            fig.update_layout(height=230, showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    with ca2:
        st.markdown("**Por Antigüedad**")
        df_a = df.dropna(subset=['antiguedad_grupo'])
        grupos2 = [df_a[df_a['antiguedad_grupo']==g]['D_1'].dropna() for g in ['Nuevo','Establecido','Veterano']]
        grupos2 = [g for g in grupos2 if len(g)>=3]
        if len(grupos2)>=2:
            f2, p2 = f_oneway(*grupos2)
            st.markdown(f'<div class="stat-box">F={f2:.2f}, p={p2:.4f} {"✅" if p2<0.05 else "❌"}</div>', unsafe_allow_html=True)
            fig2 = px.box(df_a, x='antiguedad_grupo', y='D_1', color='antiguedad_grupo',
                         category_orders={'antiguedad_grupo':['Nuevo','Establecido','Veterano']}, color_discrete_sequence=COLORES_CAT)
            fig2.update_layout(height=230, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    with ca3:
        st.markdown("**Por Región**")
        df_r = df.dropna(subset=['region_simplificada'])
        grupos3 = [df_r[df_r['region_simplificada']==g]['D_1'].dropna() for g in df_r['region_simplificada'].unique()]
        grupos3 = [g for g in grupos3 if len(g)>=3]
        if len(grupos3)>=2:
            f3, p3 = f_oneway(*grupos3)
            st.markdown(f'<div class="stat-box">F={f3:.2f}, p={p3:.4f} {"✅" if p3<0.05 else "❌"}</div>', unsafe_allow_html=True)
            fig3 = px.box(df_r, x='region_simplificada', y='D_1', color='region_simplificada', color_discrete_sequence=COLORES_CAT)
            fig3.update_layout(height=230, showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig3, use_container_width=True)

    # Regresión
    st.markdown('<p class="section-title">📐 Regresión: Predicción NPS</p>', unsafe_allow_html=True)
    df_rg = df[vars_scores + ['NPS']].dropna()
    if len(df_rg) > 20:
        X = sm.add_constant(df_rg[vars_scores])
        mod = sm.OLS(df_rg['NPS'], X).fit()
        cr1, cr2 = st.columns(2)
        with cr1:
            st.markdown(f'<div class="stat-box"><b>R² = {mod.rsquared:.3f}</b> ({mod.rsquared*100:.1f}%)<br>F = {mod.fvalue:.2f}, p = {mod.f_pvalue:.4f}</div>', unsafe_allow_html=True)
            coef_df = pd.DataFrame({'Variable': [v.replace('score_','').title() for v in vars_scores],
                                    'β': [mod.params[v] for v in vars_scores], 'p': [mod.pvalues[v] for v in vars_scores]})
            coef_df['Sig'] = coef_df['p'].apply(lambda x: '✅' if x<0.05 else '')
            st.dataframe(coef_df.round(4), hide_index=True)
        with cr2:
            coefs = pd.Series({v.replace('score_','').title(): mod.params[v] for v in vars_scores}).sort_values()
            max_v = coefs.abs().idxmax()
            fig_c = go.Figure(go.Bar(y=coefs.index, x=coefs.values, orientation='h',
                                     marker_color=[PALETA['morado_primario'] if i==max_v else PALETA['gris_medio'] for i in coefs.index],
                                     text=[f"{v:.2f}" for v in coefs.values], textposition='outside'))
            fig_c.add_vline(x=0, line_color='black')
            fig_c.update_layout(height=220, plot_bgcolor='white')
            st.plotly_chart(fig_c, use_container_width=True)
            st.markdown(f'<div class="insight-card">🎯 Variable más importante: <b>{max_v}</b></div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f'<div style="text-align:center;color:{PALETA["gris_medio"]};font-size:11px">Dashboard SERVQUAL v9 | Fundación Teletón</div>', unsafe_allow_html=True)
