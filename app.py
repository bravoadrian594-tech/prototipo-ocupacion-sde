"""
PASO 4 - Aplicación web (Sprint 4, historias PB-08 y PB-09)

Lee la base en modo solo lectura y muestra dos vistas:
  - Hospedadores: pronóstico de doce meses por zona, modelo utilizado con su
    MAE y RMSE, y eventos registrados en el mes elegido.
  - Turistas: ocupación pronosticada del mes del viaje y nivel esperado
    (bajo, medio o alto) según los terciles de los meses observados de la zona.

Cómo se corre en la compu:   streamlit run app.py
"""

import sqlite3
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from base import BASE

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
         "Septiembre", "Octubre", "Noviembre", "Diciembre"]
NOMBRE_MODELO = {"base": "Suavizado exponencial simple",
                 "xgboost_sin_eventos": "XGBoost sin eventos",
                 "xgboost_con_eventos": "XGBoost con eventos"}
NIVELES = {1: "Local", 2: "Regional", 3: "Nacional", 4: "Internacional"}
COLOR_NIVEL = {"Bajo": "#2e8b57", "Medio": "#e0a800", "Alto": "#c0392b"}
TEXTO_NIVEL = {"Bajo": "Tranquilo", "Medio": "Normal", "Alto": "Mucha gente"}
ICONO_TIPO = {"deportivo": "⚽ Deportivo", "cultural": "🎭 Cultural"}


def etiqueta_nivel(nivel, tamano="1.6rem"):
    """Etiqueta de color fuerte (la misma del semáforo) para el nivel esperado."""
    return (f"<span style='background:{COLOR_NIVEL[nivel]}; color:white; font-weight:700; "
            f"font-size:{tamano}; padding:0.1rem 0.7rem; border-radius:8px; white-space:nowrap'>"
            f"{TEXTO_NIVEL[nivel]}</span>")

st.set_page_config(page_title="Pronóstico de ocupación - Santiago del Estero", layout="wide")


def detectar_celular():
    """True si la página se abre desde un celular. Con ?pantalla=celular o ?pantalla=compu se fuerza."""
    forzado = st.query_params.get("pantalla")
    if forzado in ("celular", "compu"):
        return forzado == "celular"
    agente = (st.context.headers.get("User-Agent") or "").lower()
    return any(p in agente for p in ("mobi", "android", "iphone"))


ES_CELULAR = detectar_celular()


@st.cache_data
def leer(consulta, params=()):
    con = sqlite3.connect(f"file:{BASE}?mode=ro", uri=True)
    try:
        return pd.read_sql(consulta, con, params=params)
    finally:
        con.close()


def etiqueta(anio, mes):
    return f"{MESES[mes - 1]} {anio}"


def corta(anio, mes):
    """Etiqueta corta para el eje del gráfico, por ejemplo 'jul 26'."""
    return f"{MESES[mes - 1][:3].lower()} {str(anio)[2:]}"


# ------------------------------------------------------------------ datos
zonas = leer("SELECT * FROM zona ORDER BY id_zona")
corridas = leer("SELECT * FROM corrida WHERE version LIKE '%produccion%' ORDER BY id_corrida DESC")

st.sidebar.markdown("## Pronóstico de ocupación")
st.sidebar.caption("Ciudad de Santiago del Estero y Termas de Río Hondo")

if corridas.empty:
    st.warning("Todavía no hay pronósticos guardados. Ejecutá primero los pasos 1 a 3.")
    st.stop()

corrida = corridas.iloc[0]
id_corrida = int(corrida["id_corrida"])
meses_corrida = leer("SELECT DISTINCT anio, mes FROM pronostico WHERE id_corrida = ? ORDER BY anio, mes",
                     (id_corrida,))
opciones_mes = [etiqueta(a, m) for a, m in zip(meses_corrida["anio"], meses_corrida["mes"])]

# Selectores arriba de la página, para que se vean también en el celular
st.markdown("<div style='font-size:0.95rem; font-weight:600; opacity:0.75'>"
            "Pronóstico de ocupación · Santiago del Estero</div>", unsafe_allow_html=True)
vista = st.segmented_control("¿Quién consulta?", ["Hospedadores", "Turistas"], default="Hospedadores",
                             key="vista", help="Hospedadores: tengo un alojamiento. "
                                               "Turistas: estoy planeando un viaje.") or "Hospedadores"
c_zona, c_mes = st.columns(2)
nombre_zona = c_zona.selectbox("Zona", zonas["nombre"].tolist(), key="zona")
mes_elegido = c_mes.selectbox("Mes que quiere consultar" if vista == "Hospedadores" else "Mes del viaje",
                              opciones_mes, key="mes")
id_zona = int(zonas.loc[zonas["nombre"] == nombre_zona, "id_zona"].iloc[0])
st.divider()
metricas = leer("SELECT * FROM metrica_modelo WHERE id_corrida = ? AND id_zona = ?", (id_corrida, id_zona))
if metricas.empty:
    st.warning("No hay datos para esta zona.")
    st.stop()

ganador = metricas.loc[metricas["es_ganador"] == 1].iloc[0]
pron = leer("""SELECT anio, mes, ocupacion_pronosticada FROM pronostico
               WHERE id_corrida = ? AND id_zona = ? AND modelo = ? ORDER BY anio, mes""",
            (id_corrida, id_zona, ganador["modelo"]))
if pron.empty:
    st.warning("No hay datos para esta zona.")
    st.stop()

fecha_corrida = datetime.fromisoformat(corrida["fecha_corrida"])
origen = pd.Period(corrida["corte_entrenamiento"][:7], freq="M")
mae = float(ganador["mae"])

# Pronóstico con su rango (más o menos el MAE del modelo ganador)
pron["Mes"] = [etiqueta(a, m) for a, m in zip(pron["anio"], pron["mes"])]
pron["Eje"] = [corta(a, m) for a, m in zip(pron["anio"], pron["mes"])]
pron["Mínimo"] = (pron["ocupacion_pronosticada"] - mae).clip(lower=0)
pron["Máximo"] = (pron["ocupacion_pronosticada"] + mae).clip(upper=100)

# Niveles bajo, medio y alto: terciles de los meses observados de la zona hasta el origen
obs = leer("""SELECT anio, mes, tasa_ocupacion_plazas FROM ocupacion_mensual
              WHERE id_zona = ? AND origen = 'observado'""", (id_zona,))
obs = obs[[pd.Period(year=a, month=m, freq="M") <= origen for a, m in zip(obs["anio"], obs["mes"])]]
t1, t2 = np.quantile(obs["tasa_ocupacion_plazas"], [1 / 3, 2 / 3])
pron["Nivel"] = ["Bajo" if v <= t1 else ("Medio" if v <= t2 else "Alto") for v in pron["ocupacion_pronosticada"]]

# Eventos registrados en cada mes del pronóstico
todos_eventos = leer("""SELECT nombre, tipo, fecha_inicio, fecha_fin, nivel_alcance, estado
                        FROM evento WHERE id_zona = ? ORDER BY fecha_inicio""", (id_zona,))


def eventos_del_mes(anio, mes):
    inicio = f"{int(anio)}-{int(mes):02d}-01"
    fin = (pd.Timestamp(inicio) + pd.offsets.MonthEnd(0)).date().isoformat()
    return todos_eventos[(todos_eventos["fecha_inicio"] <= fin) & (todos_eventos["fecha_fin"] >= inicio)]


pron["Eventos"] = [len(eventos_del_mes(a, m)) for a, m in zip(pron["anio"], pron["mes"])]
pron["Nombres"] = [", ".join(eventos_del_mes(a, m)["nombre"]) for a, m in zip(pron["anio"], pron["mes"])]

# Últimos doce meses reales publicados por el INDEC
historia = leer("""SELECT anio, mes, tasa_ocupacion_plazas FROM ocupacion_mensual
                   WHERE id_zona = ? AND origen = 'observado' ORDER BY anio DESC, mes DESC LIMIT 12""",
                (id_zona,)).iloc[::-1]
historia["Mes"] = [etiqueta(a, m) for a, m in zip(historia["anio"], historia["mes"])]
historia["Eje"] = [corta(a, m) for a, m in zip(historia["anio"], historia["mes"])]

desde, hasta = pron["Mes"].iloc[0], pron["Mes"].iloc[-1]
fila = pron.loc[pron["Mes"] == mes_elegido].iloc[0]
valor = float(fila["ocupacion_pronosticada"])
ultimo_dato = etiqueta(origen.year, origen.month).lower()

st.sidebar.divider()
st.sidebar.caption(f"Meses pronosticados: {desde.lower()} a {hasta.lower()}.  \n"
                   f"Última actualización: {fecha_corrida:%d/%m/%Y}.  \n"
                   "Datos oficiales del INDEC (Encuesta de Ocupación Hotelera).")


# ------------------------------------------------------------------ avisos (riesgo R5, Tabla 22)
def titulo(texto):
    """Título principal: más chico en pantallas angostas."""
    st.markdown(f"<h2 style='font-size:clamp(1.45rem, 5vw, 2.1rem); line-height:1.25; "
                f"padding:0.2rem 0 0.4rem 0'>{texto}</h2>", unsafe_allow_html=True)


def avisos():
    meses_sin_dato = (pd.Period(datetime.now(), freq="M") - origen).n
    if meses_sin_dato > 2:
        st.warning(f"Los datos oficiales más recientes son de {ultimo_dato}. Desde entonces el INDEC no "
                   "publicó datos nuevos, así que este pronóstico se hizo con lo último disponible. "
                   "Se va a actualizar cuando haya datos nuevos.")
    if (datetime.now() - fecha_corrida).days > 62:
        st.warning("Este pronóstico no se actualiza hace más de dos meses.")


# ------------------------------------------------------------------ gráfico
def grafico(resaltado):
    orden = (list(historia["Eje"]) if not ES_CELULAR else []) + list(pron["Eje"])
    eje_x = alt.X("Eje:N", sort=orden, title=None, axis=alt.Axis(labelAngle=-45, labelOverlap=False))
    escala_y = alt.Scale(domain=[0, 105])
    datos = pron.assign(Elegido=pron["Mes"] == resaltado, Marca=pron["Máximo"] + 5)
    ayuda = [alt.Tooltip("Mes:N"),
             alt.Tooltip("ocupacion_pronosticada:Q", title="Se espera (%)", format=".0f"),
             alt.Tooltip("Mínimo:Q", title="Entre (%)", format=".0f"),
             alt.Tooltip("Máximo:Q", title="y (%)", format=".0f"),
             alt.Tooltip("Nivel:N", title="Movimiento")]
    clic = alt.selection_point(name="clic", fields=["Mes"], on="click")
    barras = (alt.Chart(datos).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, cursor="pointer")
              .add_params(clic)
              .encode(x=eje_x,
                      y=alt.Y("ocupacion_pronosticada:Q", title="Camas ocupadas (%)", scale=escala_y,
                              axis=alt.Axis(values=[0, 20, 40, 60, 80, 100])),
                      color=alt.condition("datum.Elegido", alt.value("#1f5fa8"), alt.value("#a9bfdc")),
                      tooltip=ayuda))
    rango = (alt.Chart(datos).mark_rule(color="#e08a1e", strokeWidth=2)
             .encode(x=eje_x, y=alt.Y("Mínimo:Q", scale=escala_y), y2="Máximo:Q", tooltip=ayuda))
    marcas = (alt.Chart(datos[datos["Eventos"] > 0]).mark_point(shape="triangle-down", filled=True,
                                                                size=110, color="#c0392b")
              .encode(x=eje_x, y=alt.Y("Marca:Q", scale=escala_y),
                      tooltip=[alt.Tooltip("Mes:N"), alt.Tooltip("Nombres:N", title="Eventos")]))
    real = (alt.Chart(historia).mark_line(point=alt.OverlayMarkDef(color="#8a8a8a"), color="#8a8a8a")
            .encode(x=eje_x, y=alt.Y("tasa_ocupacion_plazas:Q", scale=escala_y),
                    tooltip=[alt.Tooltip("Mes:N"),
                             alt.Tooltip("tasa_ocupacion_plazas:Q", title="Ocupación real (%)", format=".0f")]))
    if ES_CELULAR:  # en el celular, solo los doce meses pronosticados
        return (barras + rango + marcas).properties(height=280)
    return (real + barras + rango + marcas).properties(height=380)


def al_tocar_barra():
    """Al tocar una barra, ese mes pasa a ser el mes elegido."""
    puntos = st.session_state["grafico"].selection.get("clic", [])
    if puntos and puntos[0].get("Mes") in opciones_mes:
        st.session_state["mes"] = puntos[0]["Mes"]


def mostrar_grafico():
    st.altair_chart(grafico(mes_elegido), width="stretch", key="grafico",
                    on_select=al_tocar_barra, selection_mode="clic")
    st.caption((LEYENDA_CELULAR if ES_CELULAR else LEYENDA) + " Toque una barra para ver ese mes.")


LEYENDA = ("**Línea gris:** lo que pasó en los últimos 12 meses. "
           "**Barras:** lo que se espera (la más oscura es el mes elegido). "
           "**Rayita naranja:** margen de error. "
           "**Triángulo rojo:** mes con eventos.")

LEYENDA_CELULAR = ("**Barras:** lo que se espera (la más oscura es el mes elegido). "
                   "**Rayita naranja:** margen de error. **Triángulo rojo:** mes con eventos.")

AYUDA_NIVEL = ("Compara el mes con el resto del año en la zona. Tranquilo (nivel bajo): de los meses con "
               "menos gente. Normal (nivel medio): un mes común. Mucha gente (nivel alto): de los meses con "
               "más movimiento.")


def resumen_del_mes():
    if ES_CELULAR:
        st.markdown(f"<div style='display:flex; align-items:center; gap:0.8rem; margin:0.3rem 0'>"
                    f"<span style='font-size:2.4rem; font-weight:600'>{valor:.0f} %</span>"
                    f"{etiqueta_nivel(fila['Nivel'], '1.3rem')}</div>", unsafe_allow_html=True,
                    help=AYUDA_NIVEL)
        st.caption(f"Camas ocupadas que se esperan en {mes_elegido.lower()}. Puede quedar entre "
                   f"{fila['Mínimo']:.0f} % y {fila['Máximo']:.0f} %.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Ocupación esperada en {mes_elegido.lower()}", f"{valor:.0f} %",
              help=f"De cada 100 camas de la zona, se espera que estén ocupadas unas {valor:.0f}.")
    c2.metric("Puede quedar entre", f"{fila['Mínimo']:.0f} % y {fila['Máximo']:.0f} %",
              help="El pronóstico no es exacto: este es su margen de error habitual.")
    c3.markdown("Movimiento esperado", help=AYUDA_NIVEL)
    c3.markdown(f"<div style='margin-top:0.35rem'>{etiqueta_nivel(fila['Nivel'])}</div>",
                unsafe_allow_html=True)
    c3.caption(f"Nivel {fila['Nivel'].lower()}")


def lista_eventos(tabla):
    """En el celular, los eventos como lista corta en lugar de tabla."""
    st.markdown("\n".join(f"- {r.Tipo.split()[0]} **{r.Fecha[:5]}** · {r.Evento} ({r.Alcance.lower()})"
                           for r in tabla.itertuples()))


def proximos_eventos():
    """Todos los eventos cargados para los meses del pronóstico."""
    filas = []
    for anio, mes in zip(pron["anio"], pron["mes"]):
        for ev in eventos_del_mes(anio, mes).itertuples():
            filas.append({"Mes": etiqueta(anio, mes), "Evento": ev.nombre,
                          "Fecha": pd.Timestamp(ev.fecha_inicio).strftime("%d/%m/%Y"),
                          "Tipo": ICONO_TIPO[ev.tipo], "Alcance": NIVELES[ev.nivel_alcance]})
    tabla = pd.DataFrame(filas).drop_duplicates(["Evento", "Fecha"]) if filas else pd.DataFrame()
    if tabla.empty:
        st.write("No hay eventos cargados para estos meses.")
    else:
        if ES_CELULAR:
            lista_eventos(tabla)
        else:
            st.table(tabla.set_index("Mes"))


def frase_resumen():
    """Una frase que dice cómo viene el mes elegido, sin números."""
    mes = mes_elegido.capitalize()
    if fila["Nivel"] == "Alto":
        texto = (f"**{mes} viene muy movido.** Es uno de los meses con más gente del año en "
                 f"{nombre_zona}.")
    elif fila["Nivel"] == "Bajo":
        texto = f"**{mes} viene tranquilo.** Se espera menos gente que en la mayoría de los meses."
    else:
        texto = f"**{mes} viene normal.** Se espera un movimiento parecido al de la mayoría de los meses."
    st.markdown(f"<div style='font-size:1.35rem; line-height:1.5; padding:0.6rem 1rem; "
                f"border-left:6px solid {COLOR_NIVEL[fila['Nivel']]}; margin:0.5rem 0 1rem 0'>"
                f"{texto.replace('**', '<b>', 1).replace('**', '</b>', 1)}</div>",
                unsafe_allow_html=True)


def comparacion_anio_anterior():
    """Compara con el mismo mes del año anterior, en camas de cada 100."""
    anio, mes = int(fila["anio"]), int(fila["mes"])
    previo = leer("""SELECT tasa_ocupacion_plazas FROM ocupacion_mensual
                     WHERE id_zona = ? AND anio = ? AND mes = ? AND origen = 'observado'""",
                  (id_zona, anio - 1, mes))
    nombre_mes = MESES[mes - 1].lower()
    if previo.empty:
        st.markdown(f"No hay dato oficial de {nombre_mes} {anio - 1} para comparar.")
        return
    antes = float(previo["tasa_ocupacion_plazas"].iloc[0])
    if abs(valor - antes) < 2:
        cambio = "Se espera algo muy parecido."
    elif valor > antes:
        cambio = "Se espera **más movimiento** que el año pasado."
    else:
        cambio = "Se espera **menos movimiento** que el año pasado."
    if pron["ocupacion_pronosticada"].max() - pron["ocupacion_pronosticada"].min() < 1:
        cambio = ("Tenga en cuenta que en esta zona el pronóstico usa un mismo valor para todos los meses, "
                  "así que no anticipa las subidas de temporada: el dato del año pasado puede orientar mejor.")
    st.markdown(f"En {nombre_mes} del año pasado se ocuparon **{antes:.0f} de cada 100 camas** en la zona. "
                f"Para {nombre_mes} {anio} se esperan **unas {valor:.0f} de cada 100**. {cambio}")


def semaforo():
    """Los doce meses en colores, de un vistazo."""
    cajas = []
    for r in pron.itertuples():
        borde = "3px solid #1f2937" if r.Mes == mes_elegido else "1px solid rgba(0,0,0,0.1)"
        cajas.append(
            f"<div style='flex:1 0 70px; max-width:110px; background:{COLOR_NIVEL[r.Nivel]}; color:white; "
            f"border:{borde}; border-radius:8px; padding:6px 4px; text-align:center; font-size:0.85rem'>"
            f"<div style='font-weight:700'>{r.Eje}</div><div>{TEXTO_NIVEL[r.Nivel]}</div></div>")
    st.subheader("📅 El año de un vistazo")
    st.markdown(f"<div style='display:flex; flex-wrap:wrap; gap:6px; margin-bottom:0.4rem'>{''.join(cajas)}</div>",
                unsafe_allow_html=True)
    st.caption("Verde: tranquilo. Amarillo: normal. Rojo: mucha gente. El recuadro oscuro es el mes elegido.")


def meses_fuertes_y_tranquilos():
    valores = pron["ocupacion_pronosticada"]
    if valores.max() - valores.min() < 1:
        st.markdown(f"En {nombre_zona}, el pronóstico espera un movimiento parecido todos los meses.")
        return
    orden = pron.sort_values("ocupacion_pronosticada")
    lista = lambda d: ", ".join(m.lower() for m in d["Mes"])
    st.markdown(f"**Más movimiento:** {lista(orden.tail(3).iloc[::-1])}.  \n"
                f"**Más tranquilos:** {lista(orden.head(3))}.")


def detalle_tecnico():
    with st.expander("Detalle técnico"):
        st.table(pd.DataFrame({
            "Modelo": [NOMBRE_MODELO[ganador["modelo"]]],
            "MAE (puntos)": [f"{ganador['mae']:.2f}"],
            "RMSE (puntos)": [f"{ganador['rmse']:.2f}"],
        }).set_index("Modelo"))
        st.caption(
            f"Variable: tasa de ocupación de plazas de la Encuesta de Ocupación Hotelera (INDEC). "
            f"Error medido en los meses de {corrida['evaluacion_desde'][:7]} a "
            f"{corrida['evaluacion_hasta'][:7]}, que no se usaron para elegir el modelo. MAE: error absoluto "
            f"medio. RMSE: raíz del error cuadrático medio. Niveles calculados con los terciles de los meses "
            f"observados de la zona hasta {origen}, sin 2020 y 2021 (bajo hasta {t1:.1f} %, medio hasta "
            f"{t2:.1f} %, alto más de {t2:.1f} %). El pronóstico corresponde a la ocupación agregada de la "
            "zona y no está validado para alojamientos informales.")


# ------------------------------------------------------------------ hospedadores
if vista == "Hospedadores":
    titulo(f"¿Cuánta ocupación se espera en {nombre_zona}?")
    avisos()
    frase_resumen()
    resumen_del_mes()
    comparacion_anio_anterior()
    st.divider()
    semaforo()
    meses_fuertes_y_tranquilos()
    st.divider()
    st.subheader("📊 Mes por mes")
    st.markdown(f"Porcentaje de camas que se espera ocupar en los hoteles y alojamientos de la zona, "
                f"de {desde.lower()} a {hasta.lower()}.")
    mostrar_grafico()

    with st.expander("¿Cómo leer esta información?"):
        st.markdown(f"""
- **Qué muestra:** cuántas camas se espera que estén ocupadas en {nombre_zona}. Por ejemplo, 40 %
  quiere decir que de cada 100 camas, unas 40 estarían ocupadas.
- **No es exacto:** puede quedar unos puntos por encima o por debajo. Por eso se muestra el margen
  (rayita naranja).
- **Movimiento bajo, medio o alto:** compara el mes con el resto del año. "Alto" es de los meses
  con más gente, pero no significa que no haya lugar.
- **Eventos:** son un dato para tener en cuenta. Que haya un evento no asegura que llegue más gente.
- **Para qué usarlo:** como una ayuda más, junto con su experiencia, para decidir precios,
  disponibilidad y promociones con tiempo.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ ¿Qué tan confiable es?")
        st.markdown(
            f"Cuando se probó con los meses de {corrida['evaluacion_desde'][:4]}, el pronóstico **se "
            f"equivocó en promedio unos {mae:.0f} puntos**, para arriba o para abajo. En los meses de "
            "más turismo puede equivocarse un poco más.")
        detalle_tecnico()
    with col2:
        st.subheader(f"🎉 Eventos en {mes_elegido.lower()}")
        eventos = eventos_del_mes(fila["anio"], fila["mes"]).copy()
        if eventos.empty:
            st.write("No hay eventos cargados para este mes.")
        else:
            eventos["Fecha"] = pd.to_datetime(eventos["fecha_inicio"]).dt.strftime("%d/%m/%Y")
            eventos["Alcance"] = eventos["nivel_alcance"].map(NIVELES)
            eventos["Tipo"] = eventos["tipo"].map(ICONO_TIPO)
            eventos["Fecha confirmada"] = eventos["estado"].map({"confirmado": "Sí", "supuesto": "A confirmar"})
            if ES_CELULAR:
                lista_eventos(eventos.rename(columns={"nombre": "Evento"}))
            else:
                st.table(eventos[["nombre", "Tipo", "Fecha", "Alcance", "Fecha confirmada"]]
                         .rename(columns={"nombre": "Evento"}).set_index("Evento"))

    st.subheader(f"🎉 Eventos de {desde.lower()} a {hasta.lower()}")
    proximos_eventos()

    st.info("Los eventos se muestran como información útil: que haya un evento no garantiza que aumente "
            "la ocupación. El pronóstico es para toda la zona y se calcula con datos de hoteles y "
            "alojamientos registrados, así que en un alojamiento en particular puede ser distinto.")

# ------------------------------------------------------------------ turistas
else:
    titulo(f"¿Cuánta gente se espera en {nombre_zona}?")
    avisos()
    frase_resumen()
    resumen_del_mes()
    if fila["Nivel"] == "Alto":
        st.info("Conviene reservar con anticipación, aunque que haya mucha gente no significa que no haya lugar.")
    elif fila["Nivel"] == "Bajo":
        st.info("Suele ser más fácil conseguir alojamiento.")
    comparacion_anio_anterior()
    st.divider()
    semaforo()
    meses_fuertes_y_tranquilos()
    st.divider()
    st.subheader("📊 Mes por mes")
    mostrar_grafico()

    with st.expander(f"🎉 Eventos de {desde.lower()} a {hasta.lower()}"):
        proximos_eventos()

    with st.expander("¿Cómo leer esta información?"):
        st.markdown(f"""
- **Qué muestra:** cuánto se espera que estén ocupados los alojamientos de {nombre_zona} en el mes
  de su viaje, según datos oficiales.
- **Movimiento bajo:** de los meses más tranquilos; suele ser más fácil conseguir lugar.
- **Movimiento alto:** de los meses con más gente; conviene reservar antes.
- **No es exacto:** puede quedar unos puntos por encima o por debajo, y no indica si un alojamiento
  en particular tiene lugar.
""")
    detalle_tecnico()

# En el celular la barra lateral queda escondida: la información de actualización va al final
if ES_CELULAR:
    st.divider()
    st.caption(f"Meses pronosticados: {desde.lower()} a {hasta.lower()}. "
               f"Última actualización: {fecha_corrida:%d/%m/%Y}. "
               "Datos oficiales del INDEC (Encuesta de Ocupación Hotelera).")
