"""
PASO 3 - Modelos de pronóstico (Sprints 2 y 3, historias PB-05, PB-06 y PB-07)

Tres modelos por zona:
  - base: suavizado exponencial simple (statsmodels). Proyecta el mismo valor
    para los doce meses.
  - xgboost_sin_eventos: XGBoost con el mes del año y la ocupación del mismo
    mes uno y dos años antes (rezagos de 12 y 24 meses).
  - xgboost_con_eventos: lo mismo, más la cantidad de eventos de cada nivel de
    alcance en la zona y el mes. Para los meses a pronosticar solo se cuentan
    los eventos cuya fecha de publicación es anterior o igual al origen.

Evaluación (como dice la tesis):
  1. Corrida de selección: se entrena hasta dic-2023 y se pronostica 2024.
     Con 2024 se eligen los hiperparámetros de XGBoost y el modelo ganador de
     cada zona (menor MAE; si empatan, menor RMSE).
  2. Corrida de evaluación final: se reentrena hasta dic-2024 y se pronostica
     2025. Ese MAE y RMSE es el resultado que se informa.
  3. Corrida de producción: se reentrena hasta el último mes publicado y se
     pronostican los doce meses siguientes. Es la que muestra la aplicación.
     Los modelos de esta corrida se guardan en la carpeta modelos/.

Los meses imputados de 2020-2021 se usan como rezagos, pero no como meses a
aprender, porque no son observaciones de la EOH.

Cómo se corre:   python 03_entrenar_modelos.py
"""

import json
import warnings
from datetime import datetime
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from xgboost import XGBRegressor

from base import CARPETA, FIGURAS, MODELOS, PROCESADOS, ZONAS, conectar

warnings.filterwarnings("ignore")
VERSION = "v1.0"
SEMILLA = 42
NIVELES = ["eventos_nivel_1", "eventos_nivel_2", "eventos_nivel_3", "eventos_nivel_4"]
GRILLA = {"max_depth": [2, 3, 4], "n_estimators": [100, 300], "learning_rate": [0.05, 0.1]}
NOMBRE_MODELO = {"base": "Suavizado exponencial simple",
                 "xgboost_sin_eventos": "XGBoost sin eventos",
                 "xgboost_con_eventos": "XGBoost con eventos"}


# ------------------------------------------------------------ datos
def periodo(anio, mes):
    return pd.Period(year=int(anio), month=int(mes), freq="M")


def cargar_datos(con):
    serie = pd.read_sql("SELECT * FROM ocupacion_mensual", con)
    serie["periodo"] = [periodo(a, m) for a, m in zip(serie.anio, serie.mes)]
    eventos = pd.read_sql("SELECT * FROM evento", con)
    cargas = pd.read_sql("SELECT * FROM carga_datos", con)
    id_serie = int(cargas[cargas.tipo == "serie_eoh"].id_carga.max())
    id_cal = int(cargas[cargas.tipo == "calendario"].id_carga.max())
    return serie, eventos, id_serie, id_cal


def contar_eventos(eventos, id_zona, per, origen=None):
    """Eventos de cada nivel en la zona y el mes. Con origen, solo los publicados hasta ese mes."""
    ev = eventos[eventos.id_zona == id_zona]
    if origen is not None:
        limite = origen.end_time.date().isoformat()
        ev = ev[ev.fecha_publicacion <= limite]
    desde, hasta = per.start_time.date().isoformat(), per.end_time.date().isoformat()
    ev = ev[(ev.fecha_inicio <= hasta) & (ev.fecha_fin >= desde)]
    return [int((ev.nivel_alcance == n).sum()) for n in (1, 2, 3, 4)]


def tabla_entrada(serie_zona, eventos, id_zona, periodos, origen=None):
    """Mes del año, rezagos de 12 y 24 meses y eventos por nivel para cada período."""
    tasa = dict(zip(serie_zona.periodo, serie_zona.tasa_ocupacion_plazas))
    filas = []
    for per in periodos:
        filas.append([per.month, tasa.get(per - 12, np.nan), tasa.get(per - 24, np.nan)]
                     + contar_eventos(eventos, id_zona, per, origen))
    return pd.DataFrame(filas, index=periodos, columns=["mes", "rezago_12", "rezago_24"] + NIVELES)


# ------------------------------------------------------------ modelos
def entrenar_y_pronosticar(serie_zona, eventos, id_zona, corte, horizonte, params, carpeta_modelos=None):
    """Entrena los tres modelos con datos hasta `corte` y pronostica los períodos del horizonte.
    Si se indica `carpeta_modelos`, guarda ahí los modelos entrenados."""
    hist = serie_zona[serie_zona.periodo <= corte]
    resultado = {}

    # 1) Modelo base: suavizado exponencial simple sobre los meses observados
    obs = hist[hist.origen == "observado"].set_index("periodo")["tasa_ocupacion_plazas"]
    ses = SimpleExpSmoothing(obs.values, initialization_method="estimated").fit()
    resultado["base"] = np.repeat(ses.forecast(1)[0], len(horizonte))
    if carpeta_modelos:
        with open(carpeta_modelos / "base.json", "w", encoding="utf-8") as f:
            json.dump({"modelo": "Suavizado exponencial simple",
                       "alfa": round(float(ses.params["smoothing_level"]), 4),
                       "nivel_final": round(float(ses.forecast(1)[0]), 2),
                       "entrenado_hasta": str(corte)}, f, ensure_ascii=False, indent=2)

    # 2) y 3) XGBoost: se aprende de meses observados que tengan los dos rezagos
    entren = [p for p in hist[hist.origen == "observado"].periodo]
    X = tabla_entrada(serie_zona, eventos, id_zona, entren).dropna()
    y = hist.set_index("periodo").loc[X.index, "tasa_ocupacion_plazas"]
    X_h = tabla_entrada(serie_zona, eventos, id_zona, horizonte, origen=corte)
    for modelo, columnas in (("xgboost_sin_eventos", ["mes", "rezago_12", "rezago_24"]),
                             ("xgboost_con_eventos", ["mes", "rezago_12", "rezago_24"] + NIVELES)):
        xgb = XGBRegressor(**params[modelo], random_state=SEMILLA, n_jobs=1)
        xgb.fit(X[columnas], y)
        resultado[modelo] = xgb.predict(X_h[columnas])
        if carpeta_modelos:
            xgb.save_model(carpeta_modelos / f"{modelo}.json")

    return {m: np.clip(np.round(v, 2), 0, 100) for m, v in resultado.items()}


def errores(real, pron):
    return (round(mean_absolute_error(real, pron), 2),
            round(float(np.sqrt(mean_squared_error(real, pron))), 2))


def ajustar_hiperparametros(serie_zona, eventos, id_zona, corte, prueba):
    """Prueba cada combinación de la grilla en 2024 y se queda con la de menor MAE."""
    real = serie_zona.set_index("periodo").loc[prueba, "tasa_ocupacion_plazas"].values
    mejores = {}
    for modelo in ("xgboost_sin_eventos", "xgboost_con_eventos"):
        mejor = None
        for md, ne, lr in product(GRILLA["max_depth"], GRILLA["n_estimators"], GRILLA["learning_rate"]):
            p = {"max_depth": md, "n_estimators": ne, "learning_rate": lr}
            pron = entrenar_y_pronosticar(serie_zona, eventos, id_zona, corte, prueba,
                                          {"xgboost_sin_eventos": p, "xgboost_con_eventos": p})[modelo]
            mae, rmse = errores(real, pron)
            if mejor is None or (mae, rmse) < mejor[0]:
                mejor = ((mae, rmse), p)
        mejores[modelo] = mejor[1]
    return mejores


def elegir_ganador(metricas):
    """Menor MAE; si hay empate, menor RMSE."""
    return min(metricas, key=lambda m: (metricas[m][0], metricas[m][1]))


# ------------------------------------------------------------ base de datos
def registrar_corrida(con, corte, desde, hasta, version, id_serie, id_cal):
    cur = con.execute("""
        INSERT INTO corrida (fecha_corrida, corte_entrenamiento, evaluacion_desde, evaluacion_hasta,
                             version, id_carga_serie, id_carga_calendario)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(timespec="seconds"), corte.start_time.date().isoformat(),
         desde.start_time.date().isoformat(), hasta.start_time.date().isoformat(),
         version, id_serie, id_cal))
    return cur.lastrowid


def guardar_pronosticos(con, id_corrida, id_zona, horizonte, pronosticos):
    con.executemany(
        "INSERT INTO pronostico VALUES (?, ?, ?, ?, ?, ?)",
        [(id_corrida, id_zona, m, p.year, p.month, float(v))
         for m, valores in pronosticos.items() for p, v in zip(horizonte, valores)])


def guardar_metricas(con, id_corrida, id_zona, metricas, ganador):
    con.executemany(
        "INSERT INTO metrica_modelo VALUES (?, ?, ?, ?, ?, ?)",
        [(id_corrida, id_zona, m, mae, rmse, int(m == ganador)) for m, (mae, rmse) in metricas.items()])


# ------------------------------------------------------------ programa
def main():
    con = conectar()
    serie, eventos, id_serie, id_cal = cargar_datos(con)
    ultimo = serie[serie.origen == "observado"].periodo.max()

    corte_sel, prueba_sel = periodo(2023, 12), list(pd.period_range("2024-01", "2024-12", freq="M"))
    corte_eval = periodo(2024, 12)
    prueba_eval = [p for p in pd.period_range("2025-01", "2025-12", freq="M") if p <= ultimo]
    horizonte = list(pd.period_range(ultimo + 1, ultimo + 12, freq="M"))

    hiper, ganadores, filas_resumen, graficos = {}, {}, [], {}

    # --- 1. selección con 2024
    id_sel = registrar_corrida(con, corte_sel, prueba_sel[0], prueba_sel[-1],
                               f"{VERSION} seleccion", id_serie, id_cal)
    for hoja, zona in ZONAS.items():
        iz = zona["id_zona"]
        sz = serie[serie.id_zona == iz].sort_values("periodo")
        hiper[iz] = ajustar_hiperparametros(sz, eventos, iz, corte_sel, prueba_sel)
        pron = entrenar_y_pronosticar(sz, eventos, iz, corte_sel, prueba_sel, hiper[iz])
        real = sz.set_index("periodo").loc[prueba_sel, "tasa_ocupacion_plazas"].values
        met = {m: errores(real, pron[m]) for m in MODELOS}
        ganadores[iz] = elegir_ganador(met)
        guardar_pronosticos(con, id_sel, iz, prueba_sel, pron)
        guardar_metricas(con, id_sel, iz, met, ganadores[iz])
        for m in MODELOS:
            filas_resumen.append([zona["nombre"], "2024 (selección)", NOMBRE_MODELO[m], *met[m],
                                  "sí" if m == ganadores[iz] else ""])

    # --- 2. evaluación final con 2025
    id_eval = registrar_corrida(con, corte_eval, prueba_eval[0], prueba_eval[-1],
                                f"{VERSION} evaluacion " + json.dumps(hiper), id_serie, id_cal)
    met_eval = {}
    for hoja, zona in ZONAS.items():
        iz = zona["id_zona"]
        sz = serie[serie.id_zona == iz].sort_values("periodo")
        pron = entrenar_y_pronosticar(sz, eventos, iz, corte_eval, prueba_eval, hiper[iz])
        real = sz.set_index("periodo").loc[prueba_eval, "tasa_ocupacion_plazas"].values
        met_eval[iz] = {m: errores(real, pron[m]) for m in MODELOS}
        guardar_pronosticos(con, id_eval, iz, prueba_eval, pron)
        guardar_metricas(con, id_eval, iz, met_eval[iz], ganadores[iz])
        graficos[zona["nombre"]] = (prueba_eval, real, pron)
        for m in MODELOS:
            filas_resumen.append([zona["nombre"], "2025 (evaluación final)", NOMBRE_MODELO[m],
                                  *met_eval[iz][m], "sí" if m == ganadores[iz] else ""])

    # --- 3. producción: doce meses desde el último mes publicado
    id_prod = registrar_corrida(con, ultimo, prueba_eval[0], prueba_eval[-1],
                                f"{VERSION} produccion " + json.dumps(hiper), id_serie, id_cal)
    for hoja, zona in ZONAS.items():
        iz = zona["id_zona"]
        sz = serie[serie.id_zona == iz].sort_values("periodo")
        carpeta = CARPETA / "modelos" / ("ciudad" if iz == 1 else "termas")
        carpeta.mkdir(parents=True, exist_ok=True)
        pron = entrenar_y_pronosticar(sz, eventos, iz, ultimo, horizonte, hiper[iz], carpeta)
        guardar_pronosticos(con, id_prod, iz, horizonte, pron)
        guardar_metricas(con, id_prod, iz, met_eval[iz], ganadores[iz])

    con.commit()
    con.close()

    # --- salidas para la tesis
    PROCESADOS.mkdir(parents=True, exist_ok=True)
    resumen = pd.DataFrame(filas_resumen, columns=["Zona", "Período", "Modelo", "MAE", "RMSE", "Ganador"])
    resumen.to_csv(PROCESADOS / "metricas_modelos.csv", index=False, encoding="utf-8-sig")
    with open(PROCESADOS / "hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump({ZONAS[h]["nombre"]: hiper[z["id_zona"]] for h, z in ZONAS.items()}, f,
                  ensure_ascii=False, indent=2)
    graficar(graficos)

    print(f"Corridas registradas: selección N.º {id_sel}, evaluación N.º {id_eval}, producción N.º {id_prod}")
    print(f"Último mes publicado (origen): {ultimo}. Pronóstico: {horizonte[0]} a {horizonte[-1]}\n")
    print(resumen.to_string(index=False))
    print("\nMAE y RMSE en puntos porcentuales de ocupación. Tabla en datos/procesados/metricas_modelos.csv")
    print("Gráfico de 2025 en figuras/evaluacion_2025.png")
    print("Modelos entrenados guardados en la carpeta modelos/ (ciudad y termas)")


def graficar(graficos):
    FIGURAS.mkdir(exist_ok=True)
    fig, ejes = plt.subplots(len(graficos), 1, figsize=(11, 7), sharex=True)
    estilos = {"base": ("#9a9a9a", "--"), "xgboost_sin_eventos": ("#e08a1e", "-"),
               "xgboost_con_eventos": ("#1f5fa8", "-")}
    for eje, (nombre, (per, real, pron)) in zip(ejes, graficos.items()):
        x = [p.to_timestamp() for p in per]
        eje.plot(x, real, color="black", marker="o", linewidth=2, label="Ocupación real (INDEC)")
        for m, (c, ls) in estilos.items():
            eje.plot(x, pron[m], color=c, linestyle=ls, marker=".", label=NOMBRE_MODELO[m])
        eje.set_title(nombre)
        eje.set_ylabel("Ocupación de plazas (%)")
        eje.set_ylim(0, max(70, float(np.nanmax(real)) + 10))
        eje.grid(alpha=0.3)
    ejes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURAS / "evaluacion_2025.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
