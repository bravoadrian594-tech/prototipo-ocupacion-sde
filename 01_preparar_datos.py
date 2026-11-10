"""
Paso 1: arma la serie mensual de ocupacion con las planillas de la EOH.

Descomprime los .zip del INDEC (desde 2018 vienen asi), toma de la fila "Total"
las plazas disponibles y ocupadas de cada zona y mes, y calcula la tasa
(ocupadas / disponibles x 100). Los meses de 2020 y 2021 se reemplazan por el
promedio del mismo mes en 2017-2019, pero el valor original queda guardado junto
con la regla que se uso. Despues registra la carga, guarda la serie en la base y
hace el grafico figuras/serie_ocupacion.png.

Uso: python 01_preparar_datos.py
"""

import re
import zipfile
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from base import FIGURAS, INDEC, PROCESADOS, ZONAS, conectar

FUENTE = "INDEC, Encuesta de Ocupación Hotelera, https://www.indec.gob.ar/Nivel4/Tema/3/13/56"
ANIOS_PANDEMIA = [2020, 2021]
ANIOS_REFERENCIA = [2017, 2018, 2019]
REGLA = "promedio del mismo mes 2017-2019"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
EXTRAIDOS = INDEC / "extraidos"


# --- lectura
def descomprimir():
    """Saca las planillas de oferta y demanda de cada .zip (una sola vez)."""
    EXTRAIDOS.mkdir(exist_ok=True)
    for z in INDEC.glob("*.zip"):
        with zipfile.ZipFile(z) as archivo:
            for nombre in archivo.namelist():
                destino = EXTRAIDOS / nombre.replace("\\", "/").split("/")[-1]
                es_oferta = "oyd" in nombre.lower() or "ofyde" in nombre.lower()
                if nombre.lower().endswith(".xls") and es_oferta and not destino.exists():
                    destino.write_bytes(archivo.read(nombre))


def buscar_fila(tabla, texto):
    """Primera fila cuya columna 0 contiene el texto (sin importar mayusculas)."""
    for fila in range(len(tabla)):
        if texto.lower() in str(tabla.iat[fila, 0]).lower():
            return fila
    raise ValueError(f"No se encontró el bloque '{texto}'")


def columnas_de_meses(tabla):
    """Columna de cada mes; None si el INDEC todavia no lo publico."""
    for fila in range(12):
        valores = [str(v).strip().lower() for v in tabla.iloc[fila].tolist()]
        if "enero" in valores:
            return [valores.index(m) if m in valores else None for m in MESES]
    raise ValueError("No se encontró la fila de los meses")


def valores_total(tabla, texto, columnas):
    """Los 12 valores de la fila Total del bloque. '-' o '///' quedan vacios."""
    fila = buscar_fila(tabla, texto) + 1
    if "total" not in str(tabla.iat[fila, 1]).lower():
        raise ValueError(f"Debajo de '{texto}' no está la fila Total")
    return [pd.to_numeric(tabla.iat[fila, c], errors="coerce") if c is not None else None
            for c in columnas]


def leer_hoja(archivo, hoja):
    tabla = pd.read_excel(archivo, sheet_name=hoja, header=None, engine="calamine")
    anio = int(re.search(r"Año (\d{4})", str(tabla.iat[1, 0])).group(1))
    columnas = columnas_de_meses(tabla)
    return pd.DataFrame({
        "anio": anio,
        "mes": range(1, 13),
        "plazas_disponibles": valores_total(tabla, "Plazas disponibles", columnas),
        "plazas_ocupadas": valores_total(tabla, "Plazas ocupadas", columnas),
    })


def leer_serie():
    archivos = sorted(INDEC.glob("*.xls")) + sorted(EXTRAIDOS.glob("*.xls"))
    partes = []
    for archivo in archivos:
        hojas = pd.ExcelFile(archivo, engine="calamine").sheet_names
        for hoja, zona in ZONAS.items():
            if hoja in hojas:
                df = leer_hoja(archivo, hoja)
                df["id_zona"] = zona["id_zona"]
                partes.append(df)
    serie = pd.concat(partes, ignore_index=True)
    # Un mes sin plazas disponibles no fue publicado (o no tiene dato)
    serie = serie.dropna(subset=["plazas_disponibles"])
    serie = serie[serie["plazas_disponibles"] > 0]
    # Si un año aparece en dos archivos, se queda el mas reciente
    serie = serie.drop_duplicates(["id_zona", "anio", "mes"], keep="last")
    return serie.sort_values(["id_zona", "anio", "mes"]).reset_index(drop=True)


# --- tasa e imputacion
def preparar(serie):
    serie["tasa_original"] = (serie["plazas_ocupadas"] / serie["plazas_disponibles"] * 100).round(2)
    serie["tasa_ocupacion_plazas"] = serie["tasa_original"]
    serie["origen"] = "observado"
    serie["regla_imputacion"] = None

    referencia = (serie[serie["anio"].isin(ANIOS_REFERENCIA)]
                  .groupby(["id_zona", "mes"])["tasa_original"].mean())

    # Se completan los 24 meses de 2020-2021 aunque el INDEC no haya publicado alguno
    filas = []
    for z in ZONAS.values():
        for anio in ANIOS_PANDEMIA:
            for mes in range(1, 13):
                filas.append({"id_zona": z["id_zona"], "anio": anio, "mes": mes})
    pandemia = pd.DataFrame(filas).merge(serie, on=["id_zona", "anio", "mes"], how="left")
    pandemia["tasa_ocupacion_plazas"] = [
        round(referencia[(r.id_zona, r.mes)], 2) for r in pandemia.itertuples()]
    pandemia["origen"] = "imputado"
    pandemia["regla_imputacion"] = REGLA

    resto = serie[~serie["anio"].isin(ANIOS_PANDEMIA)]
    serie = pd.concat([resto, pandemia], ignore_index=True)
    return serie.sort_values(["id_zona", "anio", "mes"]).reset_index(drop=True)


# --- guardar
def guardar(serie):
    observados = serie[serie["origen"] == "observado"]
    ultimo = observados.sort_values(["anio", "mes"]).iloc[-1]
    fecha_corte = f"{int(ultimo.anio)}-{int(ultimo.mes):02d}-01"

    con = conectar()
    cur = con.execute(
        "INSERT INTO carga_datos (tipo, fuente, fecha_corte, fecha_carga) VALUES ('serie_eoh', ?, ?, ?)",
        (FUENTE, fecha_corte, date.today().isoformat()))
    id_carga = cur.lastrowid

    filas = []
    for r in serie.itertuples():
        filas.append((
            int(r.id_zona), int(r.anio), int(r.mes),
            None if pd.isna(r.plazas_disponibles) else int(r.plazas_disponibles),
            None if pd.isna(r.plazas_ocupadas) else int(r.plazas_ocupadas),
            float(r.tasa_ocupacion_plazas), r.origen,
            None if pd.isna(r.tasa_original) else float(r.tasa_original),
            r.regla_imputacion, id_carga))
    con.executemany("""
        INSERT OR REPLACE INTO ocupacion_mensual
        (id_zona, anio, mes, plazas_disponibles, plazas_ocupadas, tasa_ocupacion_plazas,
         origen, tasa_original, regla_imputacion, id_carga)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", filas)
    con.commit()
    con.close()

    PROCESADOS.mkdir(parents=True, exist_ok=True)
    serie.to_csv(PROCESADOS / "ocupacion_mensual.csv", index=False, encoding="utf-8-sig")
    return id_carga, fecha_corte


def graficar(serie):
    FIGURAS.mkdir(exist_ok=True)
    fig, ejes = plt.subplots(len(ZONAS), 1, figsize=(11, 7), sharex=True)
    for eje, zona in zip(ejes, ZONAS.values()):
        d = serie[serie["id_zona"] == zona["id_zona"]].copy()
        d["fecha"] = pd.to_datetime(dict(year=d["anio"], month=d["mes"], day=1))
        eje.plot(d["fecha"], d["tasa_original"], color="#9a9a9a", label="Dato publicado por el INDEC")
        eje.plot(d["fecha"], d["tasa_ocupacion_plazas"], color="#1f5fa8", label="Serie usada en los modelos")
        eje.set_title(zona["nombre"])
        eje.set_ylabel("Ocupación de plazas (%)")
        eje.set_ylim(0, 100)
        eje.grid(alpha=0.3)
    ejes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURAS / "serie_ocupacion.png", dpi=150)
    plt.close(fig)



if __name__ == "__main__":
    descomprimir()
    serie = preparar(leer_serie())
    id_carga, corte = guardar(serie)
    graficar(serie)

    print(f"Carga N.º {id_carga} registrada. Último mes publicado: {corte[:7]}")
    resumen = serie.groupby(["id_zona", "origen"]).size().unstack(fill_value=0)
    resumen.index = [z["nombre"] for z in ZONAS.values()]
    print("\nMeses por zona:")
    print(resumen.to_string())
    print("\nSerie guardada en la base (tabla ocupacion_mensual) y en datos/procesados/ocupacion_mensual.csv")
    print("Gráfico guardado en figuras/serie_ocupacion.png")
