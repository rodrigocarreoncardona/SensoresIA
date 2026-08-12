from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
import pandas as pd
import numpy as np
import os
import glob
import shutil
from sklearn.ensemble import RandomForestRegressor

# Inicializamos la API
app = FastAPI(
    title="API de Predicción de Paneles Solares",
    description="Procesa datos de sensores y genera predicciones de voltaje y corriente."
)

ARCHIVO_BD = 'base_datos_sensores.json'
CARPETA_DATOS = 'data'

def procesar_y_guardar_archivos(carpeta_datos):
    """Busca, limpia y agrega múltiples archivos (CSV/XLSX) a la base de datos JSON."""
    patron_csv = os.path.join(carpeta_datos, '*.csv')
    patron_xlsx = os.path.join(carpeta_datos, '*.xlsx')
    archivos_encontrados = glob.glob(patron_csv) + glob.glob(patron_xlsx)
    
    if not archivos_encontrados:
        raise ValueError(f"No se encontraron archivos en la carpeta '{carpeta_datos}'.")

    lista_dfs_limpios = []

    for ruta_archivo in archivos_encontrados:
        nombre_archivo = os.path.basename(ruta_archivo)
        if nombre_archivo.startswith('~$'):
            continue 
            
        try:
            if ruta_archivo.lower().endswith('.csv'):
                df = pd.read_csv(ruta_archivo)
            else:
                df = pd.read_excel(ruta_archivo)
        except Exception:
            continue

        df_limpio = pd.DataFrame()

        df.columns = df.columns.str.lower()

        # Detectar el formato por sus columnas
        if 'unnamed: 0' in df.columns or 'irradiance' in df.columns:
            df_limpio['Lux'] = df.iloc[:, 0]
            df_limpio['Irradiancia'] = df.iloc[:, 1]
            df_limpio['Corriente'] = df.iloc[:, 2]
            df_limpio['Voltaje'] = df.iloc[:, 3]
            df_limpio['T_Panel'] = df.iloc[:, -1]
            
        elif 'lux' in df.columns:
            df_limpio['Lux'] = df['lux']
            df_limpio['Irradiancia'] = df['irradiancia']
            
            df_limpio['Corriente'] = df['corr'] if 'corr' in df.columns else df['corriente']
            
            df_limpio['Voltaje'] = df['voltaje']
            df_limpio['T_Panel'] = df['t_panel']
        else:
            continue

        # Limpiar nulos y convertir a numérico
        df_limpio = df_limpio.dropna(subset=['Lux', 'Irradiancia', 'T_Panel', 'Voltaje', 'Corriente'])
        for col in df_limpio.columns:
            df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
        df_limpio = df_limpio.dropna()
        
        if not df_limpio.empty:
            lista_dfs_limpios.append(df_limpio)

    if not lista_dfs_limpios:
        raise ValueError("No se pudo extraer información válida de ninguno de los archivos.")

    # Combinamos todos los archivos que existen ACTUALMENTE en la carpeta
    df_total = pd.concat(lista_dfs_limpios, ignore_index=True)

    # Sobrescribimos el JSON por completo para sincronizarlo con los archivos físicos
    df_total.to_json(ARCHIVO_BD, orient='records', indent=4)
    
    # Retornamos df_total dos veces para no romper las funciones de entrenamiento
    return df_total, df_total

def entrenar_ia_y_predecir(df_historico, df_nuevo):
    """Entrena el modelo y devuelve las predicciones en formato diccionario."""
    X = df_historico[['Lux', 'Irradiancia', 'T_Panel']]
    y = df_historico[['Voltaje', 'Corriente']]

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    muestra_prueba = df_nuevo.tail(10).copy()
    entradas_para_predecir = muestra_prueba[['Lux', 'Irradiancia', 'T_Panel']]
    
    predicciones = modelo.predict(entradas_para_predecir)

    resultados = entradas_para_predecir.copy()
    resultados['Voltaje_REAL'] = muestra_prueba['Voltaje'].round(3)
    resultados['Voltaje_IA'] = predicciones[:, 0].round(3)
    resultados['Corriente_REAL'] = muestra_prueba['Corriente'].round(3)
    resultados['Corriente_IA'] = predicciones[:, 1].round(3)

    # Convertimos el DataFrame a una lista de diccionarios para la respuesta JSON
    columnas_retorno = ['Voltaje_REAL', 'Voltaje_IA', 'Corriente_REAL', 'Corriente_IA']
    return resultados[columnas_retorno].to_dict(orient='records')

# ==========================================
# ENDPOINTS DE LA API
# ==========================================
@app.post("/procesar-predicciones")
def ejecutar_prediccion():
    """
    Endpoint que lee la carpeta, actualiza el JSON histórico, 
    entrena el modelo y devuelve la comparativa de predicciones.
    """
    try:
        datos_nuevos, base_de_datos = procesar_y_guardar_archivos(CARPETA_DATOS)
        resultados_comparativos = entrenar_ia_y_predecir(base_de_datos, datos_nuevos)
        
        return {
            "estado": "exito",
            "mensaje": "Modelo entrenado y predicciones generadas correctamente.",
            "registros_totales_bd": len(base_de_datos),
            "predicciones": resultados_comparativos
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno: {str(e)}")

@app.post("/subir-y-procesar")
async def subir_y_procesar(archivos: List[UploadFile] = File(...)):
    """
    Endpoint que recibe archivos, los guarda en la carpeta 'data' 
    y ejecuta el ciclo completo de procesamiento y predicción.
    """
    # Nos aseguramos de que la carpeta exista antes de intentar guardar
    os.makedirs(CARPETA_DATOS, exist_ok=True)
    
    archivos_guardados = []
    
    # Guardamos cada archivo recibido en la carpeta local
    for archivo in archivos:
        ruta_destino = os.path.join(CARPETA_DATOS, archivo.filename)
        try:
            # Se abre el archivo de destino en modo "wb" (Write Binary)
            with open(ruta_destino, "wb") as buffer:
                # Se copia el flujo de bytes (stream) directamente, sin convertir a strings
                shutil.copyfileobj(archivo.file, buffer)
            archivos_guardados.append(archivo.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al guardar {archivo.filename}: {str(e)}")
        finally:
            archivo.file.close()

    # Una vez guardados los archivos, reutilizamos la lógica de tu código base
    try:
        datos_nuevos, base_de_datos = procesar_y_guardar_archivos(CARPETA_DATOS)
        resultados_comparativos = entrenar_ia_y_predecir(base_de_datos, datos_nuevos)
        
        return {
            "estado": "exito",
            "mensaje": f"Se subieron {len(archivos_guardados)} archivos nuevos y se actualizaron las predicciones.",
            "archivos_subidos": archivos_guardados,
            "registros_totales_bd": len(base_de_datos),
            "predicciones": resultados_comparativos
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno al procesar: {str(e)}")