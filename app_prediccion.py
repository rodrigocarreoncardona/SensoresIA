import pandas as pd
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestRegressor

ARCHIVO_BD = 'base_datos_sensores.json'

def procesar_y_guardar_archivos(carpeta_datos):
    """Busca, limpia y agrega múltiples archivos (CSV/XLSX) a la base de datos JSON."""
    
    # Buscar archivos .csv y .xlsx en la carpeta especificada
    patron_csv = os.path.join(carpeta_datos, '*.csv')
    patron_xlsx = os.path.join(carpeta_datos, '*.xlsx')
    
    archivos_encontrados = glob.glob(patron_csv) + glob.glob(patron_xlsx)
    
    if not archivos_encontrados:
        raise ValueError(f"No se encontraron archivos .csv o .xlsx en la carpeta '{carpeta_datos}'.")

    print(f"--- 1. Procesando {len(archivos_encontrados)} documentos en '{carpeta_datos}' ---")
    
    lista_dfs_limpios = []

    for ruta_archivo in archivos_encontrados:
        nombre_archivo = os.path.basename(ruta_archivo)
        if nombre_archivo.startswith('~$'):
            continue # Salta a la siguiente iteración del ciclo
        print(f" -> Leyendo: {ruta_archivo}")
        try:
            # Seleccionar la función de lectura adecuada según la extensión
            if ruta_archivo.lower().endswith('.csv'):
                df = pd.read_csv(ruta_archivo)
            else:
                df = pd.read_excel(ruta_archivo)
        except Exception as e:
            print(f" [Error] No se pudo leer {ruta_archivo}: {e}")
            continue

        df_limpio = pd.DataFrame()

        # Detectar el formato por sus columnas
        if 'Unnamed: 0' in df.columns or 'Irradiance' in str(df.columns):
            df_limpio['Lux'] = df.iloc[:, 0]
            df_limpio['Irradiancia'] = df.iloc[:, 1]
            df_limpio['Corriente'] = df.iloc[:, 2]
            df_limpio['Voltaje'] = df.iloc[:, 3]
            
            # SOLUCIÓN: Usamos -1 para tomar siempre la última columna, 
            # sin importar si el archivo tiene 7 u 8 en total.
            df_limpio['T_Panel'] = df.iloc[:, -1]
            
        elif 'Lux' in df.columns or 'lux' in df.columns.str.lower():
            df_limpio['Lux'] = df['Lux']
            df_limpio['Irradiancia'] = df['Irradiancia']
            df_limpio['Corriente'] = df['Corr'] if 'Corr' in df.columns else df['Corriente']
            df_limpio['Voltaje'] = df['Voltaje']
            df_limpio['T_Panel'] = df['T_Panel']
        else:
            print(f" [Advertencia] Formato no reconocido en {ruta_archivo}. Saltando...")
            continue

        # Limpiar nulos y convertir a numérico
        df_limpio = df_limpio.dropna(subset=['Lux', 'Irradiancia', 'T_Panel', 'Voltaje', 'Corriente'])
        for col in df_limpio.columns:
            df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
        df_limpio = df_limpio.dropna()
        
        # Agregar a la lista general si el dataframe no quedó vacío tras la limpieza
        if not df_limpio.empty:
            lista_dfs_limpios.append(df_limpio)

    if not lista_dfs_limpios:
        raise ValueError("No se pudo extraer información válida de ninguno de los archivos.")

    # Combinar todos los archivos leídos en un solo DataFrame
    df_nuevos_combinados = pd.concat(lista_dfs_limpios, ignore_index=True)

    # Lógica de Base de Datos (JSON)
    if os.path.exists(ARCHIVO_BD):
        df_historico = pd.read_json(ARCHIVO_BD)
        # Combinar el historial con el archivo nuevo y eliminar duplicados
        df_total = pd.concat([df_historico, df_nuevos_combinados]).drop_duplicates()
        print(f"Base de datos actualizada. Registros totales: {len(df_total)}")
    else:
        df_total = df_nuevos_combinados
        print(f"Creando nueva base de datos. Registros totales: {len(df_total)}")

    # Guardar en el JSON
    df_total.to_json(ARCHIVO_BD, orient='records', indent=4)
    return df_nuevos_combinados, df_total

def entrenar_ia_y_predecir(df_historico, df_nuevo):
    """Entrena el modelo con el historial y predice salidas para mostrar el avance."""
    print("\n--- 2. Entrenando Modelo de Inteligencia Artificial ---")
    
    # Variables de Entrada (Features) y Salida (Targets)
    X = df_historico[['Lux', 'Irradiancia', 'T_Panel']]
    y = df_historico[['Voltaje', 'Corriente']]

    # Usamos Random Forest porque maneja excelente las relaciones no lineales
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    print("Modelo entrenado exitosamente con el historial de datos.")

    print("\n--- 3. Generando Predicciones de Muestra ---")
    # Tomamos los últimos 10 registros de la carga nueva para hacer la demostración
    muestra_prueba = df_nuevo.tail(10).copy()
    
    # Separamos las entradas para que la IA haga la predicción "a ciegas"
    entradas_para_predecir = muestra_prueba[['Lux', 'Irradiancia', 'T_Panel']]
    
    # La IA predice el voltaje y corriente
    predicciones = modelo.predict(entradas_para_predecir)

    # Armamos una tabla comparativa para la presentación
    resultados = entradas_para_predecir.copy()
    resultados['Voltaje_REAL'] = muestra_prueba['Voltaje'].round(3)
    resultados['Voltaje_IA'] = predicciones[:, 0].round(3)
    resultados['Corriente_REAL'] = muestra_prueba['Corriente'].round(3)
    resultados['Corriente_IA'] = predicciones[:, 1].round(3)

    print("\n=== RESULTADOS: REAL vs IA ===")
    print(resultados[['Voltaje_REAL', 'Voltaje_IA', 'Corriente_REAL', 'Corriente_IA']].to_string(index=False))
    return modelo

# ==========================================
# FLUJO PRINCIPAL DE EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    carpeta_datos = 'data'
    
    try:
        # 1. Procesa todos los archivos en 'data' y los guarda en el JSON
        datos_nuevos, base_de_datos = procesar_y_guardar_archivos(carpeta_datos)
        
        # 2. Entrena y Muestra resultados
        modelo_entrenado = entrenar_ia_y_predecir(base_de_datos, datos_nuevos)
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")