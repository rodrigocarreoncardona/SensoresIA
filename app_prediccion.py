import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

ARCHIVO_BD = 'base_datos_sensores.json'

def procesar_y_guardar_excel(ruta_archivo):
    """Limpia el Excel nuevo y lo agrega a la base de datos JSON."""
    print(f"--- 1. Procesando nuevo documento: {ruta_archivo} ---")
    df = pd.read_excel(ruta_archivo)
    df_limpio = pd.DataFrame()

    # Detectar el formato por sus columnas
    if 'Unnamed: 0' in df.columns:
        df_limpio['Lux'] = df.iloc[:, 0]
        df_limpio['Irradiancia'] = df.iloc[:, 1]
        df_limpio['Corriente'] = df.iloc[:, 2]
        df_limpio['Voltaje'] = df.iloc[:, 3]
        df_limpio['T_Panel'] = df.iloc[:, 7]
    elif 'Lux' in df.columns or 'lux' in df.columns.str.lower():
        df_limpio['Lux'] = df['Lux']
        df_limpio['Irradiancia'] = df['Irradiancia']
        df_limpio['Corriente'] = df['Corr'] if 'Corr' in df.columns else df['Corriente']
        df_limpio['Voltaje'] = df['Voltaje']
        df_limpio['T_Panel'] = df['T_Panel']
    else:
        raise ValueError("Formato de archivo no reconocido.")

    # Limpiar nulos y convertir a numérico
    df_limpio = df_limpio.dropna(subset=['Lux', 'Irradiancia', 'T_Panel', 'Voltaje', 'Corriente'])
    for col in df_limpio.columns:
        df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
    df_limpio = df_limpio.dropna()

    # Lógica de Base de Datos (JSON)
    if os.path.exists(ARCHIVO_BD):
        df_historico = pd.read_json(ARCHIVO_BD)
        # Combinar el historial con el archivo nuevo y eliminar duplicados
        df_total = pd.concat([df_historico, df_limpio]).drop_duplicates()
        print(f"Base de datos actualizada. Registros totales: {len(df_total)}")
    else:
        df_total = df_limpio
        print(f"Creando nueva base de datos. Registros totales: {len(df_total)}")

    # Guardar en el JSON
    df_total.to_json(ARCHIVO_BD, orient='records', indent=4)
    return df_limpio, df_total

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
    # Tomamos los últimos 10 registros del archivo nuevo para hacer la demostración
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
    nombre_archivo_mes = '27_nov.xlsx'  # Cambia esto por el nombre de tu archivo
    
    try:
        # 1. Procesa el Excel y lo guarda en el JSON
        datos_nuevos, base_de_datos = procesar_y_guardar_excel(nombre_archivo_mes)
        
        # 2. Entrena y Muestra resultados
        modelo_entrenado = entrenar_ia_y_predecir(base_de_datos, datos_nuevos)
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")