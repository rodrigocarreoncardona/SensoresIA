import os
import json
import time
from azure.servicebus import ServiceBusClient
from main import procesar_y_guardar_archivos, entrenar_ia_y_predecir, CARPETA_DATOS, CARPETA_RESULTADOS

SERVICE_BUS_CONN_STR = os.environ.get("SERVICE_BUS_CONN_STR", "")
NOMBRE_COLA = "cola-predicciones"

def iniciar_worker():
    if not SERVICE_BUS_CONN_STR:
        print("Error: No se encontró SERVICE_BUS_CONN_STR. Worker detenido.")
        return

    print("Iniciando Worker... Escuchando la cola de Azure Service Bus.")
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    
    with ServiceBusClient.from_connection_string(SERVICE_BUS_CONN_STR) as client:
        # max_wait_time mantiene la conexión viva esperando mensajes
        with client.get_queue_receiver(queue_name=NOMBRE_COLA, max_wait_time=10) as receiver:
            while True:
                # Extraemos 1 mensaje de la cola
                mensajes = receiver.receive_messages(max_message_count=1, max_wait_time=5)
                
                for msg in mensajes:
                    tarea_id = str(msg)
                    print(f"\n[+] Iniciando análisis pesado para tarea: {tarea_id}")
                    
                    try:
                        # Ejecutamos las funciones importadas desde main.py
                        datos_nuevos, base_de_datos = procesar_y_guardar_archivos(CARPETA_DATOS)
                        resultados = entrenar_ia_y_predecir(base_de_datos, datos_nuevos)
                        
                        # Guardamos el resultado en un JSON nombrado con el ID de la tarea
                        ruta_guardado = os.path.join(CARPETA_RESULTADOS, f"{tarea_id}.json")
                        with open(ruta_guardado, "w") as archivo:
                            json.dump(resultados, archivo, indent=4)
                            
                        # Confirmamos a Azure que el trabajo terminó para borrar el mensaje de la cola
                        receiver.complete_message(msg)
                        print(f"[-] Tarea {tarea_id} completada exitosamente.")
                        
                    except Exception as e:
                        print(f"[!] Error en tarea {tarea_id}: {str(e)}")
                        # Si falla, se envía a la cola de mensajes muertos (Dead Letter)
                        receiver.dead_letter_message(msg, reason="Fallo en procesamiento de IA")
                
                # Pequeña pausa para no saturar el CPU
                time.sleep(2)

if __name__ == "__main__":
    iniciar_worker()