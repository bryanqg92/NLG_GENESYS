import subprocess
import time

start_time = time.time()

# Ejecutar el script principal
subprocess.run(["/home/lean/miniconda3/envs/nlg/bin/python", "/home/lean/Documents/Mlops_project/PROYECTO DE GRADO/src/main.py"])

end_time = time.time()
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time:.6f} segundos")
    