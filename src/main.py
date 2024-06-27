import time
from predict.predictor import Predictor

pdlr = [0.034703177372863436, 0.1794852635619095, -0.15792089491801553, 0.5969408437843678]

# Obtener el tiempo inicial antes de cualquier otra operación
start_time = time.time()

def main():
    predictor = Predictor('src/models/seq2seqLSTM_model.pt', 'src/data_utils/TextTokenizer/tokenizer.json')
    print(predictor.predict(pdlr))

    end_time = time.time()  # Obtener el tiempo actual después de ejecutar el código
    execution_time = end_time - start_time  # Calcular el tiempo de ejecución
    print(f"Tiempo de ejecución: {execution_time:.6f} segundos")

if __name__ == "__main__":
    main()