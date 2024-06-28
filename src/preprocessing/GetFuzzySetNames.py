from src.data_utils.fis.fis import FIS

"""
Diccionario que mapea los nombres de los antecedentes a sus respectivos índices.
"""
antecedent_names = {0: 'proporcional', 1: 'derivativo', 2: 'salida', 3: 'salida'}

def get_fuzzy_names(pdlr_values: list) -> str:
    """
    Obtiene los nombres de los conjuntos difusos correspondientes a los valores de entrada.

    Args:
        pdlr_values (list): Lista de valores de entrada (proporcional, derivativo, salida, salida).

    Returns:
        str: Cadena de texto que contiene los nombres de los conjuntos difusos separados por espacios.

    Raises:
        ValueError: Si la longitud de la lista de valores de entrada no es 4.
    """
    if len(pdlr_values) != 4:
        raise ValueError("La lista de valores de entrada debe tener una longitud de 4.")

    fuzzy_names = []

    for i, value in enumerate(pdlr_values): 
        if i in antecedent_names:
            antecedent_name = antecedent_names[i]
            name, _ = FIS.get_membership(antecedent_name=antecedent_name, value=value)
            fuzzy_names.append(name)

    fuzzy_names = ' '.join(fuzzy_names)

    return [fuzzy_names]
