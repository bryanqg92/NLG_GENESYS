import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FIS:
    """
    Clase que representa un sistema de inferencia difusa (FIS).
    """
    _prop = np.arange(-1.02, 1.01, 0.0007)
    _der = np.arange(-1.02, 1.01, 0.0007)
    _sal_mot = np.arange(-1.07, 1.07, 0.0007)

    # Antecedente de la entrada proporcional P y sus valores para conjuntos triangulares
    _proporcional = ctrl.Antecedent(_prop, 'proporcional')
    _proporcional['muy lejos'] = fuzz.trapmf(_proporcional.universe, [-6, -4, -1, -0.4])
    _proporcional['lejos'] = fuzz.trapmf(_proporcional.universe, [-1, -0.4, -0.4, 0])
    _proporcional['ok'] = fuzz.trapmf(_proporcional.universe, [-0.4, 0, 0, 0.4])
    _proporcional['cerca'] = fuzz.trapmf(_proporcional.universe, [0, 0.4, 0.4, 1])
    _proporcional['muy cerca'] = fuzz.trapmf(_proporcional.universe, [0.4, 1, 4, 6])

    # Se define la variable de entrada Derivativo
    _derivativo = ctrl.Antecedent(_der, 'derivativo')
    _derivativo['alejandose'] = fuzz.trapmf(_derivativo.universe, [-6, -4, -0.2, 0])
    _derivativo['sin cambio'] = fuzz.trapmf(_derivativo.universe, [-0.2, 0, 0, 0.2])
    _derivativo['acercandose'] = fuzz.trapmf(_derivativo.universe, [0, 0.2, 4, 6])

    # Se define la variable de conjuntos para los motores
    _salida = ctrl.Antecedent(_sal_mot, 'salida')
    _salida['muy rápido hacia atrás'] = fuzz.trapmf(_salida.universe, [-1.02, -1.01, -0.90, -0.80])
    _salida['bastante rápido hacia atrás'] = fuzz.trapmf(_salida.universe, [-0.90, -0.80, -0.80, -0.40])
    _salida['más o menos rápido hacia atrás'] = fuzz.trapmf(_salida.universe, [-0.80, -0.40, -0.40, -0.30])
    _salida['despacito hacia atrás'] = fuzz.trapmf(_salida.universe, [-0.40, -0.30, -0.30, -0.10])
    _salida['muy lento hacia atrás'] = fuzz.trapmf(_salida.universe, [-0.30, -0.10, -0.10, 0.10])
    _salida['muy lento hacia delante'] = fuzz.trapmf(_salida.universe, [-0.10, 0.10, 0.10, 0.30])
    _salida['despacito hacia delante'] = fuzz.trapmf(_salida.universe, [0.10, 0.30, 0.30, 0.50])
    _salida['medio rápido hacia delante'] = fuzz.trapmf(_salida.universe, [0.30, 0.50, 0.50, 0.60])
    _salida['rápidamente hacia delante'] = fuzz.trapmf(_salida.universe, [0.50, 0.60, 0.60, 0.80])
    _salida['bastante rápido hacia delante'] = fuzz.trapmf(_salida.universe, [0.60, 0.80, 0.80, 0.90])
    _salida['muy rápido hacia delante'] = fuzz.trapmf(_salida.universe, [0.80, 0.90, 1.01, 1.02])


    @classmethod
    def get_membership(cls, antecedent_name, value, method='max'):

            """
            Calcula el valor de pertenencia para un valor dado en un antecedente específico.

            Args:
                antecedent_name (str): El nombre del antecedente ('proporcional', 'derivativo' o 'salida').
                value (float): El valor para el cual se calculará la pertenencia.
                method (str, opcional): El método para calcular la pertenencia. Puede ser 'max' (valor máximo) o 'all' (todos los valores). Predeterminado a 'max'.

            Returns:
                tuple: Si method='max', devuelve una tupla con el nombre del conjunto difuso y el valor de pertenencia máximo.
                       Si method='all', devuelve un diccionario que mapea los nombres de los conjuntos difusos a sus valores de pertenencia.

            Raises:
                InvalidAntecedentError: Si se proporciona un nombre de antecedente no válido.
            """
            antecedent = None
            if antecedent_name == 'proporcional':
                antecedent = cls._proporcional
            elif antecedent_name == 'derivativo':
                antecedent = cls._derivativo
            elif antecedent_name == 'salida':
                antecedent = cls._salida
            else:
                raise InvalidAntecedentError(f"Antecedente '{antecedent_name}' no válido.")

            memberships = {term: fuzz.interp_membership(antecedent.universe, antecedent[term].mf, value) for term in antecedent.terms}

            if method == 'max':
                max_value = max(memberships.values())
                max_key = [k for k, v in memberships.items() if v == max_value][0]
                return (max_key, max_value)
            else:
                return {k: v for k, v in memberships.items() }#if v != 0.0



class InvalidAntecedentError(Exception):
        pass