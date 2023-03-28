import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd

class fis:

    def __init__(self):
        self.prop = np.arange(-2,2,0.0007)
        self.der = np.arange(-2,2,0.0007)
        self.sal_mot = np.arange(-100,100.07,0.07)

        # Antecedente de la entrada proporcional P y sus valores para conjuntos triangulares

        self.proporcional = ctrl.Antecedent(self.prop, 'proporcional')
        self.proporcional['muy lejos'] = fuzz.trapmf(self.proporcional.universe, [-6, -4, -1, -0.4])
        self.proporcional['lejos'] = fuzz.trapmf(self.proporcional.universe, [-1, -0.4, -0.4, 0])
        self.proporcional['ok'] = fuzz.trapmf(self.proporcional.universe, [-0.4, 0, 0, 0.4])
        self.proporcional['cerca'] = fuzz.trapmf(self.proporcional.universe, [0, 0.4, 0.4, 1])
        self.proporcional['muy cerca'] = fuzz.trapmf(self.proporcional.universe, [0.4, 1, 4, 6])


        # Se define la variable de entrada Derivativo
        self.derivativo = ctrl.Antecedent(self.der, 'derivativo')
        self.derivativo['alejandose'] = fuzz.trapmf(self.derivativo.universe, [-6,-4,-0.2, 0])
        self.derivativo['sin cambio'] = fuzz.trapmf(self.derivativo.universe, [-0.2, 0, 0, 0.2])
        self.derivativo['acercandose'] = fuzz.trapmf(self.derivativo.universe, [0, 0.2, 4, 6])

        # Se define la variable de conjutns para los motores
        self.salida = ctrl.Antecedent(self.sal_mot, 'salida')
        self.salida['muy rápido hacia atrás'] = fuzz.trapmf(self.salida.universe, [-102,-101,-90,-80])
        self.salida['bastante rápido hacia atrás'] = fuzz.trapmf(self.salida.universe, [-90,-80,-80,-40])
        self.salida['más o menos rápido hacia atrás'] = fuzz.trapmf(self.salida.universe, [-80,-40,-40,-30])
        self.salida['despacito hacia atrás'] = fuzz.trapmf(self.salida.universe, [-40,-30,-30,-10])
        self.salida['muy lento hacia atrás'] = fuzz.trapmf(self.salida.universe, [-30,-10,-10,10])
        self.salida['muy lento hacia delante'] = fuzz.trapmf(self.salida.universe, [-10,10,10,30])
        self.salida['despacito hacia delante'] = fuzz.trapmf(self.salida.universe, [10,30,30,50])
        self.salida['medio rápido hacia delante'] = fuzz.trapmf(self.salida.universe, [30,50,50,60])
        self.salida['rápidamente hacia delante'] = fuzz.trapmf(self.salida.universe, [50,60,60,80])
        self.salida['bastante rápido hacia delante'] = fuzz.trapmf(self.salida.universe, [60,80,80,90])
        self.salida['muy rápido hacia delante'] = fuzz.trapmf(self.salida.universe, [80,90,101,102])


    def get_membership(self,antecedent_name, value):
        # Verificar que el antecedente dado existe
        antecedent = None
        if antecedent_name == 'proporcional':
            antecedent = self.proporcional
        elif antecedent_name == 'derivativo':
            antecedent = self.derivativo
        elif antecedent_name == 'salida':
            antecedent = self.salida
        else:
            return "Antecedente no válido, pertenencia no calculada"

        # Calcular el valor de pertenencia de cada conjunto difuso para el valor dado
        memberships = {}
        for term in antecedent.terms:
            memberships[term] = fuzz.interp_membership(antecedent.universe, antecedent[term].mf, value)

        max_value = max(memberships.values())
        key = [k for k, v in memberships.items() if v == max_value][0]

        return (key, max_value)



    
        
