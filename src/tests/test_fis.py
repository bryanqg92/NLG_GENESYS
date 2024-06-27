import pytest
import numpy as np
from data_utils.fis.fis import FIS, InvalidAntecedentError

# Creamos una instancia de la clase FIS para las pruebas
fis = FIS()
def test_get_membership_boundary_values():
    # Caso de prueba para valores límite de los antecedentes
    proporcional_values = [0.0, 1.0]
    derivativo_values = [-1.0, 1.0]
    salida_values = [0.0, 1.0]
    
    for value in proporcional_values:
        term, membership = fis.get_membership('proporcional', value)
        assert term in fis._proporcional.terms
        assert 0.0 <= membership <= 1.0
    
    for value in derivativo_values:
        term, membership = fis.get_membership('derivativo', value)
        assert term in fis._derivativo.terms
        assert 0.0 <= membership <= 1.0
    
    for value in salida_values:
        term, membership = fis.get_membership('salida', value)
        assert term in fis._salida.terms
        assert 0.0 <= membership <= 1.0

def test_get_membership_negative_values():
    # Caso de prueba para valores negativos de los antecedentes
    proporcional_value = -0.5
    derivativo_value = -0.7
    salida_value = -0.3
    
    term, membership = fis.get_membership('proporcional', proporcional_value)
    assert term in fis._proporcional.terms
    assert 0.0 <= membership <= 1.0
    
    term, membership = fis.get_membership('derivativo', derivativo_value)
    assert term in fis._derivativo.terms
    assert 0.0 <= membership <= 1.0
    
    term, membership = fis.get_membership('salida', salida_value)
    assert term in fis._salida.terms
    assert 0.0 <= membership <= 1.0

def test_get_membership_proporcional():
    # Caso de prueba para el antecedente 'proporcional'
    value = 0.2
    term, membership = fis.get_membership('proporcional', value)
    assert term in fis._proporcional.terms
    assert 0.0 <= membership <= 1.0

def test_get_membership_derivativo():
    # Caso de prueba para el antecedente 'derivativo'
    value = -0.1
    term, membership = fis.get_membership('derivativo', value)
    assert term in fis._derivativo.terms
    assert 0.0 <= membership <= 1.0

def test_get_membership_salida():
    # Caso de prueba para el antecedente 'salida'
    value = 0.5
    term, membership = fis.get_membership('salida', value)
    assert term in fis._salida.terms
    assert 0.0 <= membership <= 1.0

def test_get_membership_invalid_antecedent():
    # Caso de prueba para manejar excepción por nombre de antecedente inválido
    with pytest.raises(InvalidAntecedentError):
        fis.get_membership('invalid_antecedent', 0.5)

def test_get_membership_all_memberships():
    # Caso de prueba para obtener todos los valores de pertenencia de un antecedente
    value = 0.3
    memberships = fis.get_membership('proporcional', value, method='all')
    assert isinstance(memberships, dict)
    assert len(memberships) == len(fis._proporcional.terms)
    for term, membership in memberships.items():
        assert term in fis._proporcional.terms
        assert 0.0 <= membership <= 1.0

def test_get_membership_max_membership():
    # Caso de prueba para obtener el máximo valor de pertenencia de un antecedente
    value = -0.5
    term, membership = fis.get_membership('derivativo', value, method='max')
    assert term in fis._derivativo.terms
    assert 0.0 <= membership <= 1.0

def test_get_membership_zero_value():
    # Caso de prueba para valor de pertenencia cero
    value = 2.0
    term, membership = fis.get_membership('proporcional', value)
    assert term == 'muy lejos'
    assert membership == 0.0

