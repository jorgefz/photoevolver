"""
Tests functions from photoevolver.utils
"""

import sys
import pytest
import numpy as np

from photoevolver import utils

def test_kprint(capfd):
    """ Tests photoevolver.utils.kprint """
    values = {'a':1, 'b':2}
    result = "a = 1\n" + "b = 2\n\n"
    utils.kprint(**values)
    out, _ = capfd.readouterr()
    assert out == result

def test_ezip():
    """ Tests photoevolver.utils.ezip """
    x_data = 1,2,3,4,5
    y_data = 9,8,7,6,5
    for i, (x_value, y_value) in utils.ezip(x_data, y_data):
        assert x_data[i] == x_value
        assert y_data[i] == y_value

def test_ezip_r():
    """ Tests photoevolver.utils.ezip_r """
    x_data = 1,2,3,4,5
    y_data = 9,8,7,6,5
    for i, (x_value, y_value) in utils.ezip_r(x_data, y_data):
        inds = len(x_data)-i-1, len(y_data)-i-1
        assert x_data[inds[0]] == x_value
        assert y_data[inds[1]] == y_value

def test_indexable():
    """ Tests photoevolver.utils.indexable """
    assert utils.indexable([1,2,3,4])
    assert utils.indexable([])
    assert utils.indexable({})
    assert not utils.indexable(0)

    mock_class = type('Mock', (object,), {})
    assert not utils.indexable(mock_class())

    def mock_get_item(self, index):
        return self,index
    mock_class.__getitem__ = mock_get_item
    assert utils.indexable(mock_class())

def test_suppress_stdout(capfd):
    """ Tests photoevolver.utils.suppress_stdout """
    with utils.supress_stdout():
        print("hello")
        out, _ = capfd.readouterr()
        assert out == ""

def test_is_mors_star():
    """ Tests photoevolver.utils.is_mors_star """
    mock_class = type('Mock', (object,), {})
    assert not utils.is_mors_star({})
    assert not utils.is_mors_star(mock_class())

    # Inject fake module into import path
    # so that `import Mors` gives you a mock module
    ModuleType = type(sys)
    mors_module = ModuleType('Mors')
    sys.modules['Mors'] = mors_module

    # Substitute Star class to speed up check
    mors_module.Star = mock_class
    assert utils.is_mors_star(mors_module.Star())

    # Remove Mors module altogether to test import error
    del sys.modules['Mors']
    assert not utils.is_mors_star(mors_module.Star())

def test_rebin_array():
    """ Tests photoevolver.utils.rebin_array """
    # Compare array that splits perfectly
    data = np.array([1,2,3, 1,2,3])
    result = utils.rebin_array(data, 3)
    expected = [6,6] # COVERAGE MISS ??
    assert np.isclose(result, expected).all()

    def linear_combination(arr :list[float]) -> float:
        return np.sqrt(np.square(arr).sum())

    result = utils.rebin_array(data, 3, linear_combination)
    expect = np.sqrt([14, 14])
    assert np.isclose(result, expect).all()

    # Compare array that splits with leftovers
    data = [1,2,3, 1,2,3, 100]
    result = utils.rebin_array(np.array(data), 3)
    assert np.isclose(result, [6, 6, 100]).all()

    data = [1,2,3, 1,2,3, 100]
    result = utils.rebin_array(np.array(data), 3, linear_combination)
    expect = np.sqrt([14, 14, 100**2])
    assert np.isclose(result, expect).all()

    # Wrong input
    with pytest.raises(ValueError):
        utils.rebin_array(data, 0)
