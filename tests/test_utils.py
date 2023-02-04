import numpy as np
import sys
import pytest
import importlib

from photoevolver import utils

def test_kprint(capfd):
    values = dict(a=1, b=2)
    result = "a = 1\n" + "b = 2\n\n"
    utils.kprint(**values)
    out, err = capfd.readouterr()
    assert out == result

def test_ezip():
    xd = 1,2,3,4,5
    yd = 9,8,7,6,5
    for i,(x,y) in utils.ezip(xd, yd):
        assert xd[i] == x and yd[i] == y

def test_ezip_r():
    xd = 1,2,3,4,5
    yd = 9,8,7,6,5
    for i,(x,y) in utils.ezip_r(xd, yd):
        assert xd[len(xd)-i-1] == x and yd[len(yd)-i-1] == y

def test_indexable():
    assert utils.indexable(list([1,2,3,4]))
    assert utils.indexable(list())
    assert utils.indexable(dict())
    assert not utils.indexable(int())
    class Mock: pass
    assert not utils.indexable(Mock())
    Mock.__getitem__ = lambda self,i: ...
    assert utils.indexable(Mock())

def test_is_mors_star():

    class MockType: pass
    assert not utils.is_mors_star(dict())
    assert not utils.is_mors_star(MockType())

    # Inject fake module into import path
    # so that `import Mors` gives you a mock module
    ModuleType = type(sys)
    Mors = ModuleType('Mors')
    sys.modules['Mors'] = Mors

    # Substitute Star class to speed up check
    Mors.Star = MockType
    assert utils.is_mors_star(Mors.Star())

def test_rebin_array():
    # Compare array that splits perfectly
    data = np.array([1,2,3, 1,2,3])
    result = utils.rebin_array(data, 3)
    expected = [6,6] # COVERAGE MISS ??
    assert np.isclose(result, expected).all()

    linear_combination = lambda arr: np.sqrt(np.square(arr).sum())
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
    

    
