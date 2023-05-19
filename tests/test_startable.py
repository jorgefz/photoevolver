from photoevolver.dev.startable import StarTable
import numpy as np

def test_startable():

    result = StarTable.spt("Teff", 6060)
    print(result)
    assert result == "F9V"
    
    result = StarTable.interpolate("Teff", 6060)
    expect_fields = StarTable.fields()[0:-1]
    assert all( f in expect_fields for f in result.keys() )
    assert result['SpT'] == "F9V"
