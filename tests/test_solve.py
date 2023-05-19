from photoevolver import dev
import matplotlib.pyplot as plt
import numpy as np

def _test_dev_monte_carlo():

    params = {'a':1.0, 'b':2.0}
    errors = {'a':0.5, 'b':[0.1, 0.5]}

    dist = dev.solve.solve_structure_errors(
        params, errors, None, None, None, samples = int(1e4)
    )
    
    fig, axs = plt.subplots(nrows = 1, ncols = len(dist), figsize=(11,8))
    for i,(k,d) in enumerate(dist.items()):
        axs[i].set_title(f"{k}: {params[k]} +/- {errors[k]}")
        axs[i].hist(d, bins = 100)

    plt.show()
    assert False