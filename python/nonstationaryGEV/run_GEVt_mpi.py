#%%
from test_sceua import run_GEVt_model
import numpy as np
import random


def initialize_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def main(x, run_dir, seed):
    x = np.array(x)

    initialize_random_seed(seed)

    bestf, best_params = run_GEVt_model(x, run_dir, parallel = True)
    return bestf, best_params

if __name__ == "__main__":
    import sys
    import json
    import numpy as np

    with open(sys.argv[1], 'r') as f:
        params = json.load(f)

    x = params['x']
    run_dir = params['run_dir']

    seed = int(sys.argv[2])

    main(x, run_dir,seed)