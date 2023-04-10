import gzip
import json
import numpy as np
import sys

class Logger:
    def __init__(self, filename):
        if filename:
            self.f = gzip.open(filename, "wt")
            self.f.write("runarg=" + json.dumps(sys.argv))
        else:
            self.f = None


    def log_generation(self, gen, good, limit_acc, parent_pop):
        if not self.f:
            return

        pp = []

        def cvt(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        for p in parent_pop:
            params = {k:cvt(v) for k,v in p.parameters.items()}
            pp.append({"chrom": str(p), **params})

        pp = json.dumps(pp)

        self.f.write(f"#gen={gen}, good={good}, limit_acc={limit_acc}, parent_pop = {pp}\n")