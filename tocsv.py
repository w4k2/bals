import numpy as np
import config as c

results = np.mean(np.squeeze(np.load("scores.npy")), axis=2)

print(", ".join([i for i in c.methods().keys()]))
for i, s in enumerate(results):
    print(", ".join(["%.3f" % val for val in results[i]]))
