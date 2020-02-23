import numpy as np
import config as c

results = np.mean(np.squeeze(np.load("scores.npy")), axis=2)

print(", ".join(["scenario"] + [i for i in c.methods().keys()]))
for i, s in enumerate(results):
    print(", ".join(["%i" % i]+["%.3f" % val for val in results[i]]))
