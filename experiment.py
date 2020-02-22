import numpy as np
import strlearn as sl
import config2 as config
from sklearn.base import clone
from tabulate import tabulate
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
replications = config.replications()
for replication in range(replications):
    # Select streams, methods and metrics from config
    streams = config.streams(config.random_state() + replication)
    methods = config.methods()
    metrics = config.metrics()
    clfs = list(methods.values())

    # Process
    for i, stream_n in enumerate(streams):
        print(
            "# stream %i/%i (rep %i/%i) - %s"
            % (i + 1, len(streams), replication + 1, replications, stream_n)
        )

        # Get stream
        stream = streams[stream_n]
        cloned_clfs = [clone(clf) for clf in clfs]

        # Prepare evaluator
        eval = sl.evaluators.TestThenTrain(metrics=list(metrics.values()))

        # Process
        eval.process(stream, cloned_clfs)

        # Save results
        np.save("results/%s-2" % stream_n, eval.scores)

        # Plot small table
        print(
            tabulate(
                [
                    [list(methods.keys())[j]] + ["%.3f" % value for value in row]
                    for j, row in enumerate(np.mean(eval.scores, axis=1))
                ],
                headers=["Method"] + list(metrics.keys()),
            ),
            "\n",
        )
