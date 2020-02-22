import config as config
import numpy as np
import strlearn as sl

# Variables
n_chunks, chunk_size = config.properties_var()
scores = np.zeros(
    (
        config.replications(),
        len(config.drift_var()),
        len(config.concept_var()),
        len(config.weights_var()),
        len(config.y_flip_var()),
        len(config.methods()),
        n_chunks - 1,
        len(config.metrics()),
    )
)

headers = ["   chuj", "  drift", "concept", "weights", "l-noise"]

# Present shape
for i in range(7):
    print(
        " " * (32 + i - 6),
        "",
        "  ".join("%s\\" * len(headers))
        % tuple([headers[j][i] for j in range(len(headers))]),
    )

# print("*" * 32)
print(" " * 32, "", "  ".join("%s|" * len(headers)) % scores.shape[:5])

# Iterate streams
for a, replication in enumerate(range(config.replications())):
    for b, drift in enumerate(config.drift_var()):
        n_drifts, concept_sigmoid_spacing, incremental, recurring = drift
        for c, concept in enumerate(config.concept_var()):
            (
                n_features,
                n_informative,
                n_redundant,
                n_repeated,
                n_clusters_per_class,
            ) = concept
            for d, weights in enumerate(config.weights_var()):
                for e, y_flip in enumerate(config.y_flip_var()):
                    stream = sl.streams.StreamGenerator(
                        incremental=incremental,
                        weights=weights,
                        recurring=recurring,
                        random_state=config.random_state() + replication,
                        concept_sigmoid_spacing=concept_sigmoid_spacing,
                        y_flip=y_flip,
                        n_drifts=n_drifts,
                        n_features=n_features,
                        n_informative=n_informative,
                        n_redundant=n_redundant,
                        n_repeated=n_repeated,
                        n_clusters_per_class=n_clusters_per_class,
                        n_chunks=n_chunks,
                        chunk_size=chunk_size,
                    )

                    filename = config.hash(config.stream_to_string(stream))
                    print(
                        filename, "", "  ".join("%s|" * len(headers)) % (a, b, c, d, e)
                    )
                    results = np.load("results/%s-2.npy" % filename)
                    scores[a, b, c, d, e] = results

scores = np.mean(scores, axis=0)
np.save("scores-2", scores)

print(
    "\n%i runs, %i methods, %i metrics"
    % (
        len(config.streams(0)) * config.replications(),
        len(config.methods()),
        len(config.metrics()),
    )
)
