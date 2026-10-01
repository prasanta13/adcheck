"""Worked example on synthetic data.

Run with:

    python examples/basic_usage.py

No external dataset or extra dependency is needed; the training set
below is drawn from a 5-dimensional standard normal distribution, so a
point near the origin is typical and a point many units away in every
dimension is an obvious outlier.
"""

import numpy as np

from adcheck import ApplicabilityDomain


def main():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 5))

    ad = ApplicabilityDomain(knn_k=3).fit(X_train)

    print(f"Fitted on {ad.n_samples_fit_} reference points, {ad.n_features_in_} features")
    print(f"leverage_threshold_ (h*)      = {ad.leverage_threshold_:.4f}")
    print(f"knn_threshold_ (95th pct, k=3) = {ad.knn_threshold_:.4f}")
    print()

    X_query = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],  # the centre of the training distribution
            [1.0, -1.0, 0.5, 0.0, -0.5],  # an unremarkable point
            [8.0, 8.0, 8.0, 8.0, 8.0],  # far outside the training distribution
        ]
    )
    query_labels = ["origin", "typical point", "far outlier"]

    result = ad.evaluate(X_query)

    header = f"{'point':<16}{'leverage':>12}{'leverage_ok':>14}{'knn_dist':>12}{'knn_ok':>10}{'in_domain':>12}"
    print(header)
    print("-" * len(header))
    for label, leverage, leverage_ok, knn_dist, knn_ok, in_domain in zip(
        query_labels,
        result.leverage,
        result.leverage_ok,
        result.knn_distance,
        result.knn_ok,
        result.in_domain,
    ):
        print(
            f"{label:<16}{leverage:>12.4f}{str(leverage_ok):>14}"
            f"{knn_dist:>12.4f}{str(knn_ok):>10}{str(in_domain):>12}"
        )

    print()
    print("predict() returns just the combined verdict:")
    print(ad.predict(X_query))


if __name__ == "__main__":
    main()
