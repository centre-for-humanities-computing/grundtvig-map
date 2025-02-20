from pathlib import Path

import datamapplot
import numpy as np
from gensim.models import KeyedVectors
from openTSNE import TSNE
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import scale

MODEL_PATH = "models/grundtvig.kv"
PLOT_PATH = "figures/grundtvigs_ordforraad.html"


def main():
    print("Loading embeddings")
    kv = KeyedVectors.load(MODEL_PATH)
    words = kv.index_to_key
    embeddings = kv.vectors

    print("Reducing embeddings")
    reduction = TSNE(n_components=2, metric="cosine")
    reduced_embeddings = reduction.fit(embeddings)

    print("Clustering embeddings")
    clustering = HDBSCAN(min_cluster_size=10)
    labels = clustering.fit_predict(reduced_embeddings)

    print("Assigning cluster labels")
    vocab = np.array(words)
    classes = np.sort(np.unique(labels))
    top = []
    for label in classes:
        if label == -1:
            top.append("Outlier")
        topic_vector = np.mean(embeddings[labels == label], axis=0)
        dist = np.ravel(cosine_distances([topic_vector], embeddings))
        top_words = np.argsort(dist)[:4]
        desc = "\n".join(vocab[top_words])
        top.append(desc)
    top = np.array(top)

    print("Producing plot")
    plot = datamapplot.create_interactive_plot(
        scale(reduced_embeddings) * 5,
        # Adding one so that outliers come first
        top[labels + 1],
        words,
        hover_text=words,
        initial_zoom_fraction=0.999,
        font_family="Marcellus SC",
        noise_label="Outlier",
        title="Grundtvigs Ordforråd",
        logo_width=128,
        noise_color="#959a9b",
        enable_search=True,
        darkmode=True,
    )
    print("Saving plot")
    plot_path = Path(PLOT_PATH)
    plot_path.parent.mkdir(exist_ok=True)
    plot.save(plot_path)
    print("DONE")


if __name__ == "__main__":
    main()
