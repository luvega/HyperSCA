"""Figure generation for target discovery."""
from __future__ import annotations


def generate_figure_pack(writer, inputs) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    ranking = inputs.get("target_ranking")
    if ranking is not None and not ranking.empty and {"gene", "final_score"}.issubset(ranking.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        top = ranking.head(20).iloc[::-1]
        ax.barh(top["gene"].astype(str), top["final_score"].astype(float), color="#4C78A8")
        ax.set_xlabel("Final score")
        ax.set_ylabel("Gene")
        paths.append(writer.write_figure("target_ranking_top20.png", fig, section="figures", metadata={"chart": "target_ranking_top20"}))
        plt.close(fig)

    geometry = inputs.get("geometry_results", {})
    for mode, result in geometry.items():
        embedding = result.get("embedding")
        labels = inputs.get("node_labels", [])
        if embedding is None or len(embedding) == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(embedding[:, 0], embedding[:, 1], s=40, color="#F58518")
        for idx, label in enumerate(labels):
            ax.text(embedding[idx, 0], embedding[idx, 1], str(label), fontsize=7)
        ax.set_title(f"{mode} geometry")
        paths.append(writer.write_figure(f"geometry_{mode}.png", fig, section="figures", metadata={"chart": f"geometry_{mode}"}))
        plt.close(fig)
    return paths
