#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import run_exhaustive as c


def main():
    out = Path("analysis_outputs")
    test = pd.read_csv("analysis_private/evaluation.csv")
    y = test.outcome_any_use.to_numpy()
    p = np.load("analysis_private/evaluation_predictions_private.npz")
    rows = []
    d = []
    for algo in ["ridge", "ridge_fixed", "hgb"]:
        for h in [True, False]:
            a = algo + "::" + c.name((["H"] if h else []) + c.OPTIONAL)
            b = (
                algo
                + "::"
                + c.name((["H"] if h else []) + [x for x in c.OPTIONAL if x != "K"])
            )
            rows.append(
                {
                    "algorithm": algo,
                    "has_history": h,
                    "candidate_extended": a,
                    "reference_original": b,
                }
            )
            d.append(c.loss_vector(y, p[a]) - c.loss_vector(y, p[b]))
    d = np.column_stack(d)
    mean = d.mean(0)
    rng = np.random.default_rng(c.SEED + 310)
    bs = []
    for i in range(20):
        bs.append(
            rng.multinomial(len(y), np.repeat(1 / len(y), len(y)), size=100)
            @ d
            / len(y)
        )
    bs = np.vstack(bs)
    q = np.quantile(np.max(mean[None, :] - bs, axis=1), 0.95)
    for j, r in enumerate(rows):
        r.update(
            delta_log_loss=mean[j],
            paired95_low=np.quantile(bs[:, j], 0.025),
            paired95_high=np.quantile(bs[:, j], 0.975),
            joint_upper95_basic=mean[j] + q,
            contrasts_in_family=6,
            status="post-hoc matched-learner scope sensitivity; tuned settings may differ",
        )
    pd.DataFrame(rows).to_csv(
        out / "extended_vs_original_full_paired_contrasts.csv", index=False
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = pd.read_csv(out / "frozen_selected_four_family_inference.csv")
    a = a[a.domain == "extended_K"]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    for j, (_, r) in enumerate(a.iterrows()):
        ax.plot(
            [r.delta_log_loss, r.selected_family_joint_upper95_basic],
            [j, j],
            color="#0E7490",
            lw=3,
        )
        ax.scatter(r.delta_log_loss, j, color="#0E7490", marker="o", s=35)
        ax.scatter(
            r.selected_family_joint_upper95_basic,
            j,
            color="#0E7490",
            marker=">",
            s=60,
            label="Fixed predictions: joint upper 95%" if j == 0 else "",
        )
        ax.scatter(
            r.union_refit_joint_upper95_basic,
            j,
            color="#B45309",
            marker="D",
            s=38,
            label="Participant refit: joint upper 95%" if j == 0 else "",
        )
    ax.axvline(0, color="gray", lw=1)
    ax.axvline(
        0.01, color="#991B1B", ls="--", label="Illustrative noninferiority margin 0.01"
    )
    ax.set_yticks(
        range(len(a)),
        [
            x.replace("ridge::", "Ridge: ").replace("hgb::", "HGB: ")
            for x in a.candidate
        ],
    )
    ax.set_xlabel("Selected log loss minus full-information log loss")
    ax.set_title(
        "Four development-selected portfolios: exploratory joint inference\nRefits include development/evaluation overlap; model selection is held fixed",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1)
    fig.tight_layout()
    fig.savefig(
        out / "figure_frozen_selected_inference.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
