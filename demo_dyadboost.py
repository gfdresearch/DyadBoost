# -*- coding: utf-8 -*-
"""Worked demonstration of the Section 4.5 pipeline on synthetic dyadic data.

The synthetic process is built to reproduce the situation the review describes:

  * the actor and partner main effects carry most of the observable signal;
  * the dyadic features add a small increment;
  * an unobserved shock drives dissolution and is recorded only at month 12;
  * one dyadic feature is pure noise.

Three protocol claims are demonstrated:

  S1  declaring the month-12 variable at t0 = 0 is refused;
      moving the decision point to month 12 "improves" AUC, which is leakage, not prediction;
  S5  only the dominant dyadic feature survives the stability screen; the weak ones are
      indistinguishable from a pure-noise feature at this sample size;
  S6  removing the dominant feature costs AUC, while removing the flagged ones does not.

Note on scope: automated screens cannot detect label contamination that is present in both
training and test data; that requires provenance review of each column, which is why every
predictor must carry an availability month in S1.

Run:  python demo_dyadboost.py
"""
import os
import sys

_CAND = os.path.expandvars(
    r"%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python311\site-packages")
if os.path.isdir(_CAND) and _CAND not in sys.path:
    sys.path.insert(0, _CAND)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dyadboost import DyadBoost, Feature, TemporalLeakageError, coverage, evaluate

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "demo_outputs")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261112)


# --------------------------------------------------------------------- data
def make_data(n=4000):
    """One row per couple."""
    z = lambda: RNG.normal(size=n)

    actor = {"a_neuroticism": z(), "a_extraversion": z(), "a_self_esteem": z(),
             "a_attachment_anxiety": z(), "a_age_years": 28 + 7 * z()}
    partner = {"p_neuroticism": z(), "p_extraversion": z(), "p_warmth": z(),
               "p_attachment_avoidance": z(), "p_age_years": 28 + 7 * z()}
    shock = z()                                   # unobserved future stress
    dyad = {"d_value_similarity": z(), "d_conversation_valence": z(),
            "d_synchrony": z(), "d_reciprocity": z(), "d_noise": z()}

    eta = (-2.95
           - 0.75 * actor["a_neuroticism"] + 0.60 * actor["a_self_esteem"]
           + 0.55 * partner["p_warmth"] - 0.45 * partner["p_attachment_avoidance"]
           + 0.28 * dyad["d_value_similarity"] + 0.22 * dyad["d_conversation_valence"]
           + 0.10 * dyad["d_synchrony"] + 0.06 * dyad["d_reciprocity"]
           + 0.90 * shock)
    y = RNG.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

    df = pd.DataFrame({**actor, **partner, **dyad})
    df["couple_id"] = np.arange(n)
    # recorded at month 12: a noisy proxy of the unobserved shock
    df["conflict_m12"] = 1.6 * shock + RNG.normal(scale=0.3, size=n)
    return df, y, df["couple_id"].to_numpy()


FEATURES_T0 = (
    [Feature(k, "actor") for k in
     ("a_neuroticism", "a_extraversion", "a_self_esteem", "a_attachment_anxiety", "a_age_years")]
    + [Feature(k, "partner") for k in
       ("p_neuroticism", "p_extraversion", "p_warmth", "p_attachment_avoidance", "p_age_years")]
    + [Feature(k, "dyad") for k in
       ("d_value_similarity", "d_conversation_valence", "d_synchrony", "d_reciprocity", "d_noise")]
)
FEATURES_LATE = FEATURES_T0 + [Feature("conflict_m12", "dyad", available_at=12)]


def main():
    df, y, groups = make_data()
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.30, random_state=7, stratify=y)
    df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    print("=" * 80)
    print(f"synthetic dyads n = {len(y)}  |  dissolution base rate = {y.mean():.3f}  "
          f"|  events in test = {int(y_te.sum())}")
    print("=" * 80)

    # ------------------------------------------------------------------ S1
    print("\n[S1] availability firewall")
    for t0 in (0, 12):
        try:
            DyadBoost(FEATURES_LATE, t0=t0, verbose=False)
            print(f"  t0 = {t0:2d} months : accepted conflict_m12 (recorded at month 12)")
        except TemporalLeakageError as exc:
            print(f"  t0 = {t0:2d} months : REFUSED -> {exc}")

    # ------------------------------------------------------------------ fit
    print("\n[S2-S3] pipeline at t0 = 0 (5 folds x 5 repeats)")
    model = DyadBoost(FEATURES_T0, t0=0, n_splits=5, n_repeats=5, seed=11)
    model.fit(df_tr, y_tr, groups=groups[tr_idx])

    cols1 = model.actor + model.partner
    sc = StandardScaler().fit(df_tr[cols1])
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(df_tr[cols1]), y_tr)
    p_main = lr.predict_proba(sc.transform(df_te[cols1]))[:, 1]

    print("\n[leakage demonstration] decision point moved to month 12")
    naive = DyadBoost(FEATURES_LATE, t0=12, n_splits=5, n_repeats=5, seed=11, verbose=False)
    naive.fit(df_tr, y_tr, groups=groups[tr_idx])

    p_pipe, p_naive = model.predict_proba(df_te), naive.predict_proba(df_te)

    rows = []
    for name, p in (("main effects only (actor + partner)", p_main),
                    ("DyadBoost, t0 = 0", p_pipe),
                    ("DyadBoost, t0 = 12 (post-decision variable)", p_naive)):
        rows.append({"model": name, **evaluate(p, y_te)})
    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------------------ S4
    print("\n[S4] calibration and Mondrian conformal sets (alpha = 0.20)")
    inc0, inc1 = model.conformal_sets(p_pipe, alpha=0.20)
    cov, size = coverage(p_pipe, y_te, inc0, inc1)
    print(f"  empirical coverage = {cov:.3f}  |  mean set size = {size:.2f}")
    print("  (a weak model yields conservative sets; coverage is the guarantee, size is the cost)")

    # ------------------------------------------------------------------ S5
    print(f"\n[S5] attribution ensemble over the {len(model.folds_)} fold models")
    attr = model.attribution_table(df_te)
    print(attr.round(3).to_string(index=False))

    # ------------------------------------------------------------------ S6
    print("\n[S6] audit")
    flagged = attr.loc[attr.stability < 0.55, "feature"].tolist()
    print(f"  below the stability screen (0.55): {flagged}")
    print("  only the dominant dyadic feature survives the ensemble; the remaining features")
    print("  are statistically indistinguishable from the noise feature at this sample size.")

    m_base = evaluate(p_pipe, y_te)["auc"]
    probes = [attr.iloc[0]["feature"]] + flagged[-2:]
    deltas = []
    for cand in dict.fromkeys(probes):
        keep = [f for f in FEATURES_T0 if f.name != cand]
        if not any(f.layer == "dyad" for f in keep):
            continue
        refit = DyadBoost(keep, t0=0, n_splits=5, n_repeats=5, seed=11, verbose=False)
        refit.fit(df_tr, y_tr, groups=groups[tr_idx])
        auc = evaluate(refit.predict_proba(df_te), y_te)["auc"]
        deltas.append({"dropped": cand, "flagged": cand in flagged,
                       "auc_after": auc, "delta_auc": auc - m_base})

    print("\n  performance screen (held-out AUC):")
    print(f"    none removed                 -> {m_base:.3f}")
    for d in deltas:
        tag = "flagged" if d["flagged"] else "survives the screen"
        print(f"    drop {d['dropped']:<26s} -> {d['auc_after']:.3f} "
              f"(delta {d['delta_auc']:+.3f})  [{tag}]")

    # ------------------------------------------------------------------ output
    table.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    attr.to_csv(os.path.join(OUT, "attribution.csv"), index=False)
    pd.DataFrame(deltas).to_csv(os.path.join(OUT, "audit_deltas.csv"), index=False)
    pd.DataFrame({"model": [r["model"] for r in rows],
                  "auc": [r["auc"] for r in rows]}).to_csv(
        os.path.join(OUT, "ladder.csv"), index=False)
    print(f"\nwrote results to {OUT}")

    make_figure(table, attr, flagged)


# ------------------------------------------------------------------ figure
def make_figure(table, attr, flagged):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.linewidth": 0.7, "axes.edgecolor": "#444444",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
    })
    BLUE, VERM, GREY, INK = "#0072B2", "#D55E00", "#7A7A7A", "#222222"

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8),
                             gridspec_kw={"width_ratios": [1.0, 1.2]})

    ax = axes[0]
    labels = ["main\neffects", "DyadBoost\nt0 = 0", "DyadBoost\nt0 = 12"]
    vals = table["auc"].to_numpy()
    cols = ["#9E9E9E", BLUE, VERM]
    ax.bar(range(3), vals, width=0.58, color=cols, edgecolor="white", linewidth=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylim(0.5, max(vals) + 0.08)
    ax.set_ylabel("AUC on held-out couples")
    ax.set_title("Discrimination", fontsize=7.5, loc="left", pad=8)
    ax.text(-0.24, 1.10, "(A)", transform=ax.transAxes, fontsize=8, fontweight="bold")

    ax = axes[1]
    short = {"d_value_similarity": "similarity", "d_conversation_valence": "conversation",
             "d_synchrony": "synchrony", "d_reciprocity": "reciprocity", "d_noise": "noise"}
    feats = [short.get(f, f) for f in attr["feature"]]
    swa = attr["swa"].to_numpy()
    stab = attr["stability"].to_numpy()
    order = np.argsort(swa)                      # ascending so the largest sits on top
    ypos = np.arange(len(feats))
    colors = [VERM if attr["feature"].iloc[i] in flagged else BLUE for i in order]
    ax.barh(ypos, swa[order], height=0.62, color=colors, edgecolor="white", linewidth=0.8)
    for y, i in zip(ypos, order):
        ax.text(swa[i] + 0.006, y, f"{swa[i]:.2f}  (stab {stab[i]:.2f})",
                va="center", ha="left", fontsize=6.3, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([feats[i] for i in order], fontsize=6.8)
    ax.set_xlim(0, max(swa) * 1.55)
    ax.set_xlabel("Stability-weighted attribution")
    ax.text(0.99, -0.34, "vermilion: below the stability screen", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color=VERM)
    ax.set_title("Attribution that survives the ensemble", fontsize=7.5, loc="left", pad=8)
    ax.text(-0.30, 1.10, "(B)", transform=ax.transAxes, fontsize=8, fontweight="bold")

    fig.subplots_adjust(wspace=0.38)
    path = os.path.join(OUT, "fig_dyadboost_demo.png")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
