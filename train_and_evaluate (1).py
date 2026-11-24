"""
train_and_evaluate.py
=====================
AI-Driven E-Commerce Packaging Damage & Cost Predictor
Step 2: Model training, comparison, evaluation plots and a rule-based advisor.

Run ``prepare_dataset.py`` first to create ``cleaned_packaging_logistics.csv``.

Outputs
-------
    model.pkl                          Random Forest (joblib)
    plots/actual_vs_predicted.png
    plots/feature_importance.png
    plots/burst_strength_vs_damage.png
    plots/correlation_heatmap.png

The advisor ``get_packaging_recommendation`` is purely deterministic: it uses
no external LLM or API.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend: save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_FILE = "cleaned_packaging_logistics.csv"
MODEL_FILE = "model.pkl"
PLOTS_DIR = Path("plots")
RANDOM_STATE = 42
DPI = 300

FEATURES = ["weight_kg", "distance_km", "burst_strength_kpa",
            "fragility_score", "packaging_cost_inr"]
TARGET = "damage_probability"

# Damage threshold (%) above which the advisor recommends changes.
DAMAGE_THRESHOLD_PCT = 10.0

# Corrugated board ladder: (ply label, wall type, min burst strength kPa).
PLY_LADDER = [
    ("3-ply", "single-wall", 0),
    ("5-ply", "double-wall", 600),
    ("7-ply", "triple-wall", 1100),
]
# Approximate extra unit cost (INR) when moving up one ply level.
PLY_UPGRADE_COST_INR = {"3-ply": 18.0, "5-ply": 28.0}
# Representative burst strength (kPa) of each target ply grade.
PLY_TARGET_BURST_KPA = {"5-ply": 850, "7-ply": 1400}


# --------------------------------------------------------------------------- #
# Rule-based packaging advisor (no external APIs)
# --------------------------------------------------------------------------- #
def current_ply(burst_strength: float) -> str:
    """Map a burst strength (kPa) to the 3-, 5- or 7-ply grade it represents."""
    label = PLY_LADDER[0][0]
    for ply, _, min_kpa in PLY_LADDER:
        if burst_strength >= min_kpa:
            label = ply
    return label


def cushioning_advice(fragility: float, weight: float) -> str:
    """Pick internal cushioning from the fragility score (1-5) and weight."""
    if fragility >= 5:
        advice = ("custom die-cut foam (EPE/PU) inserts with corner guards, "
                  "plus double-boxing with 50 mm clearance on all sides")
    elif fragility == 4:
        advice = ("molded pulp or EPE foam corner/edge protectors with "
                  "multi-layer bubble wrap (25 mm clearance)")
    elif fragility == 3:
        advice = "double-layer bubble wrap with kraft-paper void fill"
    elif fragility == 2:
        advice = "single-layer bubble wrap or air pillows to remove void space"
    else:
        advice = "crumpled kraft paper void fill (item is robust)"
    if weight >= 10:
        advice += "; add a rigid corrugated pad/divider to spread the load"
    return advice


def get_packaging_recommendation(weight, distance, burst_strength, fragility,
                                 cost, predicted_damage):
    """Return deterministic, actionable packaging advice.

    Parameters
    ----------
    weight : float            Parcel weight in kg.
    distance : float          Transit distance in km.
    burst_strength : float    Current board burst strength in kPa.
    fragility : int           Fragility score, 1 (robust) to 5 (very fragile).
    cost : float              Current unit packaging cost in INR.
    predicted_damage : float  Model-predicted damage probability in percent.

    Cost-to-Protection Index = cost / (1 + predicted_damage / 100).
    It discounts packaging spend by the residual damage risk. It rises when
    you spend more, so it is judged together with the risk reduction bought.
    """
    index = cost / (1.0 + predicted_damage / 100.0)
    ply = current_ply(burst_strength)

    lines = [
        "PACKAGING RECOMMENDATION",
        "-" * 52,
        f"Inputs   : {weight:.1f} kg | {distance:,.0f} km | "
        f"{burst_strength:.0f} kPa ({ply}) | fragility {fragility} | "
        f"INR {cost:.2f}",
        f"Predicted damage probability : {predicted_damage:.1f}%",
        f"Cost-to-Protection Index     : {index:.2f}",
    ]

    # ---- Low risk: no change needed -----------------------------------------
    if predicted_damage <= DAMAGE_THRESHOLD_PCT:
        lines.append(f"Status   : LOW RISK (<= {DAMAGE_THRESHOLD_PCT:.0f}%) - "
                     "current packaging is adequate.")
        if predicted_damage < 5 and cost > 60 and ply != "3-ply":
            lines.append("Savings  : risk is very low; a lighter board or "
                         "thinner cushioning could cut cost without "
                         "meaningful protection loss.")
        return "\n".join(lines)

    lines.append(f"Status   : HIGH RISK (> {DAMAGE_THRESHOLD_PCT:.0f}%) - "
                 "packaging upgrade recommended.")

    # ---- Corrugated ply upgrade ---------------------------------------------
    if ply == "3-ply":
        # Fragile, heavy or long-haul parcels skip straight to 7-ply.
        jump_to_seven = (predicted_damage >= 40 or fragility >= 5
                         or weight >= 15)
        target = "7-ply" if jump_to_seven else "5-ply"
        extra = PLY_UPGRADE_COST_INR["3-ply"] + (
            PLY_UPGRADE_COST_INR["5-ply"] if target == "7-ply" else 0.0)
    elif ply == "5-ply":
        target, extra = "7-ply", PLY_UPGRADE_COST_INR["5-ply"]
    else:
        target, extra = None, 0.0

    if target:
        wall = dict((p, w) for p, w, _ in PLY_LADDER)[target]
        lines.append(f"Board    : upgrade {ply} -> {target} {wall} corrugated "
                     f"(~{PLY_TARGET_BURST_KPA[target]} kPa burst strength).")
    else:
        lines.append("Board    : already 7-ply triple-wall; a thicker board "
                     "will not help - improve internal cushioning and "
                     "handling instead.")

    # ---- Cushioning ------------------------------------------------------------
    lines.append(f"Cushion  : {cushioning_advice(fragility, weight)}.")

    # ---- Transit-specific handling -------------------------------------------
    if distance >= 1500:
        lines.append("Transit  : long haul - add edge protectors, "
                     "'FRAGILE / THIS SIDE UP' labels and moisture-barrier "
                     "wrap.")
    elif fragility >= 4:
        lines.append("Transit  : add 'FRAGILE' labels and request careful "
                     "handling.")

    # ---- Cost trade-off ---------------------------------------------------------
    if target:
        new_cost = cost + extra
        # Expected-damage cost: assume the upgrade roughly halves the risk of
        # each lost parcel for a 5-ply step, and cuts it by ~60% for 7-ply.
        reduction = 0.60 if target == "7-ply" else 0.40
        new_damage = predicted_damage * (1.0 - reduction)
        new_index = new_cost / (1.0 + new_damage / 100.0)
        lines.append(
            f"Tradeoff : packaging cost rises ~INR {extra:.0f} "
            f"({extra / cost * 100:.0f}% more) to ~INR {new_cost:.2f}, with "
            f"damage risk expected to fall from {predicted_damage:.1f}% to "
            f"roughly {new_damage:.1f}% (estimate).")
        # Judge the upgrade by percentage points of risk removed per rupee.
        pts_per_inr = (predicted_damage - new_damage) / extra
        verdict = ("worth it" if predicted_damage >= 25 or pts_per_inr >= 0.4
                   else "marginal")
        lines.append(f"Verdict  : upgrade is {verdict} "
                     f"(~{pts_per_inr:.2f} pts of damage risk removed per INR; "
                     f"index {index:.2f} -> ~{new_index:.2f}).")
    else:
        lines.append("Tradeoff : budget extra cushioning (about INR 10-25) "
                     "rather than more board.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #
def build_models() -> dict:
    """Return the three regression models to compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            random_state=RANDOM_STATE),
    }


def evaluate(model, x_test, y_test) -> dict:
    """Compute R2, MSE, RMSE and MAE on the test set."""
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    return {"R2": r2_score(y_test, pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_test, pred)}


def print_comparison(results: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 66)
    print("MODEL COMPARISON (20% hold-out test set)")
    print("=" * 66)
    print(f"{'Model':<20}{'R2':>10}{'MSE':>12}{'RMSE':>12}{'MAE':>12}")
    print("-" * 66)
    for name, row in results.iterrows():
        print(f"{name:<20}{row['R2']:>10.4f}{row['MSE']:>12.3f}"
              f"{row['RMSE']:>12.3f}{row['MAE']:>12.3f}")
    print("=" * 66)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_actual_vs_predicted(y_test, y_pred) -> None:
    """Scatter of actual vs predicted with a y = x reference line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.65, edgecolor="k", linewidth=0.3,
               color="#1f77b4", label="Test samples")
    lims = [min(y_test.min(), y_pred.min()) - 2,
            max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set(xlim=lims, ylim=lims, xlabel="Actual damage probability (%)",
           ylabel="Predicted damage probability (%)",
           title="Random Forest: Actual vs Predicted Damage Probability")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=DPI)
    plt.close(fig)


def plot_feature_importance(model) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color="#2ca02c", edgecolor="k")
    for i, v in enumerate(imp.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set(xlabel="Importance", title="Random Forest Feature Importance")
    ax.set_xlim(0, imp.max() * 1.15)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=DPI)
    plt.close(fig)


def plot_burst_vs_damage(df: pd.DataFrame) -> None:
    """Boxplot of damage probability across burst-strength tiers."""
    fig, ax = plt.subplots(figsize=(9, 6))
    order = sorted(df["burst_strength_kpa"].unique())
    sns.boxplot(data=df, x="burst_strength_kpa", y=TARGET, order=order,
                hue="burst_strength_kpa", hue_order=order,
                palette="YlGnBu", legend=False, ax=ax)
    ax.set(xlabel="Burst strength (kPa)",
           ylabel="Damage probability (%)",
           title="Damage Probability by Corrugated Burst-Strength Tier")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "burst_strength_vs_damage.png", dpi=DPI)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Seaborn heatmap of the correlation between all features and target."""
    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap: Features and Damage Probability")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run `python prepare_dataset.py` first.")
    df = pd.read_csv(DATA_FILE)
    x, y = df[FEATURES], df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE)
    print(f"Loaded {len(df)} rows | train={len(x_train)} test={len(x_test)}")

    # ---- Train and evaluate all models --------------------------------------
    models, scores = build_models(), {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        scores[name] = evaluate(model, x_test, y_test)
    results = pd.DataFrame(scores).T
    print_comparison(results)

    best_by_r2 = results["R2"].idxmax()
    print(f"Highest test R2: {best_by_r2}")

    # ---- Persist the primary model (Random Forest) ----------------------------
    rf = models["Random Forest"]
    joblib.dump(rf, MODEL_FILE)
    print(f"Saved primary model (Random Forest) -> {MODEL_FILE}")

    # ---- Plots -----------------------------------------------------------------
    PLOTS_DIR.mkdir(exist_ok=True)
    plot_actual_vs_predicted(y_test, rf.predict(x_test))
    plot_feature_importance(rf)
    plot_burst_vs_damage(df)
    plot_correlation_heatmap(df)
    print(f"Saved 4 plots (300 dpi) -> {PLOTS_DIR}/")

    # ---- Demo: advisor on sample scenarios -------------------------------------
    print("\n" + "=" * 66)
    print("RULE-BASED ADVISOR DEMO")
    print("=" * 66)
    scenarios = [
        # weight, distance, burst, fragility, cost
        (12.0, 2200, 250, 5, 18.0),   # heavy, fragile, weak board
        (4.0, 900, 600, 3, 45.0),     # mid-range parcel
        (1.5, 300, 1400, 1, 95.0),    # robust item, over-packaged
    ]
    for w, d, b, f, c in scenarios:
        row = pd.DataFrame([[w, d, b, f, c]], columns=FEATURES)
        pred = float(rf.predict(row)[0])
        print(get_packaging_recommendation(w, d, b, f, c, pred))
        print()


if __name__ == "__main__":
    main()
