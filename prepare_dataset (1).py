"""
prepare_dataset.py
==================
AI-Driven E-Commerce Packaging Damage & Cost Predictor
Step 1: Real-world dataset ingestion, cleaning and mechanical calibration.

What this script does
---------------------
1. Loads a real-world e-commerce shipping CSV (e.g. the Kaggle
   "E-Commerce Shipping Data" -> ``Train.csv`` / ``shipping_data.csv``).
   A flexible alias map lets it work with most standard shipping CSVs.
2. Samples a fixed number of rows (default: 500).
3. Cleans the data: duplicates, missing values (median / mode), IQR outlier
   clipping, standardised column names and categorical encoding.
4. Derives the mechanical packaging features that raw shipping data does not
   contain:  weight_kg, distance_km, fragility_score, burst_strength_kpa,
   packaging_cost_inr and the target damage_probability.
5. Saves ``cleaned_packaging_logistics.csv`` and prints summary statistics.

Usage
-----
    python prepare_dataset.py                 # auto-detects the CSV
    python prepare_dataset.py path/to/file.csv
    python prepare_dataset.py archive.zip     # a zip holding one CSV also works

No external LLM / API is used anywhere in this project.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
SAMPLE_SIZE = 500                       # rows of real data to keep (None = all)
OUTPUT_CSV = "cleaned_packaging_logistics.csv"

# Candidate input files, searched in order when no path is given on the CLI.
CANDIDATE_FILES = [
    "shipping_data.csv",
    "Train.csv",
    "data/Train.csv",
    "archive.zip",
]

# In the Kaggle E-Commerce Shipping dataset the column
# "Reached.on.Time_Y.N" uses 1 = delivered LATE, 0 = delivered on time.
# Set to False if your file uses the opposite convention (1 = on time).
ON_TIME_COLUMN_FLAGS_LATE_WHEN_1 = True

# Engineering ranges required by the project specification.
WEIGHT_RANGE_KG = (0.1, 25.0)
DISTANCE_RANGE_KM = (50.0, 3000.0)
BURST_GRADES_KPA = [250, 400, 600, 850, 1100, 1400]   # corrugated board grades
PACKAGING_COST_RANGE_INR = (5.0, 150.0)
DAMAGE_RANGE_PCT = (1.0, 98.0)

# Fallback column mapping: canonical name -> accepted (normalised) aliases.
COLUMN_ALIASES = {
    "product_weight": [
        "weight_in_gms", "weight_gms", "weight_in_grams", "weight_g",
        "weight_kg", "weight_in_kg", "product_weight", "item_weight", "weight",
    ],
    "product_cost": [
        "cost_of_the_product", "product_cost", "product_value", "unit_price",
        "item_price", "order_value", "price", "cost",
    ],
    "shipment_mode": [
        "mode_of_shipment", "shipment_mode", "shipping_mode", "ship_mode",
        "transport_mode", "mode",
    ],
    "delay_raw": [
        "reached_on_time_y_n", "reached_on_time", "delivery_delay", "delayed",
        "delay_flag", "is_late", "late_delivery",
    ],
    "care_calls": [
        "customer_care_calls", "care_calls", "support_calls",
        "prior_customer_care_calls",
    ],
    "discount": ["discount_offered", "discount", "discount_pct"],
    "zone": ["warehouse_block", "regional_zone", "zone", "region", "warehouse"],
    "importance": [
        "product_importance", "importance", "product_priority", "priority",
    ],
    "rating": ["customer_rating", "rating"],
    "prior_purchases": ["prior_purchases", "previous_purchases"],
    "order_id": ["id", "order_id", "shipment_id"],
}

# Columns that must exist; everything else falls back to a neutral default.
REQUIRED = ["product_weight", "product_cost"]

# Typical door-to-door distance bands (km) per shipment mode.
MODE_DISTANCE_BANDS = {
    "road": (50.0, 900.0),
    "ship": (400.0, 3000.0),
    "flight": (700.0, 3000.0),
}
DEFAULT_DISTANCE_BAND = (50.0, 3000.0)

# Importance -> ordinal handling level.
IMPORTANCE_ORDER = {"low": 0, "medium": 1, "high": 2}

# Base board cost (INR per unit) for each burst-strength grade.
GRADE_BASE_COST_INR = {250: 5.0, 400: 11.0, 600: 20.0,
                       850: 33.0, 1100: 48.0, 1400: 68.0}
CUSHION_COST_PER_LEVEL_INR = 7.0        # per cushioning level (0-5)
WEIGHT_SURCHARGE_INR_PER_KG = 0.8       # extra board / tape for heavier parcels


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def standardize_name(name: str) -> str:
    """Lower-case a column name and replace non-alphanumerics with '_'."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def find_input_file(cli_path=None) -> Path:
    """Return the first existing input file (CLI argument wins)."""
    if cli_path:
        path = Path(cli_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path
    here = Path(__file__).resolve().parent
    for name in CANDIDATE_FILES:
        for base in (Path.cwd(), here):
            candidate = base / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "No dataset found. Place shipping_data.csv (or Train.csv) next to "
        "this script or pass a path: python prepare_dataset.py <file.csv>"
    )


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to canonical names using COLUMN_ALIASES."""
    rename, used = {}, set()
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias not in used:
                rename[alias] = canonical
                used.add(alias)
                break
    missing = [c for c in REQUIRED if c not in rename.values()]
    if missing:
        raise ValueError(
            f"Could not locate required column(s) {missing}. "
            f"Columns found: {list(df.columns)}"
        )
    mapped = df.rename(columns=rename)
    keep = [c for c in COLUMN_ALIASES if c in mapped.columns]
    print(f"[map]   matched columns: "
          f"{ {v: k for k, v in rename.items()} }")
    return mapped[keep].copy()


def weight_to_kg(series: pd.Series, raw_name: str) -> pd.Series:
    """Convert a raw weight column to kilograms (grams detected by name/size)."""
    name = raw_name.lower()
    if "kg" in name:
        return series.astype(float)
    if any(tag in name for tag in ("gm", "gram", "_g")):
        return series.astype(float) / 1000.0
    # Unit unknown: values in the hundreds/thousands are almost surely grams.
    return series.astype(float) / (1000.0 if series.median() > 100 else 1.0)


def clip_iqr(df: pd.DataFrame, columns, factor: float = 1.5) -> pd.DataFrame:
    """Clip each column to [Q1 - factor*IQR, Q3 + factor*IQR]."""
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lo, hi)
        print(f"[clean] IQR clip  {col:<16} bounds=({lo:9.2f}, {hi:9.2f}) "
              f"outliers treated={n_out}")
    return df


def pct_rank(series: pd.Series) -> pd.Series:
    """Percentile rank in (0, 1]."""
    return series.rank(method="average", pct=True)


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #
def load_raw(path: Path) -> pd.DataFrame:
    """Load CSV (or single-CSV zip), standardise names, drop duplicates, sample."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[load]  {path.name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    df.columns = [standardize_name(c) for c in df.columns]

    # Drop duplicate records (identifier columns are ignored for this check).
    id_like = [c for c in df.columns if c in COLUMN_ALIASES["order_id"]]
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c not in id_like])
    print(f"[clean] duplicates removed: {before - len(df)}")

    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        print(f"[load]  random sample kept: {len(df)} rows "
              f"(random_state={RANDOM_STATE})")
    return df.reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute, clip outliers and encode categorical variables."""
    # --- Neutral defaults for optional columns that are absent ---------------
    defaults = {"shipment_mode": "unknown", "zone": "unknown",
                "importance": "low", "care_calls": 0, "discount": 0,
                "rating": 3, "prior_purchases": 0, "delay_raw": 0}
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
            print(f"[map]   optional column '{col}' missing -> default {value!r}")

    categorical = ["shipment_mode", "zone", "importance"]
    numeric = ["product_weight", "product_cost", "care_calls", "discount",
               "rating", "prior_purchases", "delay_raw"]

    # Normalise categorical text, then coerce numerics.
    for col in categorical:
        df[col] = df[col].astype("string").str.strip().str.lower()
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Missing values: median (numeric) / mode (categorical) ---------------
    n_missing = int(df.drop(columns=["order_id"], errors="ignore")
                    .isna().sum().sum())
    for col in numeric:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical:
        mode = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "unknown")
    print(f"[clean] missing cells imputed: {n_missing}")

    # --- Outliers: IQR clipping on continuous / count columns ----------------
    df = clip_iqr(df, ["product_weight", "product_cost", "care_calls",
                       "discount", "prior_purchases", "rating"])

    # --- Delay flag: 1 = late, 0 = on time -----------------------------------
    delay = df["delay_raw"].round().clip(0, 1)
    df["delay_flag"] = (delay if ON_TIME_COLUMN_FLAGS_LATE_WHEN_1
                        else 1 - delay).astype(int)

    # --- Encode categoricals --------------------------------------------------
    df["importance_level"] = (df["importance"].map(IMPORTANCE_ORDER)
                              .fillna(0).astype(int))             # ordinal
    dummies = pd.get_dummies(df[["shipment_mode", "zone"]],
                             prefix=["mode", "zone"], dtype=int)  # one-hot
    return pd.concat([df, dummies], axis=1)


def engineer_features(df: pd.DataFrame, raw_weight_name: str) -> pd.DataFrame:
    """Derive the mechanical packaging features and the damage target."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(df)

    # 1) weight_kg ------------------------------------------------------------
    weight_kg = weight_to_kg(df["product_weight"], raw_weight_name)
    df["weight_kg"] = weight_kg.clip(*WEIGHT_RANGE_KG).round(3)

    # 2) distance_km: mode sets the band, warehouse zone positions inside it ---
    zones = sorted(df["zone"].unique())
    zone_pos = {z: (i / max(len(zones) - 1, 1)) for i, z in enumerate(zones)}
    zone_frac = df["zone"].map(zone_pos).to_numpy()
    lo = df["shipment_mode"].map(lambda m: MODE_DISTANCE_BANDS.get(
        m, DEFAULT_DISTANCE_BAND)[0]).to_numpy()
    hi = df["shipment_mode"].map(lambda m: MODE_DISTANCE_BANDS.get(
        m, DEFAULT_DISTANCE_BAND)[1]).to_numpy()
    position = np.clip(0.5 * zone_frac + 0.5 * rng.uniform(size=n), 0, 1)
    df["distance_km"] = (lo + (hi - lo) * position).clip(
        *DISTANCE_RANGE_KM).round(1)

    # 3) fragility_score (1-5): value-density + handling importance -----------
    value_density = df["product_cost"] / df["weight_kg"]       # value per kg
    base = 1.0 + 4.0 * (pct_rank(value_density) - 1.0 / n)     # 1 .. 5
    fragility = base + 0.5 * df["importance_level"]            # handling bump
    df["fragility_score"] = np.rint(fragility).clip(1, 5).astype(int)

    # 4) burst_strength_kpa: packaging tier -> corrugated board grade ----------
    # Shippers choose board from product needs (fragility, weight, importance)
    # but with substantial real-world variation -> noisy tier score.
    need = (0.30 * (df["fragility_score"] - 1) / 4
            + 0.05 * pct_rank(df["weight_kg"])
            + 0.05 * pct_rank(df["distance_km"])
            + 0.20 * df["importance_level"] / 2)
    tier_score = need + rng.normal(0, 0.35, size=n)
    tier = np.minimum((pct_rank(pd.Series(tier_score)) * 6).astype(int), 5)
    df["packaging_tier"] = tier.to_numpy() + 1                  # 1 .. 6
    df["burst_strength_kpa"] = [BURST_GRADES_KPA[t] for t in tier]

    # 5) packaging_cost_inr: board grade + cushioning + weight surcharge -------
    cushion_level = np.clip(df["fragility_score"].to_numpy()
                            + rng.integers(-1, 2, size=n), 0, 5)
    cost = (df["burst_strength_kpa"].map(GRADE_BASE_COST_INR).to_numpy()
            + CUSHION_COST_PER_LEVEL_INR * cushion_level
            + WEIGHT_SURCHARGE_INR_PER_KG * df["weight_kg"].to_numpy()
            + rng.normal(0, 2.0, size=n))
    df["packaging_cost_inr"] = np.clip(cost, *PACKAGING_COST_RANGE_INR).round(2)

    # 6) damage_probability (%): real delay signals + physical stress ----------
    # Historical risk from the dataset: late delivery, repeated care calls and
    # heavy discounting (rushed / clearance handling) all raise transit risk.
    delay_risk = (0.55 * df["delay_flag"]
                  + 0.10 * df["care_calls"]
                  + 0.50 * df["discount"] / max(df["discount"].max(), 1))
    # Physical stress (+) offset by protective packaging (-).
    logit = (-2.9
             + 0.45 * df["weight_kg"]
             + 0.00055 * df["distance_km"]
             + 0.60 * (df["fragility_score"] - 3)
             + delay_risk
             - 0.0017 * df["burst_strength_kpa"]
             - 0.008 * df["packaging_cost_inr"]
             + rng.normal(0, 0.25, size=n))                      # residual noise
    prob = DAMAGE_RANGE_PCT[0] + (DAMAGE_RANGE_PCT[1] - DAMAGE_RANGE_PCT[0]) \
        / (1.0 + np.exp(-logit))
    df["damage_probability"] = prob.clip(*DAMAGE_RANGE_PCT).round(2)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print shape, missing-value check and feature distributions."""
    features = ["weight_kg", "distance_km", "burst_strength_kpa",
                "fragility_score", "packaging_cost_inr", "damage_probability"]
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Shape                : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values total : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows       : {int(df.duplicated().sum())}")
    print("\nFeature statistics:")
    print(df[features].describe().round(2).T.to_string())
    print("\nFragility score counts:")
    print(df["fragility_score"].value_counts().sort_index().to_string())
    print("\nBurst strength grade counts (kPa):")
    print(df["burst_strength_kpa"].value_counts().sort_index().to_string())
    print("\nShipment mode counts:")
    print(df["shipment_mode"].value_counts().to_string())
    print("\nCorrelation of features with damage_probability:")
    print(df[features].corr()["damage_probability"].drop(
        "damage_probability").round(3).to_string())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    path = find_input_file(sys.argv[1] if len(sys.argv) > 1 else None)
    raw = load_raw(path)

    # Remember the raw weight column name (needed for unit detection).
    raw_weight_name = next((a for a in COLUMN_ALIASES["product_weight"]
                            if a in raw.columns), "weight")
    df = map_columns(raw)
    df = clean(df)
    df = engineer_features(df, raw_weight_name)

    # Final column order: identifiers/context first, model columns last.
    model_cols = ["weight_kg", "distance_km", "burst_strength_kpa",
                  "fragility_score", "packaging_cost_inr", "damage_probability"]
    context_cols = [c for c in df.columns if c not in model_cols
                    and c not in ("delay_raw", "product_weight")]
    df = df[context_cols + model_cols]

    print_summary(df)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved cleaned dataset -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
