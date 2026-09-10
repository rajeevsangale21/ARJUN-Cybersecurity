from __future__ import annotations

"""
ARJUN PHASE A.4 - EXPLAINABILITY

Produces a production-ready explanation report from the artifacts already
created by ARJUN.

Explanation layers:
1. XGBoost feature importance for the current attack classifier.
2. Future-state feature change from current -> t+5.
3. Temporal influence proxy: how consistently each feature changes across
   the 5 predicted future states.
4. What-if analysis: perturb important current features and rerun the
   World Model + future classifier to measure sensitivity.

Important:
- This is a model explanation/sensitivity layer, not a claim of causal
  attribution.
- It does not retrain any model.
- It does not modify the production checkpoint.
- No SHAP dependency is required for this phase.
"""

from pathlib import Path
import json
import joblib
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.state_encoder import NetworkStateEncoder
from forecasting.hybrid_k_step_forecaster import (
    ForecastStateScaler,
    forecast_k_steps,
    find_value,
    get_model_config,
    normalize_graph_sequence,
)
from world_model.hybrid_world_model import HybridWorldModel


PRODUCTION_JSON = (
    ROOT / "data" / "processed" / "production_forecast.json"
)
PRODUCTION_NPZ = (
    ROOT / "data" / "processed" / "production_forecast.npz"
)
ZEEK_GRAPHS = (
    ROOT / "data" / "processed" / "zeek_network_graphs.pkl"
)
WORLD_MODEL_FILE = (
    ROOT / "saved_models" / "hybrid_world_model.pt"
)
IMPROVED_XGB_FILE = (
    ROOT / "saved_models" / "xgboost_attack_classifier_improved.pkl"
)
FAMILY_XGB_FILE = (
    ROOT / "saved_models" / "xgboost_attack_family_classifier.pkl"
)

OUTPUT_JSON = (
    ROOT / "data" / "processed" / "explainability_report.json"
)

OUTPUT_NPZ = (
    ROOT / "data" / "processed" / "explainability_report.npz"
)

TOP_K = 10
WHAT_IF_FEATURES = 5
WHAT_IF_PERTURBATIONS = (-0.25, 0.25)


def finite(x):
    return np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )


def load_world_model():
    require(WORLD_MODEL_FILE)

    checkpoint = torch.load(
        WORLD_MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    state_dimension = int(
        find_value(
            checkpoint,
            ["state_dimension", "input_dimension"],
        )
        or 33
    )

    graph_dimension = int(
        find_value(
            checkpoint,
            ["graph_input_dimension", "graph_dimension"],
        )
        or 6
    )

    config = get_model_config(
        checkpoint,
        state_dimension,
        graph_dimension,
    )

    model = HybridWorldModel(
        state_dimension=state_dimension,
        graph_input_dimension=graph_dimension,
        hidden_dimension=config["hidden_dimension"],
        lstm_layers=config["lstm_layers"],
        residual_scale=config["residual_scale"],
    )

    state_dict = find_value(
        checkpoint,
        ["model_state_dict", "state_dict"],
    )

    if state_dict is None:
        raise RuntimeError(
            "World Model checkpoint has no model_state_dict."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )
    model.eval()

    scaler_data = find_value(
        checkpoint,
        ["state_scaler", "scaler", "state_normalizer"],
    )

    center = None
    scale = None

    if isinstance(scaler_data, dict):
        center = find_value(
            scaler_data,
            ["center", "median", "location", "mean"],
        )
        scale = find_value(
            scaler_data,
            ["scale", "iqr", "std", "spread"],
        )

    if center is None:
        center = find_value(
            checkpoint,
            ["state_center", "state_median", "state_mean"],
        )

    if scale is None:
        scale = find_value(
            checkpoint,
            ["state_scale", "state_iqr", "state_std"],
        )

    if center is None or scale is None:
        raise RuntimeError(
            "Could not recover World Model scaler."
        )

    scaler = ForecastStateScaler(
        center=center,
        scale=scale,
    )

    return model, scaler, config


def load_xgb():
    require(IMPROVED_XGB_FILE)

    artifact = joblib.load(
        IMPROVED_XGB_FILE
    )

    if not isinstance(artifact, dict):
        raise RuntimeError(
            "Improved XGBoost artifact is not a dictionary."
        )

    model = artifact["model"]
    normalizer = artifact["normalizer"]
    threshold = float(
        artifact.get("threshold", 0.50)
    )

    return model, normalizer, threshold


def predict_attack_probability(
    model,
    normalizer,
    states,
):
    states = finite(states)

    transformed = normalizer.transform(
        states
    )

    probabilities = np.asarray(
        model.predict_proba(transformed),
        dtype=np.float32,
    )

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise RuntimeError(
            f"Unexpected XGBoost probability shape: "
            f"{probabilities.shape}"
        )

    return np.clip(
        probabilities[:, 1],
        0.0,
        1.0,
    )


def load_graph_history():
    require(ZEEK_GRAPHS)

    import pickle

    with open(
        ZEEK_GRAPHS,
        "rb",
    ) as handle:
        data = pickle.load(handle)

    if isinstance(data, dict):
        for key in (
            "graphs",
            "graph_sequence",
            "sequence",
        ):
            if key in data:
                data = data[key]
                break

    data = list(data)

    if len(data) < 10:
        raise RuntimeError(
            "At least 10 graph states are required."
        )

    return normalize_graph_sequence(
        data[-10:]
    )


def model_forecast(
    model,
    scaler,
    current_history,
    graph_history,
):
    future = forecast_k_steps(
        model=model,
        initial_states=current_history,
        graph_sequence=graph_history,
        scaler=scaler,
        k=5,
    )

    return finite(future)


def normalized_change(current, future):
    current = finite(current)
    future = finite(future)

    return (
        np.abs(future - current)
        / (
            np.abs(current)
            + 1e-6
        )
    )


def feature_importance(model, feature_names):
    estimator = getattr(
        model,
        "model",
        model,
    )

    if not hasattr(
        estimator,
        "feature_importances_",
    ):
        raise RuntimeError(
            "XGBoost estimator does not expose "
            "feature_importances_."
        )

    values = finite(
        estimator.feature_importances_
    )

    if len(values) != len(feature_names):
        raise RuntimeError(
            "Feature importance count does not match "
            "feature names."
        )

    total = float(
        np.sum(values)
    )

    relative = (
        values / total
        if total > 0
        else np.zeros_like(values)
    )

    order = np.argsort(
        -relative
    )

    return [
        {
            "rank": int(rank + 1),
            "feature": feature_names[int(index)],
            "importance": float(values[index]),
            "relative_importance": float(relative[index]),
        }
        for rank, index in enumerate(order)
    ]


def build_temporal_feature_scores(
    current,
    future_states,
    feature_names,
):
    all_states = np.vstack(
        [
            current.reshape(1, -1),
            future_states,
        ]
    )

    step_changes = np.abs(
        np.diff(
            all_states,
            axis=0,
        )
    )

    mean_change = np.mean(
        step_changes,
        axis=0,
    )

    max_change = np.max(
        step_changes,
        axis=0,
    )

    # Consistency = fraction of future transitions where the feature moves.
    consistency = np.mean(
        step_changes > 1e-6,
        axis=0,
    )

    score = (
        mean_change
        * (
            0.5
            + 0.5 * consistency
        )
    )

    order = np.argsort(
        -score
    )

    return [
        {
            "rank": int(rank + 1),
            "feature": feature_names[int(index)],
            "mean_absolute_change": float(
                mean_change[index]
            ),
            "maximum_absolute_change": float(
                max_change[index]
            ),
            "temporal_consistency": float(
                consistency[index]
            ),
            "temporal_influence_score": float(
                score[index]
            ),
        }
        for rank, index in enumerate(order)
    ]


def what_if_analysis(
    model,
    scaler,
    xgb_model,
    xgb_normalizer,
    current_history,
    graph_history,
    ranked_features,
    feature_names,
    baseline_probability,
):
    results = []

    feature_to_index = {
        name: index
        for index, name in enumerate(
            feature_names
        )
    }

    selected = ranked_features[
        :WHAT_IF_FEATURES
    ]

    for item in selected:
        feature = item["feature"]
        index = feature_to_index[feature]

        original = float(
            current_history[-1, index]
        )

        # Avoid zero-valued features producing identical percentage
        # perturbations.
        magnitude = max(
            abs(original),
            1.0,
        )

        for percentage in WHAT_IF_PERTURBATIONS:
            modified_history = (
                current_history.copy()
            )

            modified_value = (
                original
                + magnitude * percentage
            )

            modified_history[
                -1,
                index
            ] = modified_value

            future = model_forecast(
                model,
                scaler,
                modified_history,
                graph_history,
            )

            future_probability = float(
                predict_attack_probability(
                    xgb_model,
                    xgb_normalizer,
                    future[-1:].copy(),
                )[0]
            )

            results.append(
                {
                    "feature": feature,
                    "original_value": original,
                    "perturbation_percent": (
                        float(percentage * 100.0)
                    ),
                    "counterfactual_value": (
                        float(modified_value)
                    ),
                    "baseline_future_attack_probability": (
                        float(baseline_probability)
                    ),
                    "counterfactual_future_attack_probability": (
                        future_probability
                    ),
                    "probability_change": (
                        future_probability
                        - baseline_probability
                    ),
                }
            )

    return results


def human_summary(
    current_probability,
    future_probability,
    top_model_features,
    top_temporal_features,
    what_if,
):
    strongest_model_feature = (
        top_model_features[0]["feature"]
        if top_model_features
        else "unknown"
    )

    strongest_temporal_feature = (
        top_temporal_features[0]["feature"]
        if top_temporal_features
        else "unknown"
    )

    strongest_what_if = None

    if what_if:
        strongest_what_if = max(
            what_if,
            key=lambda x: abs(
                x["probability_change"]
            ),
        )

    text = (
        f"Current XGBoost attack score is "
        f"{current_probability:.2%}. "
        f"The 5-step forecast reaches "
        f"{future_probability:.2%}. "
        f"The strongest current-model feature signal is "
        f"'{strongest_model_feature}', while the largest "
        f"temporal state-change signal is "
        f"'{strongest_temporal_feature}'."
    )

    if strongest_what_if is not None:
        direction = (
            "increased"
            if strongest_what_if[
                "probability_change"
            ] > 0
            else "decreased"
        )

        text += (
            f" In the sensitivity test, changing "
            f"'{strongest_what_if['feature']}' by "
            f"{abs(strongest_what_if['perturbation_percent']):.0f}% "
            f"{direction} the t+5 model score by "
            f"{abs(strongest_what_if['probability_change']):.2%}."
        )

    text += (
        " These explanations indicate model sensitivity "
        "and temporal contribution; they should not be "
        "interpreted as causal proof."
    )

    return text


def main():
    print("=" * 78)
    print("ARJUN PHASE A.4 - EXPLAINABILITY")
    print("=" * 78)

    print("\n[1/6] LOADING PRODUCTION FORECAST")
    print("-" * 78)

    require(PRODUCTION_NPZ)
    require(PRODUCTION_JSON)

    forecast_data = np.load(
        PRODUCTION_NPZ,
        allow_pickle=True,
    )

    current_state = finite(
        forecast_data["current_state"]
    )

    future_states = finite(
        forecast_data["future_states"]
    )

    future_probabilities = finite(
        forecast_data[
            "future_attack_probability"
        ]
    )

    current_probability = float(
        forecast_data[
            "current_attack_probability"
        ]
    )

    if current_state.shape != (33,):
        raise ValueError(
            f"Expected current state (33,), got "
            f"{current_state.shape}"
        )

    if future_states.shape != (5, 33):
        raise ValueError(
            f"Expected future states (5,33), got "
            f"{future_states.shape}"
        )

    print(
        f"Current state      : {current_state.shape}"
    )
    print(
        f"Future states      : {future_states.shape}"
    )
    print(
        f"Current attack     : {current_probability:.2%}"
    )
    print(
        f"t+5 attack score   : "
        f"{float(future_probabilities[-1]):.2%}"
    )
    print("Production artifact: PASS")

    print("\n[2/6] LOADING MODELS")
    print("-" * 78)

    xgb_model, xgb_normalizer, threshold = (
        load_xgb()
    )

    world_model, world_scaler, world_config = (
        load_world_model()
    )

    graph_history = load_graph_history()

    print(
        f"XGBoost threshold : {threshold:.2f}"
    )
    print(
        f"World Model hidden: "
        f"{world_config['hidden_dimension']}"
    )
    print("Models            : PASS")

    print("\n[3/6] CURRENT XGBOOST FEATURE IMPORTANCE")
    print("-" * 78)

    feature_names = list(
        NetworkStateEncoder().feature_names
    )

    model_ranked = feature_importance(
        xgb_model,
        feature_names,
    )

    top_model_features = model_ranked[
        :TOP_K
    ]

    for item in top_model_features:
        print(
            f"{item['rank']:02d}. "
            f"{item['feature']:<30} "
            f"{item['relative_importance']:.2%}"
        )

    print("\n[4/6] TEMPORAL FUTURE-STATE EXPLANATION")
    print("-" * 78)

    temporal_ranked = (
        build_temporal_feature_scores(
            current_state,
            future_states,
            feature_names,
        )
    )

    top_temporal_features = (
        temporal_ranked[:TOP_K]
    )

    for item in top_temporal_features:
        print(
            f"{item['rank']:02d}. "
            f"{item['feature']:<30} "
            f"change={item['mean_absolute_change']:.4f} "
            f"consistency={item['temporal_consistency']:.2f}"
        )

    print("\n[5/6] WHAT-IF SENSITIVITY")
    print("-" * 78)

    baseline_future_probability = float(
        future_probabilities[-1]
    )

    # Use the last 10 observed states from the Zeek state artifact if
    # available. Otherwise the current production state is repeated.
    try:
        if "state_history" in forecast_data:
            current_history = finite(
                forecast_data["state_history"]
            )
        else:
            current_history = np.repeat(
                current_state.reshape(1, -1),
                10,
                axis=0,
            )
    except Exception:
        current_history = np.repeat(
            current_state.reshape(1, -1),
            10,
            axis=0,
        )

    if current_history.shape != (10, 33):
        current_history = np.repeat(
            current_state.reshape(1, -1),
            10,
            axis=0,
        )

    what_if = what_if_analysis(
        world_model,
        world_scaler,
        xgb_model,
        xgb_normalizer,
        current_history,
        graph_history,
        temporal_ranked,
        feature_names,
        baseline_future_probability,
    )

    for item in what_if:
        print(
            f"{item['feature']:<30} "
            f"{item['perturbation_percent']:+.0f}% -> "
            f"{item['counterfactual_future_attack_probability']:.2%} "
            f"("
            f"{item['probability_change']:+.2%}"
            f")"
        )

    print("\n[6/6] SAVING EXPLAINABILITY ARTIFACT")
    print("-" * 78)

    summary = human_summary(
        current_probability,
        baseline_future_probability,
        top_model_features,
        top_temporal_features,
        what_if,
    )

    report = {
        "phase": "A.4",
        "title": "ARJUN Explainability",
        "current_attack_probability": current_probability,
        "future_attack_probability_t5": (
            baseline_future_probability
        ),
        "threshold": threshold,
        "model_feature_importance": model_ranked,
        "temporal_feature_influence": temporal_ranked,
        "what_if_sensitivity": what_if,
        "summary": summary,
        "method_notes": {
            "xgboost": (
                "Native tree feature importance."
            ),
            "temporal": (
                "Feature-state-change and temporal consistency "
                "sensitivity over the 5-step forecast."
            ),
            "what_if": (
                "Counterfactual sensitivity by perturbing one "
                "current feature and rerunning the World Model."
            ),
            "causality": (
                "These are model sensitivity explanations, "
                "not causal claims."
            ),
            "shap": (
                "Not used in this phase."
            ),
        },
        "production_checkpoint_modified": False,
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        current_state=current_state,
        future_states=future_states,
        current_attack_probability=np.float32(
            current_probability
        ),
        future_attack_probability=future_probabilities,
    )

    # Final artifact validation.
    saved = np.load(
        OUTPUT_NPZ,
        allow_pickle=True,
    )

    if saved["future_states"].shape != (
        5,
        33,
    ):
        raise RuntimeError(
            "Explainability NPZ validation failed."
        )

    if not np.isfinite(
        saved["future_states"]
    ).all():
        raise RuntimeError(
            "Explainability output contains NaN/Inf."
        )

    parsed = json.loads(
        OUTPUT_JSON.read_text(
            encoding="utf-8"
        )
    )

    if len(
        parsed["model_feature_importance"]
    ) != 33:
        raise RuntimeError(
            "Feature importance report is incomplete."
        )

    print(
        f"JSON report : {OUTPUT_JSON}"
    )
    print(
        f"NPZ report  : {OUTPUT_NPZ}"
    )
    print(
        "Artifact validation: PASS"
    )

    print("\n" + "=" * 78)
    print("PHASE A.4 EXPLAINABILITY: COMPLETE")
    print("=" * 78)
    print(
        "XGBoost importance + temporal feature influence "
        "+ model-based what-if sensitivity"
    )


if __name__ == "__main__":
    main()
