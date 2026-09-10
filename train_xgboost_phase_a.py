from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.normalizer import NetworkStateNormalizer
from evaluation.xgboost_classifier import (
    XGBoostAttackClassifier,
    XGBoostAttackFamilyClassifier,
)


STATE = ROOT / "data" / "processed" / "training_states.npz"
FAMILY = ROOT / "data" / "processed" / "unseen_attack_state_labels.npz"

MODEL = ROOT / "saved_models" / "xgboost_attack_classifier.pkl"
FAMILY_MODEL = ROOT / "saved_models" / "xgboost_attack_family_classifier.pkl"
NORM = ROOT / "saved_models" / "state_normalizer.pkl"

REPORT = ROOT / "evaluation" / "xgboost_phase_a_report.json"


def main():
    print("=" * 78)
    print("ARJUN PHASE A - XGBOOST")
    print("=" * 78)

    print("\nCHECKING INPUT FILES")
    print("-" * 78)
    print(f"Project root : {ROOT}")
    print(f"State file   : {STATE}")
    print(f"Family file  : {FAMILY}")

    if not STATE.exists():
        raise FileNotFoundError(f"Training state file not found:\n{STATE}")

    if not FAMILY.exists():
        raise FileNotFoundError(
            f"Attack-family mapping not found:\n{FAMILY}\n\n"
            "Run:\n"
            "python evaluation\\unseen_attack_generalization.py"
        )

    print("PASS  training_states.npz")
    print("PASS  unseen_attack_state_labels.npz")

    print("\nLOADING NETWORK STATES")
    print("-" * 78)

    with np.load(STATE, allow_pickle=True) as data:
        X = np.asarray(data["states"], dtype=np.float32)
        y = np.asarray(data["labels"]).reshape(-1)

    with np.load(FAMILY, allow_pickle=True) as data:
        fam = np.asarray(data["state_families"], dtype=str).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"Expected 2-D states, got {X.shape}")

    if X.shape[1] != 33:
        raise ValueError(f"Expected 33 features, got {X.shape[1]}")

    if len(X) != len(y) or len(X) != len(fam):
        raise ValueError(
            f"State/label/family alignment mismatch: "
            f"{len(X)}, {len(y)}, {len(fam)}"
        )

    if "UNMAPPED" in set(fam):
        raise RuntimeError("UNMAPPED family labels exist.")

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    n = len(X)
    a = int(n * 0.70)
    b = int(n * 0.85)

    tr, va, te = X[:a], X[a:b], X[b:]
    ytr, yv, yte = y[:a], y[a:b], y[b:]
    ftr, fv, fte = fam[:a], fam[a:b], fam[b:]

    from world_model.state_encoder import NetworkStateEncoder

    names = list(NetworkStateEncoder().feature_names)

    if len(names) != 33:
        raise RuntimeError(f"Expected 33 feature names, got {len(names)}")

    print(f"\nStates          : {n:,}")
    print(f"Features        : {X.shape[1]}")
    print(f"Training        : {len(tr):,}")
    print(f"Validation      : {len(va):,}")
    print(f"Test            : {len(te):,}")
    print(f"Family classes  : {len(np.unique(fam))}")

    print("\nATTACK-FAMILY DISTRIBUTION")
    print("-" * 78)

    unique, counts = np.unique(fam, return_counts=True)

    for family_name, count in sorted(
        zip(unique, counts),
        key=lambda item: (-int(item[1]), str(item[0])),
    ):
        print(f"{str(family_name):<40} {int(count):>10,}")

    print("\nNORMALIZATION")
    print("-" * 78)

    norm = NetworkStateNormalizer().fit(tr)

    trn = norm.transform(tr)
    van = norm.transform(va)
    ten = norm.transform(te)

    if not np.isfinite(trn).all():
        raise RuntimeError("Normalized training data contains NaN/Inf.")

    if not np.isfinite(van).all():
        raise RuntimeError("Normalized validation data contains NaN/Inf.")

    if not np.isfinite(ten).all():
        raise RuntimeError("Normalized test data contains NaN/Inf.")

    print("PASS  Normalizer fitted on training data only")

    print("\n" + "=" * 78)
    print("[1/2] BINARY ATTACK MODEL")
    print("=" * 78)

    binary = XGBoostAttackClassifier()

    binary.fit(
        trn,
        ytr,
        feature_names=names,
        validation_data=(van, yv),
    )

    bm = binary.evaluate(ten, yte)

    print(f"\nTest F1        : {bm['f1']:.4f}")
    print(f"Test Precision : {bm['precision']:.4f}")
    print(f"Test Recall    : {bm['recall']:.4f}")
    print(f"Test FPR       : {bm['fpr']:.4f}")

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    binary.save(MODEL)

    print(f"\nSaved binary model:\n{MODEL}")

    print("\n" + "=" * 78)
    print("[2/2] MULTICLASS ATTACK-FAMILY MODEL")
    print("=" * 78)

    family = XGBoostAttackFamilyClassifier()

    family.fit(
        trn,
        ftr,
        feature_names=names,
        validation_data=(van, fv),
    )

    fm = family.evaluate(ten, fte)

    print(f"\nTest Accuracy  : {fm['accuracy']:.4f}")
    print(f"Macro Precision: {fm['macro_precision']:.4f}")
    print(f"Macro Recall   : {fm['macro_recall']:.4f}")
    print(f"Macro F1       : {fm['macro_f1']:.4f}")
    print(f"Weighted F1    : {fm['weighted_f1']:.4f}")

    print("\nLEARNED ATTACK CLASSES")
    print("-" * 78)

    for index, class_name in enumerate(family.classes_):
        print(f"{index:02d} : {class_name}")

    FAMILY_MODEL.parent.mkdir(parents=True, exist_ok=True)
    family.save(FAMILY_MODEL)

    print(f"\nSaved family model:\n{FAMILY_MODEL}")

    norm.save(NORM)

    print(f"Saved normalizer:\n{NORM}")

    print("\n" + "=" * 78)
    print("SAMPLE FAMILY PREDICTIONS")
    print("=" * 78)

    for index, prediction in enumerate(
        family.predict_with_confidence(ten[:5]),
        start=1,
    ):
        print(
            f"Sample {index}: "
            f"{prediction['family']} "
            f"({prediction['confidence']:.3f})"
        )

    print("\n" + "=" * 78)
    print("ARTIFACT LOAD SMOKE TEST")
    print("=" * 78)

    loaded_binary = XGBoostAttackClassifier().load(MODEL)
    loaded_family = XGBoostAttackFamilyClassifier().load(FAMILY_MODEL)
    loaded_normalizer = NetworkStateNormalizer().load(NORM)

    sample = loaded_normalizer.transform(te[:5])

    probabilities = loaded_binary.predict_proba(sample)
    predictions = loaded_family.predict(sample)

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Binary probability smoke test produced NaN/Inf.")

    if len(predictions) != len(sample):
        raise RuntimeError("Family prediction count mismatch.")

    print("Binary model load       : PASS")
    print("Family model load       : PASS")
    print("Normalizer load         : PASS")
    print("Probability finite      : PASS")
    print("Family prediction count : PASS")

    report = {
        "experiment": "arjun_phase_a_xgboost",
        "state_dimension": int(X.shape[1]),
        "state_count": int(n),
        "family_classes": [str(x) for x in family.classes_],
        "split": {
            "train": int(len(tr)),
            "validation": int(len(va)),
            "test": int(len(te)),
        },
        "binary_test": bm,
        "family_test": {
            "accuracy": fm["accuracy"],
            "macro_precision": fm["macro_precision"],
            "macro_recall": fm["macro_recall"],
            "macro_f1": fm["macro_f1"],
            "weighted_f1": fm["weighted_f1"],
        },
        "artifacts": {
            "binary": str(MODEL.relative_to(ROOT)),
            "family": str(FAMILY_MODEL.relative_to(ROOT)),
            "normalizer": str(NORM.relative_to(ROOT)),
        },
        "world_model_modified": False,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("PHASE A XGBOOST TRAINER: READY")
    print("=" * 78)
    print(f"Binary model : {MODEL}")
    print(f"Family model : {FAMILY_MODEL}")
    print(f"Normalizer   : {NORM}")
    print(f"Report       : {REPORT}")


if __name__ == "__main__":
    main()
