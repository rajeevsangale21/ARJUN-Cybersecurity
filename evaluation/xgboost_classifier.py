from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class XGBoostAttackClassifier:
    """
    XGBoost attack classifier used by ARJUN.

    The classifier predicts:

        P(attack | network_state)
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ):

        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. "
                "Install dependencies using "
                "pip install -r requirements.txt"
            )

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

        self.feature_names = None
        self.is_fitted = False

    @staticmethod
    def _prepare_X(X):
        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim == 3:
            X = X[:, -1, :]

        if X.ndim != 2:
            raise ValueError(
                "X must have shape "
                "(samples, features) or "
                "(samples, sequence, features)."
            )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return X

    @staticmethod
    def _prepare_y(y):

        y = np.asarray(y)

        if y.ndim > 1:
            y = y.reshape(-1)

        result = []

        for value in y:

            if isinstance(
                value,
                str,
            ):

                text = (
                    value
                    .strip()
                    .lower()
                )

                normal_values = {
                    "benign",
                    "normal",
                    "0",
                    "false",
                    "none",
                }

                result.append(
                    0
                    if text in normal_values
                    else 1
                )

            else:

                try:
                    result.append(
                        0
                        if float(value) == 0
                        else 1
                    )

                except Exception:
                    result.append(1)

        return np.asarray(
            result,
            dtype=np.int32,
        )

    def fit(
        self,
        X,
        y,
        feature_names=None,
        validation_data=None,
    ):

        X = self._prepare_X(X)
        y = self._prepare_y(y)

        if len(X) == 0:
            raise ValueError(
                "Training data is empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y have different "
                "numbers of samples."
            )

        unique_classes = np.unique(y)

        # XGBoost cannot learn a binary boundary
        # from only one class.
        if len(unique_classes) < 2:

            raise ValueError(
                "XGBoost training requires both "
                "normal and attack samples. "
                f"Found classes: {unique_classes.tolist()}"
            )

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else None
        )

        eval_set = None

        if validation_data is not None:

            X_val, y_val = validation_data

            X_val = self._prepare_X(
                X_val
            )

            y_val = self._prepare_y(
                y_val
            )

            if len(X_val) > 0:

                eval_set = [
                    (
                        X_val,
                        y_val,
                    )
                ]

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=False,
        )

        self.is_fitted = True

        return self

    def predict_proba(
        self,
        X,
    ):

        if not self.is_fitted:
            raise RuntimeError(
                "Classifier has not been fitted."
            )

        X = self._prepare_X(X)

        probabilities = (
            self.model.predict_proba(X)
        )

        if probabilities.ndim == 2:

            if probabilities.shape[1] >= 2:
                return probabilities[:, 1]

            return probabilities[:, 0]

        return probabilities.reshape(-1)

    def predict(
        self,
        X,
        threshold=0.5,
    ):

        probabilities = (
            self.predict_proba(X)
        )

        return (
            probabilities >= threshold
        ).astype(int)

    def evaluate(
        self,
        X,
        y,
        threshold=0.5,
    ):

        y_true = self._prepare_y(y)

        y_pred = self.predict(
            X,
            threshold=threshold,
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        return {
            "accuracy":
                float(
                    accuracy_score(
                        y_true,
                        y_pred,
                    )
                ),

            "precision":
                float(
                    precision_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),

            "recall":
                float(
                    recall_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),

            "f1":
                float(
                    f1_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),

            "fpr":
                float(fpr),

            "true_negatives":
                int(tn),

            "false_positives":
                int(fp),

            "false_negatives":
                int(fn),

            "true_positives":
                int(tp),
        }

    def feature_importance(self):

        if not self.is_fitted:
            raise RuntimeError(
                "Classifier has not been fitted."
            )

        values = (
            self.model.feature_importances_
        )

        if self.feature_names is None:

            names = [
                f"feature_{i}"
                for i in range(len(values))
            ]

        else:

            names = self.feature_names

            if len(names) != len(values):

                names = [
                    f"feature_{i}"
                    for i in range(len(values))
                ]

        pairs = list(
            zip(
                names,
                values,
            )
        )

        pairs.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            {
                "feature": name,
                "importance": float(value),
            }
            for name, value in pairs
        ]

    def save(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "model":
                    self.model,

                "feature_names":
                    self.feature_names,

                "is_fitted":
                    self.is_fitted,
            },
            path,
        )

    def load(self, path):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Classifier not found: {path}"
            )

        checkpoint = joblib.load(
            path
        )

        if isinstance(
            checkpoint,
            dict,
        ):

            self.model = checkpoint[
                "model"
            ]

            self.feature_names = (
                checkpoint.get(
                    "feature_names"
                )
            )

            self.is_fitted = checkpoint.get(
                "is_fitted",
                True,
            )

        else:

            self.model = checkpoint
            self.is_fitted = True

        return self

    @classmethod
    def from_file(
        cls,
        path,
    ):

        instance = cls()

        instance.load(
            path
        )

        return instance

class XGBoostAttackFamilyClassifier:
    """Multiclass XGBoost classifier for CIC-IDS-2018 attack families."""
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=42):
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, subsample=subsample,
                           colsample_bytree=colsample_bytree, random_state=random_state)
        self.model = None
        self.feature_names = None
        self.classes_ = None
        self.class_to_index = None
        self.is_fitted = False

    @staticmethod
    def _prepare_X(X):
        return XGBoostAttackClassifier._prepare_X(X)

    @staticmethod
    def _prepare_y(y):
        values = np.asarray(y).reshape(-1)
        return np.asarray([str(v).strip() if str(v).strip() else "Benign" for v in values], dtype=str)

    def fit(self, X, y, feature_names=None, validation_data=None):
        X = self._prepare_X(X); y = self._prepare_y(y)
        if len(X) != len(y) or len(X) == 0:
            raise ValueError("Invalid family training data.")
        self.classes_ = np.asarray(sorted(set(y.tolist()), key=str.casefold), dtype=str)
        if len(self.classes_) < 2:
            raise ValueError("Family classifier needs at least two classes.")
        self.class_to_index = {c:i for i,c in enumerate(self.classes_)}
        y_encoded = np.asarray([self.class_to_index[v] for v in y], dtype=np.int32)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.model = XGBClassifier(objective="multi:softprob", num_class=len(self.classes_),
                                   eval_metric="mlogloss", n_jobs=-1, **self.params)
        eval_set = None
        if validation_data is not None:
            xv, yv = validation_data; xv=self._prepare_X(xv); yv=self._prepare_y(yv)
            unknown = sorted(set(yv.tolist()) - set(self.class_to_index))
            if unknown: raise ValueError(f"Validation has unseen family labels: {unknown}")
            eval_set=[(xv, np.asarray([self.class_to_index[v] for v in yv], dtype=np.int32))]
        self.model.fit(X, y_encoded, eval_set=eval_set, verbose=False)
        self.is_fitted=True
        return self

    def predict_proba(self, X):
        if not self.is_fitted: raise RuntimeError("Family classifier has not been fitted.")
        return np.asarray(self.model.predict_proba(self._prepare_X(X)), dtype=np.float32)

    def predict(self, X):
        p=self.predict_proba(X); return self.classes_[np.argmax(p, axis=1)]

    def predict_with_confidence(self, X):
        p=self.predict_proba(X); out=[]
        for row in p:
            i=int(np.argmax(row)); out.append({"family":str(self.classes_[i]), "confidence":float(row[i])})
        return out

    def evaluate(self, X, y):
        from sklearn.metrics import classification_report
        yt=self._prepare_y(y); yp=self.predict(X); labels=self.classes_.tolist()
        return {"accuracy":float(accuracy_score(yt,yp)),
                "macro_precision":float(precision_score(yt,yp,labels=labels,average="macro",zero_division=0)),
                "macro_recall":float(recall_score(yt,yp,labels=labels,average="macro",zero_division=0)),
                "macro_f1":float(f1_score(yt,yp,labels=labels,average="macro",zero_division=0)),
                "weighted_f1":float(f1_score(yt,yp,labels=labels,average="weighted",zero_division=0)),
                "classification_report":classification_report(yt,yp,labels=labels,target_names=labels,output_dict=True,zero_division=0)}

    def feature_importance(self):
        if not self.is_fitted: raise RuntimeError("Family classifier has not been fitted.")
        values=np.asarray(self.model.feature_importances_)
        names=self.feature_names if self.feature_names is not None and len(self.feature_names)==len(values) else [f"feature_{i}" for i in range(len(values))]
        return [{"feature":n,"importance":float(v)} for n,v in sorted(zip(names,values), key=lambda x:x[1], reverse=True)]

    def save(self, path):
        if not self.is_fitted: raise RuntimeError("Cannot save an unfitted family classifier.")
        path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"artifact_type":"arjun_xgboost_attack_family","artifact_version":1,
                     "model":self.model,"feature_names":self.feature_names,
                     "classes":self.classes_,"class_to_index":self.class_to_index,
                     "is_fitted":True,"params":self.params}, path)

    def load(self, path):
        path=Path(path)
        if not path.exists(): raise FileNotFoundError(f"Family classifier not found: {path}")
        ck=joblib.load(path)
        if not isinstance(ck,dict) or "model" not in ck: raise ValueError("Invalid family classifier artifact.")
        self.model=ck["model"]; self.feature_names=ck.get("feature_names")
        self.classes_=np.asarray(ck.get("classes"),dtype=str)
        self.class_to_index=ck.get("class_to_index") or {c:i for i,c in enumerate(self.classes_)}
        self.is_fitted=ck.get("is_fitted",True); return self

    @classmethod
    def from_file(cls,path): return cls().load(path)

