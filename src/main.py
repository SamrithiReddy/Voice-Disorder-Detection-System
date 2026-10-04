# ============================================================
# VOICE DISORDER DETECTION SYSTEM
# Feature Extraction + SelectKBest + XGBoost
# ============================================================

import os
import glob

import numpy as np
import librosa

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 65)
    print(f"{title:^65}")
    print("=" * 65)


def print_section(title):
    print("\n" + "-" * 65)
    print(f"{title}")
    print("-" * 65)


# ============================================================
# PROJECT HEADER
# ============================================================

print_header("VOICE DISORDER DETECTION SYSTEM")

print("Model        : XGBoost")
print("Features     : Librosa Acoustic Features")
print("Feature Sel. : SelectKBest (80 features)")
print("Validation   : 5-Fold Stratified Cross-Validation")

print("=" * 65)


# ============================================================
# DATASET PATH
# ============================================================

dataset_path = "../dataset"

healthy_path = os.path.join(
    dataset_path,
    "healthy"
)

pathological_path = os.path.join(
    dataset_path,
    "pathological"
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050
    )

    features = []

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # --------------------------------------------------------
    # Delta MFCC
    # --------------------------------------------------------

    delta = librosa.feature.delta(mfcc)

    features.extend(np.mean(delta, axis=1))
    features.extend(np.std(delta, axis=1))

    # --------------------------------------------------------
    # Delta-Delta MFCC
    # --------------------------------------------------------

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features.extend(np.mean(delta2, axis=1))
    features.extend(np.std(delta2, axis=1))

    # --------------------------------------------------------
    # Spectral Centroid
    # --------------------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    features.extend(np.mean(spectral_centroid, axis=1))
    features.extend(np.std(spectral_centroid, axis=1))

    # --------------------------------------------------------
    # Spectral Bandwidth
    # --------------------------------------------------------

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    features.extend(np.mean(spectral_bandwidth, axis=1))
    features.extend(np.std(spectral_bandwidth, axis=1))

    # --------------------------------------------------------
    # Zero Crossing Rate
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(audio)

    features.extend(np.mean(zcr, axis=1))
    features.extend(np.std(zcr, axis=1))

    # --------------------------------------------------------
    # RMS Energy
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio
    )

    features.extend(np.mean(rms, axis=1))
    features.extend(np.std(rms, axis=1))

    # --------------------------------------------------------
    # Spectral Contrast
    # --------------------------------------------------------

    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr
    )

    features.extend(np.mean(spectral_contrast, axis=1))
    features.extend(np.std(spectral_contrast, axis=1))

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        tuning=0
    )

    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    return np.array(features)


# ============================================================
# LOAD DATASET
# ============================================================

print_section("DATASET INFORMATION")

X = []
y = []

# Healthy = 0
healthy_files = glob.glob(
    os.path.join(
        healthy_path,
        "*.wav"
    )
)

for file_path in healthy_files:

    try:

        features = extract_features(file_path)

        X.append(features)
        y.append(0)

    except Exception as e:

        print(
            f"Warning: Could not process {file_path}: {e}"
        )


# Pathological = 1
pathological_files = glob.glob(
    os.path.join(
        pathological_path,
        "*.wav"
    )
)

for file_path in pathological_files:

    try:

        features = extract_features(file_path)

        X.append(features)
        y.append(1)

    except Exception as e:

        print(
            f"Warning: Could not process {file_path}: {e}"
        )


X = np.array(X)
y = np.array(y)


print(f"Total Samples        : {len(X)}")
print(f"Healthy Samples      : {np.sum(y == 0)}")
print(f"Pathological Samples : {np.sum(y == 1)}")
print(f"Original Features    : {X.shape[1]}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# FEATURE SELECTION
# ============================================================

selector = SelectKBest(
    score_func=f_classif,
    k=80
)

X_train = selector.fit_transform(
    X_train,
    y_train
)

X_test = selector.transform(
    X_test
)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# ============================================================
# DATA SPLIT INFORMATION
# ============================================================

print_section("DATA PREPARATION")

print(f"Training Samples     : {X_train.shape[0]}")
print(f"Testing Samples      : {X_test.shape[0]}")
print(f"Selected Features    : {X_train.shape[1]}")
print(f"Feature Reduction    : {X.shape[1]} → {X_train.shape[1]}")


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBClassifier(

    n_estimators=500,

    max_depth=3,

    learning_rate=0.03,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=2,

    gamma=0,

    reg_alpha=0.05,

    reg_lambda=1.5,

    eval_metric="logloss",

    random_state=42
)


# ============================================================
# MODEL TRAINING
# ============================================================

print_section("MODEL TRAINING")

print("Training XGBoost model...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# TEST SET EVALUATION
# ============================================================

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print_section("TEST SET PERFORMANCE")

print(
    f"Accuracy             : {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Healthy",
        "Pathological"
    ]
)

print("\nClassification Report:")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print()

print("                 Predicted")
print("              Healthy  Pathological")
print(f"Actual Healthy     {cm[0][0]:3d}        {cm[0][1]:3d}")
print(f"Actual Path.       {cm[1][0]:3d}        {cm[1][1]:3d}")


# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

print_section("5-FOLD CROSS-VALIDATION")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_model = Pipeline([

    (
        "feature_selection",
        SelectKBest(
            score_func=f_classif,
            k=80
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "xgb",
        XGBClassifier(

            n_estimators=500,

            max_depth=3,

            learning_rate=0.03,

            subsample=0.85,

            colsample_bytree=0.85,

            min_child_weight=2,

            gamma=0,

            reg_alpha=0.05,

            reg_lambda=1.5,

            eval_metric="logloss",

            random_state=42
        )
    )
])


cv_scores = cross_val_score(
    cv_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print("Fold Results:")

for i, score in enumerate(cv_scores, start=1):

    print(
        f"  Fold {i}                : {score * 100:.2f}%"
    )


cv_mean = cv_scores.mean()
cv_std = cv_scores.std()


print()
print(
    f"Mean CV Accuracy       : {cv_mean * 100:.2f}%"
)

print(
    f"Standard Deviation     : ±{cv_std * 100:.2f}%"
)


# ============================================================
# SINGLE AUDIO PREDICTION
# ============================================================

def predict_voice(test_file):

    print_section("VOICE PREDICTION")

    print(
        f"Input File             : {os.path.basename(test_file)}"
    )

    print(
        "Processing audio..."
    )

    features = extract_features(
        test_file
    )

    features = features.reshape(
        1,
        -1
    )

    # Feature selection
    features = selector.transform(
        features
    )

    # Scaling
    features = scaler.transform(
        features
    )

    # Prediction
    prediction = model.predict(
        features
    )

    probability = model.predict_proba(
        features
    )

    if prediction[0] == 0:

        result = "HEALTHY"

    else:

        result = "PATHOLOGICAL"


    healthy_probability = (
        probability[0][0] * 100
    )

    pathological_probability = (
        probability[0][1] * 100
    )


    print()
    print("Prediction Result")
    print("-----------------")

    print(
        f"Diagnosis              : {result}"
    )

    print(
        f"Healthy Probability    : {healthy_probability:.2f}%"
    )

    print(
        f"Pathological Probability: {pathological_probability:.2f}%"
    )


    return result


# ============================================================
# TEST AUDIO FILE
# ============================================================

test_files = glob.glob(
    os.path.join(
        healthy_path,
        "*.wav"
    )
)


if len(test_files) > 0:

    test_file = test_files[0]

    predict_voice(
        test_file
    )

else:

    print_section("VOICE PREDICTION")

    print(
        "No .wav test file found."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header("FINAL MODEL SUMMARY")

print(
    f"Test Accuracy         : {accuracy * 100:.2f}%"
)

print(
    f"5-Fold CV Accuracy    : {cv_mean * 100:.2f}% ± {cv_std * 100:.2f}%"
)

print(
    f"Features Used         : {X_train.shape[1]}"
)

print(
    "Model                 : Tuned XGBoost"
)

print(
    "Feature Selection     : SelectKBest"
)

print(
    "Scaling               : StandardScaler"
)

print("=" * 65)
print("Execution completed successfully.")
print("=" * 65)