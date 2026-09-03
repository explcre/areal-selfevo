import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "problem", "data")
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output")
)


def get_instance_data(instance_name):
    """Load train and test data for a specific instance."""
    data_path = os.path.join(DATA_DIR, instance_name)
    train_path = os.path.join(data_path, "train.csv")
    test_path = os.path.join(data_path, "x_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def get_treatment_info(df):
    """
    Identify the treatment column and counterfactual column based on naming conventions.
    Assumes columns are named like 'Var' (treatment) and 'Var_prime' (counterfactual).
    """
    # Find columns ending in _prime
    prime_cols = [c for c in df.columns if c.endswith("_prime")]

    if not prime_cols:
        # Fallback if naming convention differs, though description implies it holds
        # Check for standard names if no _prime suffix found (unlikely based on desc)
        pass

    # There should be exactly one pair
    if len(prime_cols) != 1:
        raise ValueError(
            f"Could not uniquely identify treatment/counterfactual columns in {df.columns}"
        )

    cf_col = prime_cols[0]
    # Extract base name (remove '_prime')
    # Handle cases where suffix might be different or base name logic varies
    # Based on description: 'Existing-Account-Status_prime' -> 'Existing-Account-Status'
    # 'DASP14_prime' -> 'DASP14'
    # 'X_prime' -> 'X'

    if cf_col == "X_prime":
        treat_col = "X"
    elif cf_col == "Existing-Account-Status_prime":
        treat_col = "Existing-Account-Status"
    elif cf_col == "DASP14_prime":
        treat_col = "DASP14"
    elif cf_col == "Heparin_prime":
        treat_col = "Heparin"
    else:
        # Generic logic: remove last 6 chars ('_prime')
        treat_col = cf_col[:-6]

    return treat_col, cf_col


def get_target_cols(df):
    """Identify target columns Y and Y_prime (or Y_3, Y_3_prime)."""
    cols = df.columns
    targets = []
    if "Y" in cols and "Y_prime" in cols:
        targets = ["Y", "Y_prime"]
    elif "Y_3" in cols and "Y_3_prime" in cols:
        targets = ["Y_3", "Y_3_prime"]
    return targets


def train_and_predict(instance_name):
    train_df, test_df = get_instance_data(instance_name)

    # 1. Identify columns
    target_cols = get_target_cols(train_df)
    if not target_cols:
        raise ValueError(f"Could not identify target columns for {instance_name}")

    treat_col, cf_col = get_treatment_info(train_df)

    # Feature columns: everything that is not a target
    # Note: train_df has targets, test_df does not.
    # We use train_df columns to define features, excluding targets.
    feature_cols = [c for c in train_df.columns if c not in target_cols]

    # Ensure treatment and cf columns are in feature_cols (they should be)
    if treat_col not in feature_cols:
        raise ValueError(f"Treatment column {treat_col} not found in features")
    if cf_col not in feature_cols:
        raise ValueError(f"Counterfactual column {cf_col} not found in features")

    X_train = train_df[feature_cols]
    y_train = train_df[target_cols[0]]  # We use Y for training the mechanism

    X_test = test_df[feature_cols]

    # 2. Train Model
    # We use LightGBM. It handles multiclass (3 classes) and binary (2 classes) automatically.
    # n_estimators=100 is usually sufficient for this data size.
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        objective="multiclass",  # Will switch to binary if labels are binary
        verbose=-1,
        n_jobs=-1,
    )

    # Fit the model
    clf.fit(X_train, y_train)

    # 3. Predict Factual Outcome
    # Input is just the test features
    y_pred_factual = clf.predict(X_test)

    # 4. Predict Counterfactual Outcome
    # We need to construct the input where the Treatment variable is replaced by the Counterfactual variable.
    X_test_cf = X_test.copy()
    X_test_cf[treat_col] = X_test[cf_col]

    y_pred_counterfactual = clf.predict(X_test_cf)

    # 5. Save Results
    output_path = os.path.join(OUTPUT_DIR, instance_name)
    os.makedirs(output_path, exist_ok=True)

    submission_df = pd.DataFrame(
        {"Y_pred": y_pred_factual, "Y_prime_pred": y_pred_counterfactual}
    )

    submission_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False)
    print(f"Processed {instance_name}")


def main():
    instances = ["german_credit", "ist_aspirin", "ist_heparin", "twin_mortality"]

    for instance in instances:
        try:
            train_and_predict(instance)
        except Exception as e:
            print(f"Error processing {instance}: {e}")
            # Ensure we don't crash the whole pipeline if one fails
            # But for submission, we need all files.
            # We'll re-raise or handle gracefully. Given constraints, we assume data is consistent.
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
