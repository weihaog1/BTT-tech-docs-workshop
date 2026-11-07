"""Train the hotel cancellation model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import clean_bookings

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hotel_bookings_sample.csv"
MODEL_DIR = ROOT / "models"
TARGET = "is_canceled"

NUMERIC = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "total_nights",
    "adults", "children", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests", "has_agent",
]
CATEGORICAL = [
    "hotel", "arrival_date_month", "meal", "market_segment", "distribution_channel",
    "deposit_type", "customer_type", "reserved_room_type",
]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def build_pipeline():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])


def split_data(df):
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    df = clean_bookings(load_data())
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_pipeline().fit(X_train, y_train)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    joblib.dump((X_test, y_test), MODEL_DIR / "test_set.joblib")
    print(f"Trained on {len(X_train):,} bookings.")
    print("Saved the model and test set to models/.")


if __name__ == "__main__":
    main()
