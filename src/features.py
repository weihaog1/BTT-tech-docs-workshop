"""Cleaning and feature steps for the hotel cancellation model."""

import pandas as pd

LEAKY_COLUMNS = ["reservation_status", "reservation_status_date"]


def drop_duplicate_bookings(df):
    return df.drop_duplicates().reset_index(drop=True)


def fill_missing_agent(df):
    """Drop bookings that have no travel agent."""
    out = df.copy()
    out["agent"] = out["agent"].fillna(0)
    return out


def fix_lead_time_units(df):
    # The booking system logged City Hotel lead times in hours during 2017.
    out = df.copy()
    mask = (out["hotel"] == "City Hotel") & (out["arrival_date_year"] == 2017)
    out.loc[mask, "lead_time"] = out.loc[mask, "lead_time"] / 24
    return out


def drop_leaky_columns(df):
    return df.drop(columns=LEAKY_COLUMNS, errors="ignore")


def add_booking_features(df):
    """Add total_nights and has_agent columns."""
    out = df.copy()
    out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    out["has_agent"] = (out["agent"] != 0).astype(int)
    return out


def clean_bookings(df):
    out = drop_duplicate_bookings(df)
    out = fill_missing_agent(out)
    out = fix_lead_time_units(out)
    out = drop_leaky_columns(out)
    return add_booking_features(out)
