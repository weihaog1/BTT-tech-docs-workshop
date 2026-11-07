# Data

`hotel_bookings_sample.csv` contains 20,000 sampled hotel bookings from 2015 to 2017, for one city hotel and one resort hotel in Portugal. Each row is one booking. `is_canceled` is 1 if the booking was canceled.

The 20,000-row figure describes the supplied CSV before the cleaning steps in `src/features.py`. The training workflow removes duplicate rows before splitting the cleaned data into training and test sets. Do not describe all 20,000 rows as the model's test set.

Source: Nuno Antonio, Ana de Almeida, and Luis Nunes, "Hotel booking demand datasets," *Data in Brief* 22 (2019): 41-49. Licensed under CC BY 4.0. We used the TidyTuesday copy (2020-02-11).

We drew a random sample (`random_state=42`) and made changes for teaching purposes.
