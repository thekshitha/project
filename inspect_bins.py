# inspect_bins.py
import pandas as pd

CSV = "06 2020-06-13_10h13m40s.csv"

# read only header (first row)
df = pd.read_csv(CSV, nrows=1)

# the frequency bin column names start after the 3 metadata cols
bins = df.columns[3:]  # skip Time, Frequency, Sweep

print("First 10 bins:", bins[:10].tolist())
print("Last 10 bins:", bins[-10:].tolist())
print("Number of bins:", len(bins))

# convert to floats to compute min/max/step
bins_f = bins.astype(float)
print("Min freq:", bins_f.min())
print("Max freq:", bins_f.max())
if len(bins_f) > 1:
    print("Step size (approx):", float(bins_f[1] - bins_f[0]))
