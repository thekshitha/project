# check_processed.py
import pandas as pd

CSV = "/Users/hrishikeshduraisamy/Desktop/Portfolio/project/processed.csv"

# Try reading without the deprecated argument; allow errors in parsing and show basic info
try:
    df = pd.read_csv(CSV)
except Exception as e:
    print("ERROR reading CSV:", e)
    raise

print("Columns:", df.columns.tolist())
print("\nRow count:", len(df))

# If there's a 'timestamp' or 'Time' column, try to parse it safely
if 'timestamp' in df.columns:
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except Exception as e:
        print("Timestamp parse error:", e)
elif 'Time' in df.columns:
    try:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df.rename(columns={'Time':'timestamp'}, inplace=True)
    except Exception as e:
        print("Time parse error:", e)
else:
    print("No timestamp/Time column found; continuing without datetime parsing.")

print("\nFirst 8 rows:")
print(df.head(8).to_string(index=False))

# Show occupancy columns distribution (safe detection)
occ_cols = [c for c in df.columns if c.endswith('_occupied')]
if not occ_cols:
    print("\nNo '*_occupied' columns found in CSV.")
else:
    print("\nOccupancy value counts (per channel):")
    for c in occ_cols:
        counts = df[c].value_counts(dropna=False).to_dict()
        print(f"  {c}: {counts}")

# Show a sample of SNR columns if available
snr_cols = [c for c in df.columns if c.endswith('_snr_db')]
if snr_cols:
    print("\nSNR sample (first row):")
    r0 = df.iloc[0]
    for c in snr_cols[:10]:
        print(f"  {c}: {r0.get(c)}")
