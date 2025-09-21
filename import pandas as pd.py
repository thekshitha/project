# Save as src/preprocess_spectrum.py
import argparse, os
import numpy as np
import pandas as pd

def watts_to_dbm(pw_w):
    eps = 1e-12
    return 10.0 * np.log10(np.maximum(pw_w, eps) * 1000.0)

def estimate_noise_floor_db(df_power_dbm, pct=5):
    return df_power_dbm.quantile(pct/100.0, axis=0)

def aggregate_bins_to_channels(df_dbm, bin_freqs, channel_map):
    freqs = np.array(bin_freqs)
    out = {}
    for cid, fmin, fmax in channel_map:
        mask = (freqs >= fmin) & (freqs < fmax)
        cols = np.array(df_dbm.columns)[mask]
        if len(cols)==0:
            out[f'chan_{cid}_dbm'] = np.nan
        else:
            # average in linear scale then back to dB
            lin = 10**(df_dbm[cols].astype(float)/10.0)
            mean_lin = lin.mean(axis=1)
            out[f'chan_{cid}_dbm'] = 10*np.log10(np.maximum(mean_lin,1e-12))
    return pd.DataFrame(out, index=df_dbm.index)

def snr_and_occupancy(df_chan_dbm, noise_per_channel, margin_db):
    df = df_chan_dbm.copy()
    for col in df_chan_dbm.columns:
        cid = col.replace('_dbm','')
        noisecol = f'{cid}_noise_db'
        snrcol = f'{cid}_snr_db'
        occcol = f'{cid}_occupied'
        noise_val = noise_per_channel.loc[0, noisecol]
        df[snrcol] = df[col] - noise_val
        df[occcol] = (df[col] > (noise_val + margin_db)).astype(int)
    return df

def map_snr_to_rate(snr_db):
    # simple MCS mapping (example); returns Mbps
    if np.isnan(snr_db): return 0.0
    if snr_db < 5: return 0.25
    if snr_db < 10: return 1.0
    if snr_db < 16: return 6.0
    return 24.0

def main(args):
    df = pd.read_csv(args.input, low_memory=False)
    # parse time
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors='coerce')
    df = df.set_index(args.time_col)

    # power columns start after skip_cols columns
    power_cols = df.columns[args.skip_cols:]
    df_power = df[power_cols].astype(float)

    # convert to dBm if needed
    if args.powers_in_watts:
        df_dbm = watts_to_dbm(df_power)
    else:
        df_dbm = df_power.copy()

    # frequency bin centers: try convert column names to floats
    try:
        bin_freqs = [float(c) for c in power_cols]
    except:
        # if headers are not numeric, assume user provides start/step
        if args.freq_start and args.freq_step:
            n = len(power_cols)
            bin_freqs = [args.freq_start + i*args.freq_step for i in range(n)]
        else:
            raise RuntimeError("Cannot infer frequency bin centers. Provide --freq_start and --freq_step.")

    # define channel_map automatically for common WiFi 2.4 GHz channels (adjust if needed)
    if args.channel_set == '2.4GHz_wifi':
        # create 20 MHz channels across 2.4-2.5 GHz
        channel_map = []
        start = 2.4
        cid = 0
        while start < 2.5:
            channel_map.append((cid, start, start+0.020))
            start += 0.020
            cid += 1
    else:
        channel_map = args.channel_map

    # aggregate
    df_chan = aggregate_bins_to_channels(df_dbm, bin_freqs, channel_map)

    # estimate noise per bin, then per channel (median of bins in channel)
    noise_per_bin = estimate_noise_floor_db(df_dbm, pct=args.noise_pct)
    # build noise per channel row
    noise_row = {}
    freqs = np.array(bin_freqs)
    for cid, fmin, fmax in channel_map:
        mask = (freqs >= fmin) & (freqs < fmax)
        if mask.sum()==0:
            noise_db = np.nan
        else:
            noise_db = noise_per_bin[mask].median()
        noise_row[f'chan_{cid}_noise_db'] = noise_db
    noise_df = pd.DataFrame([noise_row])

    # compute snr and occupancy
    df_snrocc = snr_and_occupancy(df_chan, noise_df, args.threshold_db)

    # compute rate mapping columns
    for cid, _, _ in channel_map:
        snrcol = f'chan_{cid}_snr_db'
        ratecol = f'chan_{cid}_rate_mbps'
        df_snrocc[ratecol] = df_snrocc[snrcol].apply(map_snr_to_rate)

    # resample if requested (slot_ms > 0)
    if args.slot_ms and args.slot_ms > 0:
        df_out = df_snrocc.resample(f'{args.slot_ms}ms').mean()
        # occupancy should be thresholded again after resample:
        for cid, _, _ in channel_map:
            occcol = f'chan_{cid}_occupied'
            df_out[occcol] = (df_out[f'chan_{cid}_dbm'] > (noise_df[f'chan_{cid}_noise_db'].iloc[0] + args.threshold_db)).astype(int)
    else:
        df_out = df_snrocc

    # save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_out.reset_index().rename(columns={'index':'timestamp'}).to_csv(args.out, index=False)
    print("Saved:", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', default='data/processed/processed.csv')
    p.add_argument('--time_col', default='Time')
    p.add_argument('--skip_cols', type=int, default=1)
    p.add_argument('--powers_in_watts', action='store_true')
    p.add_argument('--noise_pct', type=float, default=5.0)
    p.add_argument('--threshold_db', type=float, default=6.0)
    p.add_argument('--slot_ms', type=int, default=500)
    p.add_argument('--freq_start', type=float, default=None)
    p.add_argument('--freq_step', type=float, default=None)
    p.add_argument('--channel_set', default='2.4GHz_wifi')
    p.add_argument('--channel_map', nargs='*', type=float, help="flat list of triples cid fmin fmax", default=None)
    args = p.parse_args()

    # if channel_map provided as flat list convert to tuples
    if args.channel_map:
        lst = args.channel_map
        if len(lst) % 3 != 0:
            raise RuntimeError("channel_map must be triples: cid fmin fmax ...")
        cm = []
        for i in range(0, len(lst), 3):
            cm.append((int(lst[i]), float(lst[i+1]), float(lst[i+2])))
        args.channel_map = cm

    main(args)

