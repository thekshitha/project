import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the processed data
df = pd.read_csv('processed_data.csv')

# Convert Time to datetime if not already
df['Time'] = pd.to_datetime(df['Time'])

# Split data into training (80%) and testing (20%) sets
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

print("\nDataset Information:")
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")

# Analyze channel characteristics
def analyze_channel(data, channel_num):
    print(f"\nChannel {channel_num} Analysis:")
    
    # Get channel columns
    dbm_col = f'chan_{channel_num}_dbm'
    snr_col = f'chan_{channel_num}_snr_db'
    occ_col = f'chan_{channel_num}_occupied'
    rate_col = f'chan_{channel_num}_rate_mbps'
    
    # Calculate statistics
    print(f"Average Power (dBm): {data[dbm_col].mean():.2f}")
    print(f"Average SNR (dB): {data[snr_col].mean():.2f}")
    print(f"Channel Occupancy Rate: {(data[occ_col].sum() / len(data)) * 100:.2f}%")
    print(f"Average Data Rate (Mbps): {data[rate_col].mean():.2f}")

# Analyze each channel in training data
print("\nTraining Data Analysis:")
for i in range(5):  # We have 5 channels
    analyze_channel(train_data, i)

# Plot channel occupancy over time
plt.figure(figsize=(15, 8))
for i in range(5):
    plt.plot(train_data['Time'], train_data[f'chan_{i}_occupied'], 
             label=f'Channel {i}', alpha=0.7)
plt.title('Channel Occupancy Over Time (Training Data)')
plt.xlabel('Time')
plt.ylabel('Occupied (1) / Free (0)')
plt.legend()
plt.grid(True)
plt.savefig('channel_occupancy.png')
plt.close()

# Plot SNR distribution
plt.figure(figsize=(15, 8))
for i in range(5):
    plt.hist(train_data[f'chan_{i}_snr_db'], 
             bins=50, alpha=0.5, label=f'Channel {i}')
plt.title('SNR Distribution (Training Data)')
plt.xlabel('SNR (dB)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('snr_distribution.png')
plt.close()

# Save training and testing datasets
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Analyze test data
print("\nTest Data Analysis:")
for i in range(5):
    analyze_channel(test_data, i)

# Plot channel occupancy over time for test data
plt.figure(figsize=(15, 8))
for i in range(5):
    plt.plot(test_data['Time'], test_data[f'chan_{i}_occupied'], 
             label=f'Channel {i}', alpha=0.7)
plt.title('Channel Occupancy Over Time (Test Data)')
plt.xlabel('Time')
plt.ylabel('Occupied (1) / Free (0)')
plt.legend()
plt.grid(True)
plt.savefig('test_channel_occupancy.png')
plt.close()

# Plot SNR distribution for test data
plt.figure(figsize=(15, 8))
for i in range(5):
    plt.hist(test_data[f'chan_{i}_snr_db'], 
             bins=50, alpha=0.5, label=f'Channel {i}')
plt.title('SNR Distribution (Test Data)')
plt.xlabel('SNR (dB)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('test_snr_distribution.png')
plt.close()

# Power Levels Over Time
plt.figure(figsize=(15, 8))
for i in range(5):
    plt.plot(train_data['Time'], train_data[f'chan_{i}_dbm'], 
             label=f'Channel {i}', alpha=0.7)
plt.title('Power Levels Over Time')
plt.xlabel('Time')
plt.ylabel('Power (dBm)')
plt.legend()
plt.grid(True)
plt.savefig('power_levels.png')
plt.close()

# Data Rate Distribution
plt.figure(figsize=(15, 8))
for i in range(5):
    plt.hist(train_data[f'chan_{i}_rate_mbps'], 
             bins=50, alpha=0.5, label=f'Channel {i}')
plt.title('Data Rate Distribution')
plt.xlabel('Data Rate (Mbps)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('data_rate_distribution.png')
plt.close()

# Channel Statistics Comparison
channels = range(5)
metrics = {
    'Power (dBm)': [train_data[f'chan_{i}_dbm'].mean() for i in channels],
    'SNR (dB)': [train_data[f'chan_{i}_snr_db'].mean() for i in channels],
    'Occupancy (%)': [train_data[f'chan_{i}_occupied'].mean() * 100 for i in channels],
    'Data Rate (Mbps)': [train_data[f'chan_{i}_rate_mbps'].mean() for i in channels]
}

# Bar plot for channel statistics
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Channel Statistics Comparison')

for (metric, values), ax in zip(metrics.items(), axes.ravel()):
    ax.bar(channels, values)
    ax.set_title(f'Average {metric} by Channel')
    ax.set_xlabel('Channel')
    ax.set_ylabel(metric)
    ax.grid(True)

plt.tight_layout()
plt.savefig('channel_statistics.png')
plt.close()

# Channel Quality Score
# Normalize and combine SNR and Data Rate for an overall quality score
quality_scores = []
for i in channels:
    snr_norm = (train_data[f'chan_{i}_snr_db'] - train_data[f'chan_{i}_snr_db'].min()) / \
               (train_data[f'chan_{i}_snr_db'].max() - train_data[f'chan_{i}_snr_db'].min())
    rate_norm = (train_data[f'chan_{i}_rate_mbps'] - train_data[f'chan_{i}_rate_mbps'].min()) / \
                (train_data[f'chan_{i}_rate_mbps'].max() - train_data[f'chan_{i}_rate_mbps'].min())
    quality_score = (snr_norm + rate_norm) / 2
    quality_scores.append(quality_score.mean())

plt.figure(figsize=(10, 6))
plt.bar(channels, quality_scores)
plt.title('Channel Quality Score (Normalized SNR + Data Rate)')
plt.xlabel('Channel')
plt.ylabel('Quality Score')
plt.grid(True)
plt.savefig('channel_quality_scores.png')
plt.close()

print("\nAnalysis completed! The following visualization files have been generated:")
print("1. channel_occupancy.png - Channel occupation patterns over time")
print("2. snr_distribution.png - Distribution of SNR values for each channel")
print("3. power_levels.png - Power levels over time for each channel")
print("4. data_rate_distribution.png - Distribution of data rates")
print("5. channel_statistics.png - Comparative bar plots of channel metrics")
print("6. channel_quality_scores.png - Overall channel quality scores")
print("\nTest data visualizations:")
print("7. test_channel_occupancy.png - Channel occupation patterns in test data")
print("8. test_snr_distribution.png - SNR distribution in test data")