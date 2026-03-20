import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 5000

# Create folder automatically
os.makedirs("data", exist_ok=True)

time = pd.date_range(start="2024-01-01", periods=n, freq="h")

df = pd.DataFrame({
    "time": time,
    "temperature": np.random.normal(300, 10, n),
    "vibration": np.random.normal(50, 10, n),
    "pressure": np.random.normal(100, 20, n),
})

# Add randomness (real-world behavior)
prob = (
    0.3 * (df["temperature"] > 310) +
    0.3 * (df["vibration"] > 65) +
    0.4 * (df["pressure"] > 130)
)

# Convert probability to binary (random)
df["failure"] = (np.random.rand(n) < prob).astype(int)

df.to_csv("data/ai4i2020.csv", index=False)

print("✅ Dataset created!")