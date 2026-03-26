import sys
print(sys.executable)
import numpy as np
import pandas as pd

data = np.load('saved_runs/20260223_121314_c588cd11/paths.npz')
print(data.files)

print(data["returns_pred"].shape)

returns_df = pd.DataFrame(data["returns_pred"])
signal_df = pd.DataFrame(data["signal_pred"])
actual_df = pd.DataFrame(data["signal_pred_actual"])

# Add prefixes so columns are distinguishable
returns_df = returns_df.add_prefix("returns_")
signal_df = signal_df.add_prefix("signal_")
actual_df = actual_df.add_prefix("actual_")

df = pd.concat([returns_df, signal_df, actual_df], axis=1)

df.to_csv("combined_predictions.csv", index=False)