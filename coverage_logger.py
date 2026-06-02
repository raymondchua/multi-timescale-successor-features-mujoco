import pandas as pd
import os
from collections import defaultdict


class CoverageLoggerCSV:
    def __init__(self, output_dir="coverage_snapshots", bin_size=0.25, enable_binning=True):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.bin_size = bin_size
        self.enable_binning = enable_binning
        self.counter = defaultdict(lambda: [0.0, 0])  # (x_bin, y_bin) → [reward_sum, count]

    def _bin_value(self, value):
        return int(value / self.bin_size) if self.enable_binning else value

    def log(self, x, y, reward):
        x_key = self._bin_value(x)
        y_key = self._bin_value(y)

        key = (x_key, y_key)
        self.counter[key][0] += reward if reward is not None else 0.0
        self.counter[key][1] += 1

    def save(self, train_step):
        if not self.counter:
            return

        rows = []
        for (x_key, y_key), (reward_sum, count) in self.counter.items():
            rows.append({
                "x_bin": x_key,
                "y_bin": y_key,
                "reward_sum": reward_sum,
                "visit_count": count,
                "train_step": train_step,
            })

        df = pd.DataFrame(rows)
        filename = f"coverage_step_{train_step}.csv"
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        self.counter.clear()

    def reset(self):
        self.counter.clear()
