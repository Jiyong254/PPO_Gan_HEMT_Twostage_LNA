from pathlib import Path
import pandas as pd
import numpy as np

class SparamCSVLogger:

    def __init__(self, out_dir):

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.files = {
            "s11": self.out_dir / "S11.csv",
            "s22": self.out_dir / "S22.csv",
            "NF":  self.out_dir / "NF.csv",
        }

        self.episode_idx = 0

        for f in self.files.values():
            if not f.exists():
                pd.DataFrame().to_csv(f, index=False)
            else:
                f.write_text("")

    def append(self, name: str, spec: np, freq: np):

        path = self.files[name]
        
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
            

        if(self.episode_idx == 0):
            df["freq"] = freq
            df[f'{self.episode_idx}'] = spec
        else:
            df[f'{self.episode_idx}'] = spec

        df.to_csv(
            path, index=False
        )

    def next_episode(self):
        self.episode_idx += 1
