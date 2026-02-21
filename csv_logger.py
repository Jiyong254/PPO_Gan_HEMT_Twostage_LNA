from pathlib import Path
import pandas as pd
import numpy as np

class CSVLogger:

    def __init__(self, target_spec_out_dir=str, device_param_out_dir=str):

        self.target_spec_out_dir = Path(target_spec_out_dir)
        self.target_spec_out_dir.mkdir(parents=True, exist_ok=True)

        self.device_param_out_dir = Path(device_param_out_dir)
        self.device_param_out_dir.mkdir(parents=True, exist_ok=True)

        self.files = {
            "s11": self.target_spec_out_dir / "S11.csv",
            "s22": self.target_spec_out_dir / "S22.csv",
            "s21": self.target_spec_out_dir / "s21.csv",
            "NF":  self.target_spec_out_dir / "NF.csv",
            "K": self.target_spec_out_dir / "K.csv",
            "general_TL": self.device_param_out_dir / "general_TL.csv",
            "bias_TL" : self.device_param_out_dir / "bias_TL.csv",
            "ELC" : self.device_param_out_dir / "ELC.csv"
        }

        self.episode_idx = 0

        for f in self.files.values():
            if not f.exists():
                pd.DataFrame().to_csv(f, index=False)
            else:
                f.write_text("")

    def append(self, name: str, spec: np, freq: np, data_type: str):

        path = self.files[name]
        
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
            
        if(data_type=="target_spec"):
            if(self.episode_idx == 0):
                df["freq"] = freq
                df[f'{self.episode_idx}'] = spec
            else:
                df[f'{self.episode_idx}'] = spec
        elif(data_type=="device"):
            df[f'{self.episode_idx}'] = spec

        df.to_csv(
            path, index=False
        )

    def next_episode(self):
        self.episode_idx += 1
