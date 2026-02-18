import os

# Mobaxterm으로 linux 서버 직접 접속해서 DIPLAY 환경변수 확인하고 그 내용을 직접 할당
os.environ["DISPLAY"]="localhost:13.0"
print(os.environ.get("DISPLAY"))

"""
PPO2-style circuit optimization template
- PPO2 naming comes from Stable-Baselines(TF) era; in SB3(Pytorch) use PPO.
- Env.step() calls ADS in a separate process (safe/robust for ADS + Jupyter issues).
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Dict, Tuple, Any, Optional

import gymnasium as gym
from gymnasium import spaces
import tensorboard
from torch.utils.tensorboard import SummaryWriter

from keysight.ads import de
from keysight.ads.de import db_uu as db
from keysight.edatoolbox import circuit
from keysight.edatoolbox import ads
import keysight.ads.dataset as dataset

from pathlib import Path
import pandas as pd
import numpy as np

from csv_logger import SparamCSVLogger

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# -----------------------------
# Config
# -----------------------------
@dataclass
class CircuitSpec:
    # Example targets (edit to your spec)
    s11_db_target: float = -10.0     # want <= -10 dB
    s22_db_target: float = -10.0
    NF_db_target: float = 10.0     # want >= 10 dB (|S21| in dB)
    # Frequency band for evaluation
    f_min_ghz: float = 3.0
    f_max_ghz: float = 5.0

    # Reward weights
    w_s11: float = 1.0
    w_s22: float = 1.0
    w_NF: float = 0.5
    w_penalty: float = 0.2
    w_trade: float = 0.5


@dataclass
class AdsRunnerConfig:
    ADS_sim_output_dir: str         # ADS simulation output directory





# -----------------------------
# Gymnasium environment
# -----------------------------
class AdsCircuitEnv(gym.Env):
    """
    Action: continuous vector of normalized parameter updates (or absolute params)
    Observation: current normalized params + last reward stats (simple baseline)
    Episode: fixed number of steps; can be 1-step bandit style too.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        design: db.Design,
        spec: CircuitSpec,
        runner: AdsRunnerConfig,
        csv_logger: SparamCSVLogger,
        param_init: Dict[str, np.ndarray],
        param_low: Dict[str, np.ndarray],
        param_high: Dict[str, np.ndarray],
        max_steps: int = 30,
        action_scale: float = 0.05,  # step size as fraction of range
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert self._flatten(param_init).shape == self._flatten(param_low).shape == self._flatten(param_high).shape
        
        self.design = design
        self.spec = spec
        self.runner = runner
        self.csv_logger = csv_logger

        self.param_init = param_init
        self.param_low = param_low
        self.param_high = param_high

        self.max_steps = int(max_steps)
        self.action_scale = float(action_scale)

        n = self.param_init["general_TL"].size + self.param_init["bias_TL"].size + self.param_init["ELC"].size

        # Action: [-1, 1]^n
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

        # Observation: normalized params (0..1) + last penalties (3) + step fraction (1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n + 3 + 1,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self.params = self.param_init.copy()
        self.step_idx = 0
        self.last_info = {"p_s11": 0.0, "p_s22": 0.0, "p_NF": 0.0}
        self.last_params = None
        self.best_params = None
        self.best_reward = -np.inf

    def softplus(self, x: np.ndarray) -> np.ndarray:
        # stable softplus
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


    def band_mask(self, freq_ghz: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
        return (freq_ghz >= fmin) & (freq_ghz <= fmax)


    def compute_reward(self, spec: CircuitSpec, sim: Dict[str, Any], runner: AdsRunnerConfig) -> Tuple[float, Dict[str, float]]:
        """
        ADS simulation 결과 불러와서 target 값과 비교 후 reward 계산
        """
        if not sim.get("ok", True):
            return -100.0, {"fail": 1.0}

        # ADS simulation 결과 추출
        ads_sim_output = dataset.open(runner.ADS_sim_output_dir + "sample_gan_25GHz_twostage_ppo.ds")
        sim_output_data = ads_sim_output["SP1.SP"].to_dataframe().reset_index()

        f = np.asarray(sim_output_data["freq"], dtype=float)
        s11 = np.asarray(sim_output_data["S[1,1]"], dtype=complex)
        s22 = np.asarray(sim_output_data["S[2,2]"], dtype=complex)
        NF = np.asarray(sim_output_data["nf[2]"], dtype=float)

        # 관심 있는 주파수 영역만 남긴다
        idx_target_freq = np.where((f >= spec.f_min_ghz) & (f <= spec.f_max_ghz))[0]

        if not np.any(idx_target_freq):
            print("no target freq")
            return -50.0, {"fail": 1.0}

        s11_b = 20*np.log10(np.abs(s11[idx_target_freq]))
        s22_b = 20*np.log10(np.abs(s22[idx_target_freq]))
        NF_b = NF[idx_target_freq]

        # Convert targets to "violations"
        # For S11/S22: want <= target (more negative is better)
        v_s11 = (s11_b - spec.s11_db_target)  # <= 0 good
        v_s22 = (s22_b - spec.s22_db_target)

        # For gain: want >= target
        v_NF = (NF_b - spec.NF_db_target)  # <= 0 good

        # penalties: only when violation > 0
        p_s11 = np.mean(np.maximum(v_s11, 0.0))
        p_s22 = np.mean(np.maximum(v_s22, 0.0))
        p_NF = np.mean(np.maximum(v_NF, 0))
        p_trade = float(p_s11 * p_NF)

        # reward as negative penalties (you can redesign this)
        reward = (
            -spec.w_s11 * p_s11
            -spec.w_s22 * p_s22
            -spec.w_NF * p_NF
            -spec.w_trade * p_trade
        )


        # small bonus if fully meets all specs in-band
        meets = (np.max(v_s11) <= 0) and (np.max(v_s22) <= 0) and (np.max(v_NF) <= 0)
        if meets:
            reward += 5.0

        #
        info = {
            "p_s11": float(p_s11),
            "p_s22": float(p_s22),
            "p_NF": float(p_NF),
            "meets": float(meets),
            "reward": float(reward),
            "s11" : 20*np.log10(np.abs(s11)),
            "s22" : 20*np.log10(np.abs(s22)),
            "NF" : NF,
            "f" : f
        }

        
        print("reward computation done")
        print(reward)
        
        return float(reward), info


    # -----------------------------
    # ADS simulation run
    # -----------------------------
    def run_ads_sim(
        self,
        runner: AdsRunnerConfig,
        current_param : Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        schematic의 소자 값들을 변경 후 simulation 실행
        """
    # param_key = np.array(["ELC19_W", "ELC19_L", "TL5_L", "TL4_L", "ELC5_W", "ELC5_L", "TL1_L", "TL9_L", "TL10_L", "TL11_L", "ELC15_W", "ELC15_L", "ELC20_W", "ELC20_L", "TL13_L", "TL21_L", "ELC25_W", "ELC25_L", "TL14_L", "TL18_L", "TL15_L", "ELC22_W", "ELC22_L", "TL19_L", "ELC23_W", "ELC23_L", "TL17_L"])

        try: 
            TL1 = self.design.get_instance("TL1")
            TL1.parameters["L"].value = str(current_param["general_TL"][0])

            TL2 = self.design.get_instance("TL2")
            TL2.parameters["L"].value = str(current_param["general_TL"][1])

            TL3 = self.design.get_instance("TL3")
            TL3.parameters["L"].value = str(current_param["general_TL"][2])

            TL4 = self.design.get_instance("TL4")
            TL4.parameters["L"].value = str(current_param["general_TL"][3])

            TL5 = self.design.get_instance("TL5")
            TL5.parameters["L"].value = str(current_param["general_TL"][4])

            TL6 = self.design.get_instance("TL6")
            TL6.parameters["L"].value = str(current_param["general_TL"][5])

            TL7 = self.design.get_instance("TL7")
            TL7.parameters["L"].value = str(current_param["general_TL"][6])

            TL8 = self.design.get_instance("TL8")
            TL8.parameters["L"].value = str(current_param["general_TL"][7])

            TL9 = self.design.get_instance("TL9")
            TL9.parameters["L"].value = str(current_param["general_TL"][8])

            TL10 = self.design.get_instance("TL10")
            TL10.parameters["L"].value = str(current_param["general_TL"][9])

            TL11 = self.design.get_instance("TL11")
            TL11.parameters["L"].value = str(current_param["general_TL"][10])

            TL12 = self.design.get_instance("TL12")
            TL12.parameters["L"].value = str(current_param["bias_TL"][0])

            TL13 = self.design.get_instance("TL13")
            TL13.parameters["L"].value = str(current_param["bias_TL"][1])

            TL14 = self.design.get_instance("TL14")
            TL14.parameters["L"].value = str(current_param["bias_TL"][2])

            TL15 = self.design.get_instance("TL15")
            TL15.parameters["L"].value = str(current_param["bias_TL"][3])

            ELC1 = self.design.get_instance("ELC1")
            ELC1.parameters["W"].value = str(current_param["ELC"][0])
            ELC1.parameters["L"].value = str(current_param["ELC"][1])

            ELC2 = self.design.get_instance("ELC2")
            ELC2.parameters["W"].value = str(current_param["ELC"][2])
            ELC2.parameters["L"].value = str(current_param["ELC"][3])

            ELC3 = self.design.get_instance("ELC3")
            ELC3.parameters["W"].value = str(current_param["ELC"][4])
            ELC3.parameters["L"].value = str(current_param["ELC"][5])

            ELC4 = self.design.get_instance("ELC4")
            ELC4.parameters["W"].value = str(current_param["ELC"][6])
            ELC4.parameters["L"].value = str(current_param["ELC"][7])

            ELC5 = self.design.get_instance("ELC5")
            ELC5.parameters["W"].value = str(current_param["ELC"][8])
            ELC5.parameters["L"].value = str(current_param["ELC"][9])

            ELC6 = self.design.get_instance("ELC6")
            ELC6.parameters["W"].value = str(current_param["ELC"][10])
            ELC6.parameters["L"].value = str(current_param["ELC"][11])


            ELC7 = self.design.get_instance("ELC7")
            ELC7.parameters["W"].value = str(current_param["ELC"][12])
            ELC7.parameters["L"].value = str(current_param["ELC"][13])

            netlist = self.design.generate_netlist()

            simulator = ads.CircuitSimulator()
            simulator.run_netlist(netlist, runner.ADS_sim_output_dir)

            print(current_param["ELC"][12])

            return {"ok": True}
        except Exception as e:
            print("simulation fail:", repr(e))
            traceback.print_exc()
            return {"ok": False}
        
    def _flatten(self, param: Dict[str, np.ndarray]):
        return np.concatenate([param["general_TL"], param["bias_TL"], param["ELC"]])
    
    def _norm_params(self, p: Dict[str, np.ndarray]) -> np.ndarray:
        params = self._flatten(p)
        low = self._flatten(self.param_low)
        high = self._flatten(self.param_high)

        return (params - low) / (high - low + 1e-12)

    def _get_obs(self) -> np.ndarray:
        nrm = self._norm_params(self.params)
        obs = np.concatenate([
            nrm,
            np.array([self.last_info.get("p_s11", 0.0),
                      self.last_info.get("p_s22", 0.0),
                      self.last_info.get("p_NF", 0.0)], dtype=np.float64),
            np.array([self.step_idx / max(1, self.max_steps)], dtype=np.float64),
        ])
        return obs.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset params (optionally randomize around init for exploration)
        self.params = self.param_init.copy()
        self.step_idx = 0
        self.last_info = {"p_s11": 0.0, "p_s22": 0.0, "p_NF": 0.0}

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)

        idx_general_TL = self.param_init["general_TL"].size
        idx_bias_TL = self.param_init["bias_TL"].size
        idx_ELC = self.param_init["ELC"].size

        action_clip = np.clip(action, -1.0, 1.0)
        action_general_TL = action_clip[0:idx_general_TL]
        action_bias_TL = action_clip[idx_general_TL:idx_general_TL+idx_bias_TL]
        action_ELC = action_clip[idx_general_TL+idx_bias_TL: idx_general_TL+idx_bias_TL+idx_ELC]

        # Apply scaled update in real parameter space
        param_range = {"general_TL" : self.param_high["general_TL"] - self.param_low["general_TL"],
                       "bias_TL" : self.param_high["bias_TL"] - self.param_low["bias_TL"],
                       "ELC" : self.param_high["ELC"] - self.param_low["ELC"]
                       }
        
        # delta = self.action_scale * param_range * np.clip(action, -1.0, 1.0)
        delta = {"general_TL" : self.action_scale * param_range["general_TL"] * action_general_TL,
                 "bias_TL" : self.action_scale * param_range["bias_TL"] * action_bias_TL,
                 "ELC" : self.action_scale * param_range["ELC"] * action_ELC
                 }
        
        self.params["general_TL"] = np.clip(self.params["general_TL"] + delta["general_TL"], self.param_low["general_TL"], self.param_high["general_TL"])
        self.params["bias_TL"] = np.clip(self.params["bias_TL"] + delta["bias_TL"], self.param_low["bias_TL"], self.param_high["bias_TL"])
        self.params["ELC"] = np.clip(self.params["ELC"] + delta["ELC"], self.param_low["ELC"], self.param_high["ELC"])    
        
        sim = self.run_ads_sim(self.runner, self.params)
            
        reward, info = self.compute_reward(self.spec, sim, self.runner)
        self.last_info = info

        # 최고의 파라미터들을 저장 ==> 학습 후에 값들을 확인하기 위해

        self.last_params = {"general_TL": self.params["general_TL"].copy(),
                            "bias_TL": self.params["bias_TL"].copy(),
                            "ELC": self.params["ELC"].copy()
                            }
        
        if reward > self.best_reward:
            self.best_reward = float(reward)
            self.best_params = {
                "general_TL": self.params["general_TL"].copy(),
                "bias_TL": self.params["bias_TL"].copy(),
                "ELC": self.params["ELC"].copy(),
            }

        self.step_idx += 1
        terminated = False
        truncated = self.step_idx >= self.max_steps

        # Optional: early stop when meets spec
        if info.get("meets", 0.0) > 0.5:
            terminated = True            

        if (terminated or truncated):
            s11 = info.get("s11")
            s22 = info.get("s22")
            NF = info.get("NF")
            f = info.get("f")

            self.csv_logger.append("s11", s11, f)
            self.csv_logger.append("s22", s22, f)
            self.csv_logger.append("NF", NF, f)
            
            self.csv_logger.next_episode()
            print(self.csv_logger.episode_idx)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

# -----------------------------
# Train script
# -----------------------------
def main():
    # ---- TODO: set these to your actual values ----
    spec = CircuitSpec(
        s11_db_target=-10.0,
        s22_db_target=-10.0,
        NF_db_target=2.0,
        f_min_ghz=12.0e9,
        f_max_ghz=18.0e9,
    )

    runner = AdsRunnerConfig(
        ADS_sim_output_dir="/home/jychung/ADS_project/test/sim_logfile/"
    )

    csv_logger = SparamCSVLogger(
        out_dir="/home/jychung/python/target_spec/reward_mean_calc"
    )

    # (ELC19_W, ELC19_L, TL6_L)
    # param_key = np.array(["ELC19_W", "ELC19_L", "TL5_L", "TL4_L", "ELC5_W", "ELC5_L", "TL1_L", "TL9_L", "TL10_L", "TL11_L", "ELC15_W", "ELC15_L", "ELC20_W", "ELC20_L", "TL13_L", "TL21_L", "ELC25_W", "ELC25_L", "TL14_L", "TL18_L", "TL15_L", "ELC22_W", "ELC22_L", "TL19_L", "ELC23_W", "ELC23_L", "TL17_L"])
    
    # param_init = np.array([50e-6, 50e-6, 200e-6, 1000e-6, 50e-6, 50e-6, 200e-6, 200e-6, 200e-6, 1000e-6, 50e-6, 50e-6, 50e-6, 50e-6, 200e-6, 1000e-6, 50e-6, 50e-6, 200e-6, 1000e-6, 50e-6, 50e-6, 200e-6, 50e-6, 50e-6, 200e-6], dtype=np.float64) 
    # param_low  = np.array([ 16e-6, 16e-6,  9e-6, 700e-6, 16e-6, 16e-6, 9e-6, 9e-6, 9e-6, 700e-6, 16e-6, 16e-6, 16e-6, 16e-6, 9e-6, 700e-6, 16e-6, 16e-6, 9e-6, 700e-6, 16e-6, 16e-6, 9e-6, 16e-6, 16e-6, 9e-6], dtype=np.float64)
    # param_high = np.array([100e-6, 100e-6, 500e-6, 1800e-6, 100e-6, 100e-6, 500e-6, 500e-6, 500e-6, 1800e-6, 100e-6, 100e-6, 100e-6, 100e-6, 500e-6, 1800e-6, 100e-6, 100e-6, 500e-6, 1800e-6, 100e-6, 100e-6, 500e-6, 100e-6, 100e-6, 500e-6], dtype=np.float64)
    
    param_init = {"general_TL" : np.array([200e-6, 200e-6, 200e-6, 200e-6, 200e-6, 200e-6, 200e-6, 200e-6, 200e-6, 200e-6, 200e-6], dtype=np.float64),
                  "bias_TL" : np.array([1000e-6, 1000e-6, 1000e-6, 1000e-6], dtype=np.float64),
                  "ELC" : np.array([50e-6, 50e-6, 50e-6, 50e-6, 50e-6, 50e-6, 50e-6, 50e-6, 50e-6, 50e-6, 50e-6, 50e-6, 50e-6, 50e-6], dtype=np.float64)
                  }
    
    param_low = {"general_TL" : np.array([9e-6, 9e-6, 9e-6, 9e-6, 9e-6, 9e-6, 9e-6, 9e-6, 9e-6, 9e-6, 9e-6], dtype=np.float64),
                 "bias_TL" : np.array([700e-6, 700e-6, 700e-6, 700e-6], dtype=np.float64),
                 "ELC" : np.array([16e-6, 16e-6, 16e-6, 16e-6, 16e-6, 16e-6, 16e-6, 16e-6, 16e-6, 16e-6, 16e-6, 16e-6, 16e-6, 16e-6], dtype=np.float64)
                 }
    param_high = {"general_TL" : np.array([500e-6, 500e-6, 500e-6, 500e-6, 500e-6, 500e-6, 500e-6, 500e-6, 500e-6, 500e-6, 500e-6], dtype=np.float64),
                  "bias_TL" : np.array([1800e-6, 1800e-6, 1800e-6, 1800e-6], dtype=np.float64),
                  "ELC" : np.array([100e-6, 100e-6, 100e-6, 100e-6, 100e-6, 100e-6, 100e-6, 100e-6, 100e-6, 100e-6, 100e-6, 100e-6, 100e-6, 100e-6], dtype=np.float64)
                  }
    
    

    try :
        cell_name = "sample_gan_25GHz_twostage_ppo"

        workspace_path = "/home/jychung/ADS_project/test"
        workspace = de.open_workspace(workspace_path)

        library_name = "tutorial1_lib"
        library_path = workspace.path / library_name
        design = db.open_design(f"tutorial1_lib:{cell_name}:schematic", db.DesignMode.APPEND)
    except :
        workspace.close()


    def make_env():
        env = AdsCircuitEnv(
            design = design,
            spec=spec,
            runner=runner,
            param_init=param_init,
            csv_logger=csv_logger,
            param_low=param_low,
            param_high=param_high,
            max_steps=25,
            action_scale=0.08,
            seed=1,
        )
        return Monitor(env)

    # PPO2-style: vectorized env (even if DummyVecEnv for now)
    vec_env = DummyVecEnv([make_env])

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=5.0,
        clip_reward=5.0,
        gamma=0.99
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=265,           # rollout length
        batch_size=265,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="/home/jychung/python/runs_25GHz_twostage_PPO",
        device="cpu",
    )

    try :
        model.learn(total_timesteps=100_00)

        env0 = vec_env.envs[0]
        base_env = env0.unwrapped

        best_params = base_env.best_params

        np.savez("/home/jychung/python/best_reward_parameter/final_params2.npz",
                 general_TL=best_params["general_TL"],
                 bias_TL=best_params["bias_TL"],
                 ELC=best_params["ELC"])
        
        model.save("/home/jychung/python/model_saved/ppo_ads_circuit2")
        
        print("end 1 iter")
    except KeyboardInterrupt:
        model.save("/home/jychung/python/model_saved/ppo_ads_circuit2")
        if (workspace.close()):
            print("workspace is closed successfully")


    # Test a few episodes
    obs = vec_env.reset()
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print("reward:", reward, "info:", info)

if __name__ == "__main__":
    
    main()
 