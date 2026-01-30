import os

# Mobaxterm으로 linux 서버 직접 접속해서 DIPLAY 환경변수 확인하고 그 내용을 직접 할당
os.environ["DISPLAY"]="localhost:10.0"
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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
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
        param_init: np.ndarray,
        param_low: np.ndarray,
        param_high: np.ndarray,
        max_steps: int = 30,
        action_scale: float = 0.05,  # step size as fraction of range
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert param_init.shape == param_low.shape == param_high.shape
        self.design = design
        self.spec = spec
        self.runner = runner

        self.param_init = param_init.astype(np.float64)
        self.param_low = param_low.astype(np.float64)
        self.param_high = param_high.astype(np.float64)

        self.max_steps = int(max_steps)
        self.action_scale = float(action_scale)

        n = self.param_init.size

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
            return -50.0, {"fail": 1.0}

        s11_b = 20*np.log10(abs(s11[idx_target_freq]))
        s22_b = 20*np.log10(abs(s22[idx_target_freq]))
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
        p_NF = np.mean(np.maximum(v_NF, 0.0))
        p_trade = float(p_s11 * p_NF)
        print(sim_output_data["freq"][idx_target_freq])

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
        }
        return float(reward), info


    # -----------------------------
    # ADS simulation run
    # -----------------------------
    def run_ads_sim(
        self,
        runner: AdsRunnerConfig,
        current_param : np.ndarray
    ) -> Dict[str, Any]:
        """
        schematic의 소자 값들을 변경 후 simulation 실행
        """

        try: 
            ELC19 = self.design.get_instance("ELC19")
            ELC19.parameters["W"].value = str(current_param[0])
            ELC19.parameters["L"].value = str(current_param[1])

            TL1 = self.design.get_instance("TL1")
            TL1.parameters["L"].value = str(current_param[2])

            netlist = self.design.generate_netlist()
            simulator = ads.CircuitSimulator()
            simulator.run_netlist(netlist, runner.ADS_sim_output_dir)

            return {"ok": True}
        except :
            return {"ok": False}
        

    def _norm_params(self, p: np.ndarray) -> np.ndarray:
        return (p - self.param_low) / (self.param_high - self.param_low + 1e-12)

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

        # Apply scaled update in real parameter space
        param_range = (self.param_high - self.param_low)
        delta = self.action_scale * param_range * np.clip(action, -1.0, 1.0)
        self.params = np.clip(self.params + delta, self.param_low, self.param_high)

        sim = self.run_ads_sim(self.runner, self.params)
            
        reward, info = self.compute_reward(self.spec, sim, self.runner)
        self.last_info = info

        self.step_idx += 1
        terminated = False
        truncated = self.step_idx >= self.max_steps

        # Optional: early stop when meets spec
        if info.get("meets", 0.0) > 0.5:
            terminated = True

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
        ADS_sim_output_dir="/home/jychung/ADS_project/test/"
    )

    # Example parameter vector (e.g., TL lengths, widths, bias resistors...)
    # (ELC19_W, ELC19_L, TL6_L)
    param_init = np.array([50e-6, 50e-6, 200e-6], dtype=np.float64) 
    param_low  = np.array([ 16e-6, 16e-6,  9e-6], dtype=np.float64)
    param_high = np.array([100e-6, 100e-6, 500e-6], dtype=np.float64)
    
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
            param_low=param_low,
            param_high=param_high,
            max_steps=25,
            action_scale=0.08,
            seed=1,
        )
        return Monitor(env)

    # PPO2-style: vectorized env (even if DummyVecEnv for now)
    vec_env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=256,           # rollout length
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./runs_sb3",
        device="cpu",
    )

    try :
        model.learn(total_timesteps=200_000)
        print("end 1 iter")
    except KeyboardInterrupt:
        model.save("ppo_ads_circuit")
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
 