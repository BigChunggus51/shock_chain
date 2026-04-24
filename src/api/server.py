import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_all_configs
from src.environment.supply_chain_env import SupplyChainEnv
from src.agents.heuristic_baselines import BaseStockPolicy, BeerGamePolicy
from stable_baselines3 import PPO

app = FastAPI(title="Supply Chain Digital Twin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
configs = load_all_configs("configs/default.yaml", "configs/shocks.yaml")
echelon_names = [e["name"] for e in configs["topology"]["echelons"]]

# Pre-load models
try:
    drl_model = PPO.load("outputs/models/final_model")
except Exception:
    drl_model = None

policies = {
    "drl": drl_model,
    "base_stock": BaseStockPolicy(echelon_names, configs["environment"]),
    "beer_game": BeerGamePolicy(echelon_names, configs["environment"])
}

class SimulationRequest(BaseModel):
    policy: str
    scenario: str

@app.get("/api/config")
def get_config():
    scenarios = ["stochastic"] + list(configs["shocks"].get("scenario_presets", {}).keys())
    return {
        "policies": list(policies.keys()),
        "scenarios": scenarios,
        "echelons": echelon_names,
    }

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    if req.policy not in policies:
        raise HTTPException(status_code=400, detail="Unknown policy")
        
    env = SupplyChainEnv(
        env_config=configs["environment"],
        reward_config=configs["reward"],
        topology_config=configs["topology"],
        shock_config=configs["shocks"],
        shock_mode="deterministic" if req.scenario != "stochastic" else "stochastic",
        scenario_preset=req.scenario if req.scenario != "stochastic" else None,
        seed=42
    )
    
    obs, info = env.reset()
    done = False
    
    while not done:
        if req.policy == "drl":
            if policies["drl"] is None:
                raise HTTPException(status_code=500, detail="DRL model not trained yet")
            action, _ = policies["drl"].predict(obs, deterministic=True)
        else:
            action = policies[req.policy].get_action(obs, info)
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    metrics = env.episode_metrics
    env.close()
    
    return {"status": "success", "metrics": metrics}

# Serve the static frontend
app.mount("/", StaticFiles(directory="public", html=True), name="public")
