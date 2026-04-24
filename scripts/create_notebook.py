import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

cells = [
    new_markdown_cell("# Supply Chain Shock Simulation — Interactive Dashboard"),
    new_markdown_cell("This notebook allows you to interactively test the trained DRL agent against "
                      "heuristic baselines under various disruption scenarios (demand surges, supplier outages, etc)."),
    
    new_code_cell("""import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from ipywidgets import interact, widgets

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

from src.utils.config_loader import load_all_configs
from src.environment.supply_chain_env import SupplyChainEnv
from src.agents.heuristic_baselines import BaseStockPolicy, BeerGamePolicy
from stable_baselines3 import PPO

# Set plot style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.figsize'] = (14, 6)"""),

    new_markdown_cell("### 1. Load Configurations and Models"),
    
    new_code_cell("""# Load configs
configs = load_all_configs('../configs/default.yaml', '../configs/shocks.yaml')
echelon_names = [e['name'] for e in configs['topology']['echelons']]

# Load the trained single-agent model
model_path = '../outputs/models/final_model'
print(f"Loading model from {model_path}...")
model = PPO.load(model_path)

# Initialize baselines
policies = {
    'DRL Agent (PPO)': None,  # Will use model.predict
    'Base-Stock': BaseStockPolicy(echelon_names, configs['environment']),
    'Beer Game': BeerGamePolicy(echelon_names, configs['environment'])
}
print("Policies loaded.")"""),

    new_markdown_cell("### 2. Interactive Simulation Engine"),
    
    new_code_cell("""def run_simulation(policy_name, scenario_preset):
    # Initialize environment with the selected scenario
    env = SupplyChainEnv(
        env_config=configs['environment'],
        reward_config=configs['reward'],
        topology_config=configs['topology'],
        shock_config=configs['shocks'],
        shock_mode='deterministic' if scenario_preset != 'stochastic' else 'stochastic',
        scenario_preset=scenario_preset if scenario_preset != 'stochastic' else None,
        seed=42
    )
    
    obs, info = env.reset()
    done = False
    
    # Run episode
    while not done:
        if policy_name == 'DRL Agent (PPO)':
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = policies[policy_name].get_action(obs, info)
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    metrics = env.episode_metrics
    env.close()
    return metrics

def plot_dashboard(metrics, policy_name, scenario):
    steps = [m['step'] for m in metrics]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Plot Retailer
    ax = axes[0]
    inv = [m['retailer/inventory'] for m in metrics]
    backlog = [m['retailer/backlog'] for m in metrics]
    ax.fill_between(steps, inv, alpha=0.3, color='#2196F3', label='Inventory')
    ax.plot(steps, inv, color='#2196F3')
    ax.fill_between(steps, [-b for b in backlog], alpha=0.3, color='#F44336', label='Backlog')
    ax.plot(steps, [-b for b in backlog], color='#F44336')
    ax.set_title(f"Retailer - {policy_name} ({scenario})", fontweight='bold')
    ax.legend(loc='upper left')
    
    # Plot Order Actions
    ax = axes[1]
    for name in echelon_names:
        orders = [m[f'{name}/order_qty'] for m in metrics]
        ax.plot(steps, orders, label=f"{name.capitalize()} Orders", alpha=0.8)
    ax.set_title("Order Quantities (Bullwhip Effect)", fontweight='bold')
    ax.legend(loc='upper left')
    
    # Plot Reward
    ax = axes[2]
    rewards = [m['reward'] for m in metrics]
    ax.plot(steps, rewards, color='purple', label='Step Reward')
    ax.set_title("Reward Signal", fontweight='bold')
    ax.set_xlabel("Decision Epoch (Days)")
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()"""),

    new_markdown_cell("### 3. Run What-If Scenarios\n"
                      "Use the dropdowns below to select a policy and a shock scenario. "
                      "The DRL agent is trained to handle unexpected shocks, while baselines will likely fail catastrophically."),
    
    new_code_cell("""# Get available scenarios from config
available_scenarios = ['stochastic'] + list(configs['shocks'].get('scenario_presets', {}).keys())

@interact(
    policy=widgets.Dropdown(options=list(policies.keys()), value='DRL Agent (PPO)', description='Policy:'),
    scenario=widgets.Dropdown(options=available_scenarios, value='cascading_crisis', description='Scenario:')
)
def interactive_dashboard(policy, scenario):
    metrics = run_simulation(policy, scenario)
    plot_dashboard(metrics, policy, scenario)
    
    total_reward = sum(m['reward'] for m in metrics)
    stockout_days = sum(1 for m in metrics if m['retailer/backlog'] > 0)
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Retailer Stockout Days: {stockout_days}/365")""")
]

nb['cells'] = cells
with open('notebooks/03_what_if_scenarios.ipynb', 'w') as f:
    nbformat.write(nb, f)
