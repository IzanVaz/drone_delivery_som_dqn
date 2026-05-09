import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from env.drone_env import DroneDeliveryEnv
from som.som import SOM
from dqn.agent import DQNAgent
import traceback

def run_experiment(args):
    """
    Ejecuta un entrenamiento según el tipo de barrido (LR del SOM o Epsilon del DQN).
    Fijamos el tamaño del SOM en 20x20 (400 nodos) porque descubrimos que es el óptimo.
    """
    sweep_type, param_val, seed_value = args
    
    try:
        print(f"> Iniciando: {sweep_type.upper()}={param_val} | Semilla {seed_value}")
        env = DroneDeliveryEnv(max_steps=100)
        
        # Configuraciones por defecto (las mejores que tenemos)
        grid_n = 20
        n_nodes = 400
        lr_som = 0.5
        eps_decay = 15000
        
        # Sobrescribimos el parámetro que estamos barriendo
        if sweep_type == "lr":
            lr_som = param_val
        elif sweep_type == "eps":
            eps_decay = param_val
            
        som = SOM(grid_n=grid_n, input_dim=7, lr_init=lr_som, seed=seed_value)
        agent = DQNAgent(n_inputs=n_nodes, n_actions=6, is_discrete=True, epsilon_decay=eps_decay)
        
        num_episodes = 3000 # Entrenamiento corto para el barrido
        returns = []
        
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed_value + ep)
            done = False
            total_return = 0
            steps = 0
            
            freeze_som = ep > 1000 
            
            while not done:
                if not freeze_som:
                    state_idx = som.update(obs)
                else:
                    state_idx = som.predict(obs)
                
                action = agent.select_action(state_idx, training=True)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                
                next_state_idx = som.predict(next_obs)
                agent.store(state_idx, action, reward, next_state_idx, done)
                
                if steps % 2 == 0:
                    agent.update()
                
                obs = next_obs
                total_return += reward
                steps += 1
                
            returns.append(total_return)
            
        # Puntuación de convergencia
        final_score = np.mean(returns[-500:])
        print(f"✓ Fin {sweep_type.upper()}={param_val} | Semilla {seed_value} -> {final_score:.1f}")
        
        return sweep_type, param_val, seed_value, final_score

    except Exception as e:
        print(f"!!! ERROR en {sweep_type.upper()}={param_val} | Semilla {seed_value} !!!")
        traceback.print_exc()
        return sweep_type, param_val, seed_value, None

def plot_sweep(results_dict, values, title, xlabel, filename):
    """Función auxiliar para dibujar y guardar las gráficas"""
    means = []
    stds = []
    labels = [str(v) for v in values]
    
    for v in values:
        scores = results_dict[v]
        means.append(np.mean(scores))
        stds.append(np.std(scores))

    plt.figure(figsize=(8, 6))
    plt.errorbar(range(len(values)), means, yerr=stds, fmt='-o', color='teal', 
                 linewidth=2, capsize=5, capthick=2, markersize=8)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Retorno de Convergencia", fontsize=12)
    plt.xticks(range(len(values)), labels)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f"figures/{filename}.pdf")
    plt.savefig(f"figures/{filename}.png")
    print(f"-> Gráfica guardada en 'figures/{filename}.pdf'")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    seeds = [42, 123, 7, 99, 1]
    
    # Definimos los valores a probar
    lr_values = [0.1, 0.5, 0.9]
    eps_values = [5000, 15000, 30000]
    
    # Preparamos las tareas
    tasks_lr = [("lr", val, s) for val in lr_values for s in seeds]
    tasks_eps = [("eps", val, s) for val in eps_values for s in seeds]
    all_tasks = tasks_lr + tasks_eps
    
    results_lr = {v: [] for v in lr_values}
    results_eps = {v: [] for v in eps_values}
    
    print("==================================================")
    print(" INICIANDO BARRIDOS FINALES (SOM LR y DQN EPSILON)")
    print("==================================================")
    
    with ProcessPoolExecutor() as executor:
        for s_type, p_val, s_val, score in executor.map(run_experiment, all_tasks):
            if score is not None:
                if s_type == "lr":
                    results_lr[p_val].append(score)
                elif s_type == "eps":
                    results_eps[p_val].append(score)

    print("\n==================================================")
    print(" BARRIDOS COMPLETADOS. GENERANDO GRÁFICAS... ")
    print("==================================================")
    
    plot_sweep(results_lr, lr_values, 
               "Efecto de la Tasa de Aprendizaje del SOM (lr_init)", 
               "Learning Rate Inicial (SOM)", 
               "sweep_som_lr")
               
    plot_sweep(results_eps, eps_values, 
               "Efecto del Decaimiento de Epsilon (DQN)", 
               "Epsilon Decay (Pasos)", 
               "sweep_dqn_epsilon")
               
    print("\n¡Todo listo! Tienes el 100% de los datos experimentales para el informe.")
    