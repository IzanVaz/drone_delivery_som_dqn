import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from env.drone_env import DroneDeliveryEnv
from som.som import SOM
from dqn.agent import DQNAgent
import traceback

def run_sweep_config(args):
    """
    Ejecuta un entrenamiento corto para una semilla y un tamaño de SOM específico.
    """
    seed_value, grid_n = args
    n_nodes = grid_n * grid_n  # Total de nodos en el SOM
    
    try:
        print(f"> Iniciando: SOM {grid_n}x{grid_n} | Semilla {seed_value}")
        env = DroneDeliveryEnv(max_steps=100)
        
        # Inicializamos SOM con el tamaño variable y el Agente adaptado a esos nodos
        som = SOM(grid_n=grid_n, input_dim=7, seed=seed_value)
        agent = DQNAgent(n_inputs=n_nodes, n_actions=6, is_discrete=True, epsilon_decay=15000)
        
        num_episodes = 3000 # Reducido para el barrido
        returns = []
        
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed_value + ep)
            done = False
            total_return = 0
            steps = 0
            
            # Congelamos antes por ser una prueba corta
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
            
        # Puntuación de convergencia: Media de los últimos 500 episodios
        final_score = np.mean(returns[-500:])
        print(f"✓ Completado: SOM {grid_n}x{grid_n} | Semilla {seed_value} -> Puntuación: {final_score:.1f}")
        
        return grid_n, seed_value, final_score

    except Exception as e:
        print(f"!!! ERROR en SOM {grid_n}x{grid_n} | Semilla {seed_value} !!!")
        traceback.print_exc()
        return grid_n, seed_value, None

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    print("==================================================")
    print(" INICIANDO BARRIDO DE PARÁMETROS: TAMAÑO DEL SOM ")
    print("==================================================")
    
    seeds = [42, 123, 7, 99, 1]
    grid_sizes = [10, 20, 30] # Tamaños a probar: 100, 400 y 900 nodos
    
    # Preparamos las combinaciones (5 semillas * 3 tamaños = 15 ejecuciones)
    tasks = [(s, g) for g in grid_sizes for s in seeds]
    
    results = {g: [] for g in grid_sizes}
    
    # Ejecución en paralelo
    with ProcessPoolExecutor() as executor:
        for grid_n, seed_value, score in executor.map(run_sweep_config, tasks):
            if score is not None:
                results[grid_n].append(score)
                
    print("\n==================================================")
    print(" BARRIDO COMPLETADO. GENERANDO GRÁFICA... ")
    print("==================================================")
    
    # --- Generación de la gráfica automática ---
    means = []
    stds = []
    labels = []
    
    for g in grid_sizes:
        scores = results[g]
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        means.append(mean_val)
        stds.append(std_val)
        labels.append(f"{g}x{g}\n({g*g} nodos)")
        print(f"SOM {g}x{g}: Media = {mean_val:.2f} ± {std_val:.2f}")

    plt.figure(figsize=(8, 6))
    
    # Gráfica de líneas con barras de error (desviación estándar)
    plt.errorbar(grid_sizes, means, yerr=stds, fmt='-o', color='purple', 
                 linewidth=2, capsize=5, capthick=2, markersize=8)
    
    plt.title("Efecto del Tamaño del SOM en la Convergencia", fontsize=14)
    plt.xlabel("Resolución de la Cuadrícula del SOM", fontsize=12)
    plt.ylabel("Retorno de Convergencia", fontsize=12)
    plt.xticks(grid_sizes, labels)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # Guardar como pide el boletín
    plt.savefig("figures/sweep_som_size.pdf")
    plt.savefig("figures/sweep_som_size.png")
    
    print("\n-> Gráfica guardada exitosamente en 'figures/sweep_som_size.pdf'")
    print("¡Listo para incluir en el informe LaTeX!")