import os
import numpy as np
import torch
import random
from env.drone_env import DroneDeliveryEnv
from som.som import SOM
from dqn.agent import DQNAgent
import time

def train_hybrid(seed_value, num_episodes=10000):
    # 1. Fijar todas las semillas
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    # 2. Inicializar el entorno
    env = DroneDeliveryEnv(render_mode=None)
    obs_dim = env.observation_space.shape[0]  
    action_dim = env.action_space.n           
    
    # 3. Inicializar el SOM (Codificador de Estado)
    grid_size = 20
    n_nodes = grid_size * grid_size
    som = SOM(grid_n=grid_size, input_dim=obs_dim, seed=seed_value)
    
    # 4. Inicializar el Agente DQN
    # AJUSTE MATEMÁTICO: Bajamos a 80,000 pasos para que el Epsilon baje
    # a 0.05 hacia la mitad del entrenamiento.
    agent = DQNAgent(
        n_inputs=n_nodes,
        n_actions=action_dim,
        is_discrete=True,
        epsilon_decay=80_000
    )
    
    history = {
        "returns": [],
        "lengths": [],
        "som_errors": []
    }

    print(f"\n[{seed_value}] Iniciando entrenamiento de {num_episodes} episodios...")

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed_value + ep)
        done = False
        total_return = 0
        steps = 0
        
        while not done:
            # A) SOM
            state_idx = som.update(obs)
            
            # B) Agente DQN
            action = agent.select_action(state_idx, training=True)
            
            # C) Entorno
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # D) Clasificación siguiente estado
            next_state_idx = som.predict(next_obs)
            
            # E) Entrenamiento
            agent.store(state_idx, action, reward, next_state_idx, done)
            agent.update()
            
            obs = next_obs
            total_return += reward
            steps += 1
            
        history["returns"].append(total_return)
        history["lengths"].append(steps)
        
        # Log cada 500 episodios
        if (ep + 1) % 500 == 0:
            avg_return = np.mean(history["returns"][-500:])
            eps = agent.epsilon()
            print(f"[{seed_value}] Episodio {ep + 1:5d}/{num_episodes} | Recompensa media (últ. 500): {avg_return:7.1f} | Epsilon: {eps:.2f}")

    return history, som, agent

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(" INICIANDO ENTRENAMIENTO HÍBRIDO SOM+DQN (10x10)")
    print(f" DISPOSITIVO DE COMPUTO: {device.type.upper()}")
    if device.type == 'cuda':
        print(f" GPU DETECTADA: {torch.cuda.get_device_name(0)}")
    print("==================================================")
    
    seeds = [42, 123, 7, 99, 1]
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    for s in seeds:
        inicio = time.time() # Empezamos a cronometrar
        hist, final_som, final_agent = train_hybrid(seed_value=s, num_episodes=10000)
        
        npz_path = os.path.join(results_dir, f"raw_data_seed_{s}.npz")
        model_path = os.path.join(results_dir, f"dqn_seed_{s}.pth")
        som_path = os.path.join(results_dir, f"som_seed_{s}.npz")
        
        np.savez(npz_path, returns=hist["returns"], lengths=hist["lengths"])
        final_agent.save(model_path)
        final_som.save(som_path)
        
        print(f"[{s}] Entrenamiento completado. Datos guardados en '{results_dir}/'")
        
            
        fin = time.time() # Paramos el cronómetro
        
        tiempo_total = fin - inicio
        minutos = int(tiempo_total // 60)
        segundos = int(tiempo_total % 60)
        print(f"\nTiempo total de ejecución de la semilla {s}: {minutos} minutos y {segundos} segundos.")

    print("\n¡Todos los entrenamientos han finalizado con éxito!")