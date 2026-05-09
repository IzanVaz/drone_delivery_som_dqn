import os
import numpy as np
import torch
import random
from concurrent.futures import ProcessPoolExecutor
from env.drone_env import DroneDeliveryEnv
from som.som import SOM
from dqn.agent import DQNAgent
import traceback
import time

def run_seed(seed_value):
    try:
        print(f"\n>>> [Semilla {seed_value}] Iniciando entrenamiento paralelo...")
        env = DroneDeliveryEnv(max_steps=100)
        
        # Inicializamos SOM y Agente
        som = SOM(grid_n=20, input_dim=7, seed=seed_value)
        agent = DQNAgent(n_inputs=400, n_actions=6, is_discrete=True, epsilon_decay=60000)
        
        num_episodes = 10000
        history = {"returns": [], "lengths": []}
        
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed_value + ep)
            done = False
            total_return = 0
            steps_in_ep = 0
            
            # Congelar SOM después de 3000 episodios
            freeze_som = ep > 3000
            
            while not done:
                if not freeze_som:
                    state_idx = som.update(obs)
                else:
                    state_idx = som.predict(obs)
                
                action = agent.select_action(state_idx, training=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_state_idx = som.predict(next_obs)
                agent.store(state_idx, action, reward, next_state_idx, done)
                
                if steps_in_ep % 2 == 0:
                    agent.update()
                
                obs = next_obs
                total_return += reward
                steps_in_ep += 1
                
            history["returns"].append(total_return)
            
            if (ep + 1) % 1000 == 0:
                avg_rew = np.mean(history["returns"][-1000:])
                eps = agent.epsilon()
                print(f"[{seed_value}] Ep {ep+1:5d} | Media últ. 1000: {avg_rew:8.1f} | Epsilon: {eps:.2f}")

        results_dir = "results"
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        
        np.savez(f"{results_dir}/raw_data_seed_{seed_value}.npz", returns=history["returns"])
        agent.save(f"{results_dir}/dqn_seed_{seed_value}.pth")
        som.save(f"{results_dir}/som_seed_{seed_value}.npz")
        
        print(f"!!! [Semilla {seed_value}] Entrenamiento completado y guardado.")
        return seed_value

    except Exception as e:
        # ¡EL CHIVATO! Si algo falla, lo imprimirá en rojo chillón
        print(f"\n!!! ERROR FATAL EN SEMILLA {seed_value} !!!")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
        
    print("==================================================")
    print(" INICIANDO ENTRENAMIENTO MULTI-NÚCLEO (PARALELO)")
    print("==================================================")
    
    seeds = [42, 123, 7, 99, 1]
    inicio = time.time() # Empezamos a cronometrar

    # El list() fuerza a la consola a "esperar" a que terminen o muestren el error
    with ProcessPoolExecutor() as executor:
        list(executor.map(run_seed, seeds))
        
    print("\n==================================================")
    print(" ¡PROCESO COMPLETO! Todas las semillas entrenadas.")
    print("==================================================")
    
    
    fin = time.time() # Paramos el cronómetro
    
    tiempo_total = fin - inicio
    minutos = int(tiempo_total // 60)
    segundos = int(tiempo_total % 60)
    print(f"\nTiempo total de ejecución: {minutos} minutos y {segundos} segundos.")