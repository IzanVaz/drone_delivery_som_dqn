import os
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from env.drone_env import DroneDeliveryEnv
from dqn.agent import DQNAgent
import traceback
import time

def run_seed_ablacion(seed_value):
    try:
        print(f"\n>>> [ABLACIÓN - Semilla {seed_value}] Iniciando...")
        env = DroneDeliveryEnv(max_steps=100)
        
        # EL CAMBIO CLAVE: n_inputs=7 y is_discrete=False
        agent = DQNAgent(n_inputs=7, n_actions=6, is_discrete=False, epsilon_decay=60000)
        
        num_episodes = 10000
        history = {"returns": [], "lengths": []}
        
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed_value + ep)
            done = False
            total_return = 0
            steps_in_ep = 0
            
            while not done:
                action = agent.select_action(obs, training=True) 
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store(obs, action, reward, next_obs, done)
                
                if steps_in_ep % 2 == 0:
                    agent.update()
                
                obs = next_obs
                total_return += reward
                steps_in_ep += 1
                
            history["returns"].append(total_return)
            
            if (ep + 1) % 1000 == 0:
                avg_rew = np.mean(history["returns"][-1000:])
                print(f"[ABLACIÓN {seed_value}] Ep {ep+1:5d} | Media: {avg_rew:8.1f}")

        results_dir = "results_ablacion"
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        
        np.savez(f"{results_dir}/baseline_data_seed_{seed_value}.npz", returns=history["returns"])
        agent.save(f"{results_dir}/baseline_dqn_seed_{seed_value}.pth")
        
        print(f"!!! [ABLACIÓN {seed_value}] Completada.")
        return seed_value
        
    except Exception as e:
        print(f"\n!!! ERROR FATAL EN SEMILLA {seed_value} !!!")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    seeds = [42, 123, 7, 99, 1]
    inicio = time.time() # Empezamos a cronometrar
    # Forzamos a convertir el mapa en lista para que no silencie los errores
    with ProcessPoolExecutor() as executor:
        list(executor.map(run_seed_ablacion, seeds))
    print("\nEstudio de ablación finalizado.")
    
    
    
    fin = time.time() # Paramos el cronómetro
    
    tiempo_total = fin - inicio
    minutos = int(tiempo_total // 60)
    segundos = int(tiempo_total % 60)
    print(f"\nTiempo total de ejecución: {minutos} minutos y {segundos} segundos.")