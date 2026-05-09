import time
import torch
from env.drone_env import DroneDeliveryEnv
from som.som import SOM
from dqn.agent import DQNAgent

def watch_trained_agent(seed=42):
    print(f"Cargando modelo entrenado de la semilla {seed}...")
    
    # Inicializar entorno en modo visual
    env = DroneDeliveryEnv(render_mode="human")
    
    # Cargar el SOM y el DQN
    som = SOM.load(f"results/som_seed_{seed}.npz")
    agent = DQNAgent(n_nodes=64, n_actions=6)
    agent.load(f"results/dqn_seed_{seed}.pth")

    obs, _ = env.reset(seed=seed)
    done = False
    recompensa_total = 0

    print("Iniciando vuelo de prueba (Epsilon = 0)...")
    while not done:
        env.render()
        time.sleep(0.2)  # Pausa para que el ojo humano pueda seguir el movimiento
        
        # Inferencia híbrida
        state_idx = som.predict(obs)
        action = agent.select_action(state_idx, training=False)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        recompensa_total += reward
        done = terminated or truncated

    env.render()
    print(f"Episodio finalizado. Recompensa total: {recompensa_total}")
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    watch_trained_agent(seed=42)