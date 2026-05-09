import pygame
from env.drone_env import DroneDeliveryEnv

# Inicializamos el entorno en modo visual
env = DroneDeliveryEnv(render_mode="human")
env.reset()

print("Generando mapa... Cierra la ventana de Pygame para salir.")

# Bucle infinito simple para mantener la ventana abierta
running = True
while running:
    env.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()