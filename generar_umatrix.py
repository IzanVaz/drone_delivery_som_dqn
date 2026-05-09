import numpy as np
import matplotlib.pyplot as plt
import os
from som.som import SOM

def plot_umatrix(seed, model_path_template="results/som_seed_{seed}.npz"):
    model_path = model_path_template.format(seed=seed)
    if not os.path.exists(model_path):
        print(f"No se encontró {model_path}. Prueba con otra semilla.")
        return

    # Cargamos el modelo SOM ya entrenado
    som = SOM.load(model_path)
    
    # Calculamos la matriz unificada de distancias (U-Matrix)
    umat = som.get_umatrix()

    # Generamos la gráfica
    plt.figure(figsize=(8, 6))
    plt.title(f"Dinámica Interna (U-Matrix) - SOM 20x20 (Semilla {seed})", fontsize=14)
    
    # Usamos un mapa de colores 'viridis' que es el estándar científico
    im = plt.imshow(umat, cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar(im)
    cbar.set_label('Distancia media a los vecinos (Fronteras topológicas)', rotation=270, labelpad=15)
    
    plt.xlabel("Coordenada X del SOM")
    plt.ylabel("Coordenada Y del SOM")
    plt.tight_layout()
    
    # Guardamos para el informe con el nombre de la semilla
    os.makedirs("figures", exist_ok=True)
    ruta_pdf = f"figures/umatrix_interna_seed_{seed}.pdf"
    ruta_png = f"figures/umatrix_interna_seed_{seed}.png"
    
    plt.savefig(ruta_pdf)
    plt.savefig(ruta_png)
    print(f"-> U-Matrix generada y guardada en '{ruta_pdf}'")
    
    # Cerramos la figura para liberar memoria en el bucle
    plt.close()

if __name__ == "__main__":
    # Lista de semillas usadas en el entrenamiento
    seeds = [42, 123, 7, 99, 1]
    for s in seeds:
        plot_umatrix(seed=s)