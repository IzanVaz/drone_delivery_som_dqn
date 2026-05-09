"""
som.py
======
Mapa Autoorganizado (Self-Organizing Map) implementado desde cero
con NumPy para el proyecto SOM+DQN de reparto por dron.

El SOM actúa como CODIFICADOR DE ESTADO:
    observación (7 floats) → índice de nodo (1 entero)

Esto convierte el espacio continuo de observación en un espacio
discreto que DQN puede manejar eficientemente.

Parámetros clave
----------------
grid_n : int
    Tamaño de la cuadrícula NxN del SOM (ej: 8 → 64 nodos).
input_dim : int
    Dimensión del vector de entrada (7 para nuestro dron).
lr_init : float
    Tasa de aprendizaje inicial (decrece con el tiempo).
sigma_init : float
    Radio inicial de vecindad (decrece con el tiempo).
"""

import numpy as np


class SOM:
    """
    Mapa Autoorganizado sobre cuadrícula cuadrada NxN.

    Uso típico en el bucle de entrenamiento
    ----------------------------------------
    som = SOM(grid_n=8, input_dim=7, seed=42)

    # En cada paso:
    node_idx = som.update(obs)   # entrena Y devuelve el índice ganador
                                 # (durante entrenamiento)

    node_idx = som.predict(obs)  # solo inferencia, sin actualizar pesos
                                 # (durante evaluación)
    """

    def __init__(
        self,
        grid_n    = 8,
        input_dim = 7,
        lr_init   = 0.5,
        sigma_init= None,     # por defecto: la mitad del tamaño del mapa
        seed      = None,
    ):
        self.grid_n     = grid_n
        self.input_dim  = input_dim
        self.lr_init    = lr_init
        self.sigma_init = sigma_init if sigma_init else grid_n / 2.0
        self.rng        = np.random.default_rng(seed)

        # Número total de nodos
        self.n_nodes = grid_n * grid_n

        # Pesos del SOM: shape (n_nodes, input_dim)
        # Inicializados aleatoriamente en [0, 1] (igual que las obs normalizadas)
        self.weights = self.rng.uniform(
            0.0, 1.0, size=(self.n_nodes, input_dim)
        ).astype(np.float32)

        # Coordenadas 2D de cada nodo en la cuadrícula (para calcular vecindad)
        # nodo i → (fila i//grid_n, columna i%grid_n)
        rows = np.arange(self.n_nodes) // grid_n
        cols = np.arange(self.n_nodes) %  grid_n
        self.node_coords = np.stack([rows, cols], axis=1).astype(np.float32)
        # shape: (n_nodes, 2)

        # Contador de actualizaciones (para el decay de lr y sigma)
        self.t = 0

        # Historial para diagnósticos del informe
        self.quantization_errors = []   # error de cuantización medio
        self._error_log_interval  = 100  # registrar cada N updates

    # ──────────────────────────────────────────────
    #  Funciones de decay
    # ──────────────────────────────────────────────

    def _learning_rate(self, t):
        """Tasa de aprendizaje que decae exponencialmente."""
        return self.lr_init * np.exp(-t / 10_000)

    def _sigma(self, t):
        """Radio de vecindad que decae exponencialmente."""
        return self.sigma_init * np.exp(-t / 10_000)

    # ──────────────────────────────────────────────
    #  Funciones core
    # ──────────────────────────────────────────────

    def _best_matching_unit(self, x):
        """
        Encuentra el nodo ganador (BMU): el más cercano a x en
        espacio de pesos (distancia euclídea).

        Parámetros
        ----------
        x : np.ndarray, shape (input_dim,)

        Devuelve
        --------
        bmu_idx : int
            Índice del nodo ganador en [0, n_nodes-1].
        """
        # Distancia cuadrática entre x y cada nodo (sin raíz: más rápido)
        diff       = self.weights - x          # (n_nodes, input_dim)
        dist_sq    = np.sum(diff ** 2, axis=1) # (n_nodes,)
        return int(np.argmin(dist_sq))

    def _neighborhood(self, bmu_idx, sigma):
        """
        Calcula el factor de vecindad h(i, bmu) para todos los nodos.
        Usa una función gaussiana centrada en el BMU.

        h(i) = exp( -dist²(i, bmu) / (2 * sigma²) )

        Devuelve
        --------
        h : np.ndarray, shape (n_nodes,)
        """
        bmu_coord  = self.node_coords[bmu_idx]          # (2,)
        diff_coord = self.node_coords - bmu_coord        # (n_nodes, 2)
        dist_sq    = np.sum(diff_coord ** 2, axis=1)     # (n_nodes,)
        return np.exp(-dist_sq / (2 * sigma ** 2))       # (n_nodes,)

    # ──────────────────────────────────────────────
    #  API pública
    # ──────────────────────────────────────────────

    def update(self, x):
        """
        Procesa una observación: encuentra el BMU, actualiza los pesos
        y devuelve el índice del nodo ganador.

        Llamar durante el ENTRENAMIENTO en cada paso del entorno.

        Parámetros
        ----------
        x : np.ndarray, shape (input_dim,)
            Observación normalizada [0, 1].

        Devuelve
        --------
        bmu_idx : int
            Índice del nodo ganador (el "estado discreto" para DQN).
        """
        x = np.asarray(x, dtype=np.float32)

        # 1. Encontrar BMU
        bmu_idx = self._best_matching_unit(x)

        # 2. Calcular lr y sigma actuales
        lr    = self._learning_rate(self.t)
        sigma = self._sigma(self.t)

        # 3. Calcular vecindad gaussiana
        h = self._neighborhood(bmu_idx, sigma)  # (n_nodes,)

        # 4. Actualizar pesos:
        #    Δw_i = lr * h(i) * (x - w_i)
        delta = x - self.weights                          # (n_nodes, input_dim)
        self.weights += lr * h[:, np.newaxis] * delta     # broadcast correcto

        # 5. Registrar diagnóstico periódicamente
        self.t += 1
        if self.t % self._error_log_interval == 0:
            err = float(np.sqrt(np.sum((x - self.weights[bmu_idx]) ** 2)))
            self.quantization_errors.append((self.t, err))

        return bmu_idx

    def predict(self, x):
        """
        Devuelve el índice del nodo más cercano SIN actualizar pesos.
        Usar durante EVALUACIÓN o cuando no queremos que el SOM aprenda.

        Parámetros
        ----------
        x : np.ndarray, shape (input_dim,)

        Devuelve
        --------
        bmu_idx : int
        """
        x = np.asarray(x, dtype=np.float32)
        return self._best_matching_unit(x)

    def get_weights_grid(self):
        """
        Devuelve los pesos reorganizados en la cuadrícula 2D.
        Útil para visualizaciones del informe.

        Devuelve
        --------
        np.ndarray, shape (grid_n, grid_n, input_dim)
        """
        return self.weights.reshape(self.grid_n, self.grid_n, self.input_dim)

    def get_activation_map(self, x):
        """
        Devuelve el mapa de distancias de todos los nodos a x.
        Útil para visualizar qué zonas del SOM responden a una observación.

        Devuelve
        --------
        np.ndarray, shape (grid_n, grid_n)
            Distancias (menor = más activado).
        """
        x    = np.asarray(x, dtype=np.float32)
        diff = self.weights - x
        dist = np.sqrt(np.sum(diff ** 2, axis=1))
        return dist.reshape(self.grid_n, self.grid_n)

    def get_umatrix(self):
        """
        Calcula la U-Matrix: distancia media entre cada nodo y sus vecinos.
        Es la visualización estándar de un SOM en papers y en el informe.
        Zonas oscuras = fronteras entre clusters.
        Zonas claras  = regiones homogéneas.

        Devuelve
        --------
        np.ndarray, shape (grid_n, grid_n)
        """
        grid  = self.get_weights_grid()  # (N, N, input_dim)
        umat  = np.zeros((self.grid_n, self.grid_n))

        for r in range(self.grid_n):
            for c in range(self.grid_n):
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_n and 0 <= nc < self.grid_n:
                        d = np.sqrt(np.sum(
                            (grid[r, c] - grid[nr, nc]) ** 2
                        ))
                        neighbors.append(d)
                umat[r, c] = np.mean(neighbors) if neighbors else 0.0

        return umat

    def save(self, path):
        """Guarda los pesos y el estado del SOM en un archivo .npz."""
        np.savez(
            path,
            weights    = self.weights,
            t          = np.array([self.t]),
            grid_n     = np.array([self.grid_n]),
            input_dim  = np.array([self.input_dim]),
            lr_init    = np.array([self.lr_init]),
            sigma_init = np.array([self.sigma_init]),
        )

    @classmethod
    def load(cls, path):
        """Carga un SOM previamente guardado."""
        data      = np.load(path)
        som       = cls(
            grid_n    = int(data['grid_n'][0]),
            input_dim = int(data['input_dim'][0]),
            lr_init   = float(data['lr_init'][0]),
            sigma_init= float(data['sigma_init'][0]),
        )
        som.weights = data['weights']
        som.t       = int(data['t'][0])
        return som