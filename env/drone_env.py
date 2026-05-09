"""
drone_env.py
============
Entorno de reparto por dron sobre cuadrícula 2D (10x10) con:
  - Mapa con margen de error y obstáculos estratégicos.
  - Altura máxima de 5 metros.
  - Viento fijo (Este a Oeste) con p = 0.01 * z.
  - Recompensas: Meta +5000, Choque -1000, Paso -1, Acercarse +1.1.
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# Constantes de diseño
CELL_SIZE      = 60       
HUD_WIDTH      = 220      
FPS            = 30       

# Colores (RGB)
C_BG           = (15,  20,  30)   
C_GRID         = (30,  40,  55)   
C_BUILDING     = (70,  80,  100)  
C_BUILD_TEXT   = (180, 200, 220)  
C_DRONE        = (0,   220, 180)  
C_DEST         = (255, 200, 0)    
C_WIND_ARROW   = (100, 160, 255)  
C_HUD_BG       = (10,  15,  25)   
C_HUD_TEXT     = (200, 220, 240)  
C_HUD_VAL      = (0,   220, 180)  
C_CRASH        = (220, 50,  50)   

WIND_DIRS = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
WIND_DIR_LABELS = {0: "N", 1: "S", 2: "E", 3: "O"}

ACTION_DELTAS = {
    0: (0, -1, 0), 1: (0, 1, 0), 2: (1, 0, 0), 3: (-1, 0, 0),
    4: (0, 0, 1),  5: (0, 0, -1)
}
ACTION_LABELS = {0:"N", 1:"S", 2:"E", 3:"O", 4:"↑", 5:"↓"}

class DroneDeliveryEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, grid_w=10, grid_h=10, max_z=5, max_steps=200, render_mode=None, seed=None):
        super().__init__()
        self.grid_w, self.grid_h, self.max_z, self.max_steps = grid_w, grid_h, max_z, max_steps
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.buildings = {}
        self._build_custom_map()
        
        self.screen = None
        self.clock = None
        self.font_sm = self.font_md = self.font_lg = None

    def _build_custom_map(self):
        self.buildings = {}
        # Barrera en y=4 con hueco central amplio (x=3 a x=6 libres)
        h_row = [4, 5, 3, 0, 0, 0, 0, 3, 5, 4]
        for x, h in enumerate(h_row):
            if h > 0: self.buildings[(x, 4)] = h
        # Edificios extra bajos para margen de error
        self.buildings[(7, 2)] = 3
        self.buildings[(2, 6)] = 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.drone_x, self.drone_y, self.drone_z = 5, 9, 1
        self.dest_x, self.dest_y = 8, 1
        self.global_wind_dir, self.global_wind_force = 3, 0.8
        self.step_count = 0
        self.crashed = False
        self.last_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.drone_x/9, self.drone_y/9, self.drone_z/5, 
                         self.dest_x/9, self.dest_y/9, 
                         self.global_wind_dir/3, self.global_wind_force], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        dist_before = abs(self.drone_x - self.dest_x) + abs(self.drone_y - self.dest_y)
        
        # Viento: p = 0.01 * altura
        if self.drone_z > 0 and self.rng.random() < (0.01 * self.drone_z):
            self.drone_x = max(0, self.drone_x - 1)

        # Movimiento
        dx, dy, dz = ACTION_DELTAS[action]
        nx, ny, nz = self.drone_x + dx, self.drone_y + dy, np.clip(self.drone_z + dz, 0, self.max_z)

        # Colisiones (-1000 castigo)
        if not (0 <= nx < 10 and 0 <= ny < 10) or nz <= self.buildings.get((nx, ny), 0):
            self.crashed = True
            return self._get_obs(), -1000.0, True, False, {}

        self.drone_x, self.drone_y, self.drone_z = nx, ny, nz
        dist_after = abs(self.drone_x - self.dest_x) + abs(self.drone_y - self.dest_y)

        # Recompensas
        if self.drone_x == self.dest_x and self.drone_y == self.dest_y:
            return self._get_obs(), 5000.0, True, False, {}
        
        reward = -1.0 # Coste base
        if dist_after < dist_before: reward += 1.5 # Neto +0.5
        
        self.last_reward = reward
        return self._get_obs(), reward, False, self.step_count >= self.max_steps, {}

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((10 * CELL_SIZE + HUD_WIDTH, 10 * CELL_SIZE))
        pygame.display.set_caption("Drone Delivery APBIO")
        self.clock = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("monospace", 11)
        self.font_md = pygame.font.SysFont("monospace", 14)
        self.font_lg = pygame.font.SysFont("monospace", 18, bold=True)

    def render(self):
        if self.render_mode != "human": return
        if self.screen is None: self._init_pygame()
        self.screen.fill(C_BG)
        for x in range(10):
            for y in range(10):
                pygame.draw.rect(self.screen, C_GRID, (x*60, y*60, 60, 60), 1)
        for (bx, by), bh in self.buildings.items():
            color = (int(50+bh*20), int(60+bh*20), int(80+bh*20))
            pygame.draw.rect(self.screen, color, (bx*60+2, by*60+2, 56, 56))
            self.screen.blit(self.font_sm.render(f"h:{bh}", True, (255,255,255)), (bx*60+5, by*60+25))
        pygame.draw.rect(self.screen, C_DEST, (self.dest_x*60+4, self.dest_y*60+4, 52, 52), 2)
        color = C_CRASH if self.crashed else C_DRONE
        pygame.draw.circle(self.screen, color, (int(self.drone_x*60+30), int(self.drone_y*60+30)), 15)
        # HUD simplificado
        pygame.draw.rect(self.screen, C_HUD_BG, (600, 0, HUD_WIDTH, 600))
        self.screen.blit(self.font_md.render(f"Z: {self.drone_z}", True, C_HUD_VAL), (610, 50))
        self.screen.blit(self.font_md.render(f"Rew: {self.last_reward}", True, C_HUD_VAL), (610, 80))
        pygame.display.flip(); self.clock.tick(FPS)

    def close(self):
        if self.screen: pygame.quit(); self.screen = None