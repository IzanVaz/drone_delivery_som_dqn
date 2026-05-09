import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, is_discrete=True, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.is_discrete = is_discrete
        
        if is_discrete:
            # Caso SOM: Usamos embedding para el índice del nodo
            self.feature_layer = nn.Embedding(n_inputs, embed_dim)
            in_features = embed_dim
        else:
            # Caso Ablación: Pasamos los 7 valores de la observación directamente
            self.feature_layer = nn.Identity()
            in_features = n_inputs

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        x = self.feature_layer(x)
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        n_inputs      = 400,
        n_actions     = 6,
        is_discrete   = True,  # Clave para diferenciar SOM de Ablación
        embed_dim     = 32,
        hidden_dim    = 128,
        lr            = 1e-3,
        gamma         = 0.99,
        epsilon_start = 1.0,
        epsilon_end   = 0.05,
        epsilon_decay = 60000,
        buffer_size   = 20000,
        batch_size    = 64,
        target_update = 1000,
        device        = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.is_discrete = is_discrete
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.updates_done = 0

        self.q_online = QNetwork(n_inputs, n_actions, is_discrete, embed_dim, hidden_dim).to(self.device)
        self.q_target = QNetwork(n_inputs, n_actions, is_discrete, embed_dim, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        
        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.loss_fn = nn.SmoothL1Loss()

    def _prep_state(self, state):
        if self.is_discrete:
            return torch.tensor([state], dtype=torch.long, device=self.device)
        else:
            return torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)

    def epsilon(self):
        progress = min(self.steps_done / self.epsilon_decay, 1.0)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1.0 - progress)

    def select_action(self, state, training=True):
        eps = self.epsilon() if training else 0.0
        if training: self.steps_done += 1

        if random.random() < eps:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                s_t = self._prep_state(state)
                return self.q_online(s_t).argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size: return

        batch = random.sample(self.memory, self.batch_size)
        
        if self.is_discrete:
            states = torch.tensor([b[0] for b in batch], dtype=torch.long, device=self.device)
            next_states = torch.tensor([b[3] for b in batch], dtype=torch.long, device=self.device)
        else:
            states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device)

        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_online(states).gather(1, actions).squeeze()
        with torch.no_grad():
            q_next = self.q_target(next_states).max(1)[0]
            q_target = rewards + (self.gamma * q_next * (1 - dones))

        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), 10.0)
        self.optimizer.step()

        self.updates_done += 1
        if self.updates_done % self.target_update == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

    def save(self, path):
        torch.save(self.q_online.state_dict(), path)

    def load(self, path):
        self.q_online.load_state_dict(torch.load(path, map_location=self.device))
        self.q_target.load_state_dict(self.q_online.state_dict())