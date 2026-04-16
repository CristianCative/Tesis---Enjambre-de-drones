# =============================================================================
# agente_mappo.py — Agente MAPPO con política compartida
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from config_mappo import CFG


# ── Red Actor-Crítico compartida ──────────────────────────────────────────────
class RedActorCritico(nn.Module):
    def __init__(self):
        super().__init__()
        capas_actor  = [CFG.dim_obs] + CFG.capas_actor
        capas_critico = [CFG.dim_obs] + CFG.capas_critico

        # Tronco actor
        actor_layers = []
        for i in range(len(capas_actor) - 1):
            actor_layers += [nn.Linear(capas_actor[i], capas_actor[i+1]),
                             nn.Tanh()]
        actor_layers.append(nn.Linear(capas_actor[-1], CFG.dim_accion))
        self.actor = nn.Sequential(*actor_layers)

        # Tronco crítico
        critico_layers = []
        for i in range(len(capas_critico) - 1):
            critico_layers += [nn.Linear(capas_critico[i], capas_critico[i+1]),
                               nn.Tanh()]
        critico_layers.append(nn.Linear(capas_critico[-1], 1))
        self.critico = nn.Sequential(*critico_layers)

        self._init_pesos()

    def _init_pesos(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        logits = self.actor(obs)
        valor  = self.critico(obs).squeeze(-1)
        return logits, valor

    def accion_y_log_prob(self, obs):
        logits, valor = self.forward(obs)
        dist    = Categorical(logits=logits)
        accion  = dist.sample()
        log_p   = dist.log_prob(accion)
        entropia = dist.entropy()
        return accion, log_p, valor, entropia

    def evaluar(self, obs, acciones):
        logits, valor = self.forward(obs)
        dist    = Categorical(logits=logits)
        log_p   = dist.log_prob(acciones)
        entropia = dist.entropy()
        return log_p, valor, entropia


# ── Buffer de rollout ─────────────────────────────────────────────────────────
class BufferRollout:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs       = []
        self.acciones  = []
        self.log_probs = []
        self.recompensas = []
        self.valores   = []
        self.terminados = []

    def guardar(self, obs, acc, lp, r, v, done):
        # obs: (n_drones, dim_obs)  acc: (n_drones,)
        self.obs.append(obs)
        self.acciones.append(acc)
        self.log_probs.append(lp)
        self.recompensas.append(r)
        self.valores.append(v)
        self.terminados.append(done)

    def calcular_ventajas(self, ultimo_valor):
        """GAE-Lambda."""
        T  = len(self.recompensas)
        n  = self.recompensas[0].shape[0]
        ventajas  = np.zeros((T, n), dtype=np.float32)
        retornos  = np.zeros((T, n), dtype=np.float32)
        gae       = np.zeros(n, dtype=np.float32)

        for t in reversed(range(T)):
            r    = self.recompensas[t]
            done = float(self.terminados[t])
            v_t  = self.valores[t]
            v_t1 = ultimo_valor if t == T - 1 else self.valores[t + 1]
            delta = r + CFG.gamma * v_t1 * (1 - done) - v_t
            gae   = delta + CFG.gamma * CFG.lam * (1 - done) * gae
            ventajas[t] = gae
            retornos[t] = gae + v_t

        # Normalizar ventajas
        v_flat = ventajas.reshape(-1)
        ventajas = (ventajas - v_flat.mean()) / (v_flat.std() + 1e-8)
        return ventajas, retornos

    def a_tensores(self):
        obs       = torch.FloatTensor(np.array(self.obs))       # (T, n, dim_obs)
        acciones  = torch.LongTensor(np.array(self.acciones))   # (T, n)
        log_probs = torch.FloatTensor(np.array(self.log_probs)) # (T, n)
        valores   = torch.FloatTensor(np.array(self.valores))   # (T, n)
        return obs, acciones, log_probs, valores


# ── Agente MAPPO ──────────────────────────────────────────────────────────────
class AgenteMAPPO:
    def __init__(self):
        self.red   = RedActorCritico()
        self.opt_a = optim.Adam(self.red.actor.parameters(),
                                lr=CFG.lr_actor)
        self.opt_c = optim.Adam(self.red.critico.parameters(),
                                lr=CFG.lr_critico)
        self.buffer = BufferRollout()
        self.n_updates = 0

    # ── Selección de acciones ─────────────────────────────────────────────────
    def seleccionar_acciones(self, obs: np.ndarray, determinista: bool = False):
        """obs: (n_drones, dim_obs) → acciones: (n_drones,)"""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs)          # (n, dim_obs)
            logits, valores = self.red(obs_t)
            dist = Categorical(logits=logits)
            if determinista:
                acciones = logits.argmax(dim=-1)
            else:
                acciones = dist.sample()
            log_probs = dist.log_prob(acciones)
        return (acciones.numpy(),
                log_probs.numpy(),
                valores.numpy())

    # ── Actualización PPO ─────────────────────────────────────────────────────
    def actualizar(self, ultimo_valor: np.ndarray):
        ventajas, retornos = self.buffer.calcular_ventajas(ultimo_valor)
        obs_t, acc_t, lp_old_t, _ = self.buffer.a_tensores()
        T, n = obs_t.shape[:2]

        # Aplanar (T*n, ...)
        obs_flat  = obs_t.view(T * n, -1)
        acc_flat  = acc_t.view(T * n)
        lp_flat   = lp_old_t.view(T * n)
        vent_flat = torch.FloatTensor(ventajas.reshape(T * n))
        ret_flat  = torch.FloatTensor(retornos.reshape(T * n))

        total_loss_actor  = 0.0
        total_loss_critico = 0.0
        total_entropia    = 0.0

        for _ in range(CFG.epochs_ppo):
            idx = torch.randperm(T * n)
            for start in range(0, T * n, CFG.tam_minibatch):
                mb = idx[start:start + CFG.tam_minibatch]
                lp_new, v_new, entropia = self.red.evaluar(
                    obs_flat[mb], acc_flat[mb])

                ratio = torch.exp(lp_new - lp_flat[mb])
                v_mb  = vent_flat[mb]
                surr1 = ratio * v_mb
                surr2 = torch.clamp(ratio, 1 - CFG.clip_eps,
                                    1 + CFG.clip_eps) * v_mb
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_entr  = -CFG.coef_entropia * entropia.mean()
                loss_valor = CFG.coef_valor * nn.functional.mse_loss(
                    v_new, ret_flat[mb])

                loss = loss_actor + loss_entr + loss_valor

                self.opt_a.zero_grad()
                self.opt_c.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.red.parameters(), CFG.grad_max)
                self.opt_a.step()
                self.opt_c.step()

                total_loss_actor  += loss_actor.item()
                total_loss_critico += loss_valor.item()
                total_entropia    += entropia.mean().item()

        self.buffer.reset()
        self.n_updates += 1
        n_minibatches = max(1, CFG.epochs_ppo * (T * n // CFG.tam_minibatch))
        return (total_loss_actor  / n_minibatches,
                total_loss_critico / n_minibatches,
                total_entropia    / n_minibatches)
