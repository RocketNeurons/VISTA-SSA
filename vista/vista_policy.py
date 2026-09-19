"""Checkpoint-compatible VISTA and flat recurrent policies."""
import numpy as np
import torch
from torch import nn
import pufferlib.models
from pufferlib.pytorch import layer_init

class OrbitalEyesFlatRecurrent(pufferlib.models.LSTMWrapper):
    """Recurrent wrapper whose dimensions follow OrbitalEyesFlatPolicy.

    Set [rnn] input_size and hidden_size to auto in the INI. Then
    policy.hidden_size is the single width control for the flat encoder,
    LSTM input, LSTM hidden state, actor, and value head.
    """
    def __init__(self, env, policy, input_size="auto", hidden_size="auto", **kwargs):
        policy_size = int(policy.hidden_size)
        input_size = policy_size if input_size == "auto" else int(input_size)
        hidden_size = policy_size if hidden_size == "auto" else int(hidden_size)

        if input_size != policy_size or hidden_size != policy_size:
            raise ValueError(
                "OrbitalEyesFlatRecurrent requires rnn.input_size and "
                "rnn.hidden_size to match policy.hidden_size; use auto "
                "for both in the INI."
            )
        super().__init__(env, policy, input_size, hidden_size)

class OrbitalEyesFlatPolicy(nn.Module):
    """Flat fixed-slot policy for the single-sensor LSTM baseline."""

    SELF_DIM = 12
    AGENT_DIM = 7
    RSO_DIM = 19
    RSO_VISIBILITY_IDX = 12

    def __init__(self, env, hidden_size=128, num_agents=1, rso_top_k=30,
                 action_mode=0, mask_actions=True, **kwargs):
        super().__init__()
        if action_mode != 0:
            raise ValueError("OrbitalEyesFlatPolicy only supports discrete action_mode=0")

        self.hidden_size = hidden_size
        self.is_continuous = False
        self.num_other_agents = num_agents - 1
        self.rso_top_k = rso_top_k
        self.mask_actions = bool(mask_actions)
        self._action_mask = None

        num_obs = int(np.prod(env.single_observation_space.shape))
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(num_obs, hidden_size)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
        )
        self.actor = layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value = layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        return self.decode_actions(hidden, state)

    def forward_train(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        obs = observations.view(batch_size, -1).float()

        if self.mask_actions:
            rso_start = self.SELF_DIM + self.AGENT_DIM * self.num_other_agents
            rso_end = rso_start + self.rso_top_k * self.RSO_DIM
            rso_obs = obs[:, rso_start:rso_end].view(batch_size, self.rso_top_k, self.RSO_DIM)
            feasible = rso_obs[:, :, self.RSO_VISIBILITY_IDX] > 0.5
            # The final action is always the explicit hold/no-op action.
            no_op = torch.ones(batch_size, 1, dtype=torch.bool, device=obs.device)
            self._action_mask = torch.cat([feasible, no_op], dim=1)
        else:
            self._action_mask = None

        return self.encoder(obs)

    def decode_actions(self, hidden, state=None):
        logits = self.actor(hidden)
        if self._action_mask is not None:
            logits = logits.masked_fill(~self._action_mask, -1e9)
        values = self.value(hidden)
        return logits, values

class VISTALSTM(pufferlib.models.LSTMWrapper):
    """LSTMWrapper around VISTA.
    input_size must match the inner policy's hidden_size."""
    def __init__(self, env, policy, input_size=128, hidden_size=128):
        super().__init__(env, policy, input_size, hidden_size)

class VISTA(nn.Module):
    """VISTA policy for orbital_eyes SSA sensor scheduling.

    Observation layout (flat, from C env)
    ─────────────────────────────────────
        [self (12) | other_agents (N-1)×7 | rso_tokens K×19]

    Per-token features
    ──────────────────
        self       (12): pos(3), vel(3), sensor_dir(3), prev_action_norm(1),
                         last_delta(1), fov_norm(1)
        agent       (7): rel_pos(3), sensor_dir(3), is_observing(1)
        rso        (19): rel_pos(3), rel_vel(3), u_r(1), u_t(1), u_n(1), sigma_vel(1),
                         age(1), size(1), visibility(1), angular_proximity(1),
                         was_my_target(1), steps_since_i_observed(1), num_agents_observing(1),
                         delta_az_norm(1), delta_el_norm(1)
    """

    SELF_DIM = 12
    AGENT_DIM = 7
    RSO_DIM = 19
    RSO_VISIBILITY_IDX = 12

    def __init__(self, env,
                 # ── env layout ──
                 num_agents=5, rso_top_k=30, action_mode=0,
                 # ── network ──
                 d_model=64, nhead=4, transformer_num_layers=2,
                 hidden_size=128, mask_actions=False,
                 **kwargs):
        super().__init__()
        if action_mode != 0:
            raise ValueError("VISTA only supports discrete action_mode=0")
        self.hidden_size = hidden_size
        self.action_mode = action_mode
        self.is_continuous = False
        self.d_model = d_model
        self.rso_top_k = rso_top_k
        self.mask_actions = bool(mask_actions)
        self._action_mask = None

        self.num_other_agents = num_agents - 1

        # ── encoders ──
        self.self_encoder = nn.Sequential(
            layer_init(nn.Linear(self.SELF_DIM, d_model)), nn.GELU())
        self.agent_encoder = nn.Sequential(
            layer_init(nn.Linear(self.AGENT_DIM, d_model)), nn.GELU())
        self.rso_encoder = nn.Sequential(
            layer_init(nn.Linear(self.RSO_DIM, d_model)), nn.GELU())

        # Type embeddings: 0=self, 1=agent, 2=rso
        self.type_embed = nn.Embedding(3, d_model)

        # ── transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_num_layers)

        # Project CLS token → hidden_size
        self.proj = layer_init(nn.Linear(d_model, hidden_size))

        # ── actor / critic heads ──
        # Pointer over K RSO tokens, optionally followed by hold/no-RSO.
        num_actions = int(env.single_action_space.n)
        if num_actions not in (rso_top_k, rso_top_k + 1):
            raise ValueError(
                "VISTA expects K or K+1 discrete actions; "
                f"got {num_actions} for K={rso_top_k}"
            )
        self.has_noop_action = num_actions == rso_top_k + 1
        self.pointer_query = nn.Linear(hidden_size, d_model)
        self.pointer_key = nn.Linear(d_model, d_model)
        if self.has_noop_action:
            self.noop_head = layer_init(
                nn.Linear(hidden_size, 1), std=0.01)

        self.value = layer_init(nn.Linear(hidden_size, 1), std=1)
        self._rso_hidden = None

    def encode_observations(self, observations, state=None):
        B = observations.shape[0]
        obs = observations.view(B, -1).float()
        device = observations.device

        idx = 0
        self_obs = obs[:, idx:idx + self.SELF_DIM]
        idx += self.SELF_DIM

        agent_obs = obs[:, idx:idx + self.num_other_agents * self.AGENT_DIM]
        agent_obs = agent_obs.view(B, self.num_other_agents, self.AGENT_DIM)
        idx += self.num_other_agents * self.AGENT_DIM

        rso_obs = obs[:, idx:idx + self.rso_top_k * self.RSO_DIM]
        rso_obs = rso_obs.view(B, self.rso_top_k, self.RSO_DIM)

        if self.mask_actions:
            feasible = rso_obs[:, :, self.RSO_VISIBILITY_IDX] > 0.5
            if self.has_noop_action:
                no_op = torch.ones(B, 1, dtype=torch.bool, device=device)
                feasible = torch.cat([feasible, no_op], dim=1)
            self._action_mask = feasible
        else:
            self._action_mask = None

        # ── encode tokens ──
        self_tok = self.self_encoder(self_obs).unsqueeze(1)      # (B,1,d)
        agent_tok = self.agent_encoder(agent_obs)                 # (B,N-1,d)
        rso_tok = self.rso_encoder(rso_obs)                       # (B,K,d)

        # ── type embeddings ──
        self_tok = self_tok + self.type_embed.weight[0]
        agent_tok = agent_tok + self.type_embed.weight[1]
        rso_tok = rso_tok + self.type_embed.weight[2]

        # ── assemble sequence ──
        tokens = torch.cat([self_tok, agent_tok, rso_tok], dim=1)

        # ── transformer ──
        hidden = self.transformer(tokens)

        # CLS token (index 0)
        cls = hidden[:, 0, :]                                    # (B, d_model)
        out = self.proj(cls)                                     # (B, hidden_size)

        # Cache RSO-position hidden states for pointer network
        rso_start = 1 + self.num_other_agents
        self._rso_hidden = hidden[:, rso_start:rso_start + self.rso_top_k, :]  # (B, K, d)

        return out

    def decode_actions(self, hidden, state=None):
        # Pointer-network: query from hidden, keys from RSO tokens
        query = self.pointer_query(hidden).unsqueeze(1)       # (B,1,d)
        keys = self.pointer_key(self._rso_hidden)             # (B,K,d)
        logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # (B,K)
        logits = logits / (self.d_model ** 0.5)
        if self.has_noop_action:
            logits = torch.cat([logits, self.noop_head(hidden)], dim=1)
        if self._action_mask is not None:
            logits = logits.masked_fill(~self._action_mask, -1e9)
        action = logits  # raw logits for PufferLib sample_logits

        values = self.value(hidden)
        return action, values
