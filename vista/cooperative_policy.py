"""Policies for the heterogeneous fixed-ground Orbital Eyes scenario."""

import numpy as np
import torch
from torch import nn

import pufferlib.models
from pufferlib.pytorch import layer_init


class OrbitalEyesCooperativeAttentionRecurrent(pufferlib.models.LSTMWrapper):
    """Recurrent core for the modality-aware entity policy."""

    def __init__(self, env, policy, input_size="auto", hidden_size="auto", **kwargs):
        policy_size = int(policy.hidden_size)
        input_size = policy_size if input_size == "auto" else int(input_size)
        hidden_size = policy_size if hidden_size == "auto" else int(hidden_size)
        if input_size != policy_size or hidden_size != policy_size:
            raise ValueError(
                "Attention recurrent dimensions must match policy.hidden_size; "
                "set both rnn dimensions to auto."
            )
        super().__init__(env, policy, input_size, hidden_size)


class OrbitalEyesCooperativeFlatRecurrent(pufferlib.models.LSTMWrapper):
    """Recurrent core for the flat multi-sensor baseline."""

    def __init__(self, env, policy, input_size="auto", hidden_size="auto", **kwargs):
        policy_size = int(policy.hidden_size)
        input_size = policy_size if input_size == "auto" else int(input_size)
        hidden_size = policy_size if hidden_size == "auto" else int(hidden_size)
        if input_size != policy_size or hidden_size != policy_size:
            raise ValueError(
                "Flat recurrent dimensions must match policy.hidden_size; "
                "set both rnn dimensions to auto."
            )
        super().__init__(env, policy, input_size, hidden_size)


class _MaskedCandidatePolicy(nn.Module):
    SELF_DIM = 18
    AGENT_DIM = 11
    RSO_DIM = 19
    RSO_VISIBILITY_IDX = 12

    def _split_observation(self, observations):
        batch_size = observations.shape[0]
        obs = observations.reshape(batch_size, -1).float()

        idx = 0
        self_obs = obs[:, idx:idx + self.SELF_DIM]
        idx += self.SELF_DIM

        agent_obs = obs[:, idx:idx + self.num_other_agents * self.AGENT_DIM]
        agent_obs = agent_obs.reshape(
            batch_size, self.num_other_agents, self.AGENT_DIM
        )
        idx += self.num_other_agents * self.AGENT_DIM

        rso_obs = obs[:, idx:idx + self.rso_top_k * self.RSO_DIM]
        rso_obs = rso_obs.reshape(batch_size, self.rso_top_k, self.RSO_DIM)
        return obs, self_obs, agent_obs, rso_obs

    def _set_action_mask(self, rso_obs):
        if not self.mask_actions:
            self._action_mask = None
            return

        feasible = rso_obs[:, :, self.RSO_VISIBILITY_IDX] > 0.5
        # Hold is a fallback, not an attractive idle action. When any target is
        # reachable the policy must allocate the sensor; otherwise hold is the
        # sole valid action and preserves the current boresight.
        hold = ~feasible.any(dim=1, keepdim=True)
        self._action_mask = torch.cat([feasible, hold], dim=1)


class OrbitalEyesCooperativeAttentionPolicy(_MaskedCandidatePolicy):
    """Entity policy with independent role and sensor-modality embeddings.

    Observation layout:
        self(18) | teammates((N-1)*11) | candidate_RSOs(K*19)

    Self modality is feature 12. Teammate modality is feature 7. These scalar
    labels are removed before the continuous role encoders and represented by
    a learned Type-A/Type-B embedding instead.
    """

    SELF_MODALITY_IDX = 12
    AGENT_MODALITY_IDX = 7
    NUM_MODALITIES = 2

    def __init__(
        self,
        env,
        num_agents=4,
        rso_top_k=60,
        action_mode=0,
        d_model=64,
        nhead=4,
        transformer_num_layers=1,
        hidden_size=64,
        mask_actions=True,
        **kwargs,
    ):
        super().__init__()
        if action_mode != 0:
            raise ValueError(
                "OrbitalEyesCooperativeAttentionPolicy requires discrete actions"
            )
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if int(env.single_action_space.n) != rso_top_k + 1:
            raise ValueError("Action space must contain K candidates plus hold")

        self.hidden_size = int(hidden_size)
        self.is_continuous = False
        self.d_model = int(d_model)
        self.num_other_agents = int(num_agents) - 1
        self.rso_top_k = int(rso_top_k)
        self.mask_actions = bool(mask_actions)
        self._action_mask = None
        self._rso_hidden = None

        self.self_encoder = nn.Sequential(
            layer_init(nn.Linear(self.SELF_DIM - 1, self.d_model)),
            nn.GELU(),
        )
        self.agent_encoder = nn.Sequential(
            layer_init(nn.Linear(self.AGENT_DIM - 1, self.d_model)),
            nn.GELU(),
        )
        self.rso_encoder = nn.Sequential(
            layer_init(nn.Linear(self.RSO_DIM, self.d_model)),
            nn.GELU(),
        )
        self.role_embedding = nn.Embedding(3, self.d_model)
        self.sensor_modality_embedding = nn.Embedding(
            self.NUM_MODALITIES, self.d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=self.d_model * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=int(transformer_num_layers)
        )
        self.summary_projection = layer_init(
            nn.Linear(self.d_model, self.hidden_size)
        )
        self.pointer_query = nn.Linear(self.hidden_size, self.d_model)
        self.pointer_key = nn.Linear(self.d_model, self.d_model)
        self.hold_head = layer_init(nn.Linear(self.hidden_size, 1), std=0.01)
        self.value = layer_init(nn.Linear(self.hidden_size, 1), std=1.0)

    @staticmethod
    def _drop_feature(features, index):
        return torch.cat([features[..., :index], features[..., index + 1:]], dim=-1)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        return self.decode_actions(hidden, state)

    def forward_train(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        _, self_obs, agent_obs, rso_obs = self._split_observation(observations)
        self._set_action_mask(rso_obs)

        self_modality = self_obs[:, self.SELF_MODALITY_IDX].long().clamp(
            0, self.NUM_MODALITIES - 1
        )
        agent_modality = agent_obs[:, :, self.AGENT_MODALITY_IDX].long().clamp(
            0, self.NUM_MODALITIES - 1
        )

        self_continuous = self._drop_feature(
            self_obs, self.SELF_MODALITY_IDX
        )
        agent_continuous = self._drop_feature(
            agent_obs, self.AGENT_MODALITY_IDX
        )

        self_token = (
            self.self_encoder(self_continuous)
            + self.role_embedding.weight[0]
            + self.sensor_modality_embedding(self_modality)
        ).unsqueeze(1)
        agent_tokens = (
            self.agent_encoder(agent_continuous)
            + self.role_embedding.weight[1]
            + self.sensor_modality_embedding(agent_modality)
        )
        rso_tokens = (
            self.rso_encoder(rso_obs)
            + self.role_embedding.weight[2]
        )

        tokens = torch.cat([self_token, agent_tokens, rso_tokens], dim=1)
        contextualized = self.transformer(tokens)
        rso_start = 1 + self.num_other_agents
        self._rso_hidden = contextualized[
            :, rso_start:rso_start + self.rso_top_k, :
        ]
        return self.summary_projection(contextualized[:, 0, :])

    def decode_actions(self, hidden, state=None):
        query = self.pointer_query(hidden).unsqueeze(1)
        keys = self.pointer_key(self._rso_hidden)
        candidate_logits = torch.bmm(
            query, keys.transpose(1, 2)
        ).squeeze(1) / (self.d_model ** 0.5)
        logits = torch.cat([candidate_logits, self.hold_head(hidden)], dim=1)
        if self._action_mask is not None:
            logits = logits.masked_fill(~self._action_mask, -1e9)
        return logits, self.value(hidden)


class OrbitalEyesCooperativeFlatPolicy(_MaskedCandidatePolicy):
    """Flat MLP encoder plus LSTM baseline using the same information and mask."""

    def __init__(
        self,
        env,
        hidden_size=512,
        num_agents=4,
        rso_top_k=60,
        action_mode=0,
        mask_actions=True,
        **kwargs,
    ):
        super().__init__()
        if action_mode != 0:
            raise ValueError(
                "OrbitalEyesCooperativeFlatPolicy requires discrete actions"
            )
        if int(env.single_action_space.n) != rso_top_k + 1:
            raise ValueError("Action space must contain K candidates plus hold")

        self.hidden_size = int(hidden_size)
        self.is_continuous = False
        self.num_other_agents = int(num_agents) - 1
        self.rso_top_k = int(rso_top_k)
        self.mask_actions = bool(mask_actions)
        self._action_mask = None

        observation_size = int(np.prod(env.single_observation_space.shape))
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(observation_size, self.hidden_size)),
            nn.GELU(),
            layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.GELU(),
        )
        self.actor = layer_init(
            nn.Linear(self.hidden_size, env.single_action_space.n), std=0.01
        )
        self.value = layer_init(nn.Linear(self.hidden_size, 1), std=1.0)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        return self.decode_actions(hidden, state)

    def forward_train(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        obs, _, _, rso_obs = self._split_observation(observations)
        self._set_action_mask(rso_obs)
        return self.encoder(obs)

    def decode_actions(self, hidden, state=None):
        logits = self.actor(hidden)
        if self._action_mask is not None:
            logits = logits.masked_fill(~self._action_mask, -1e9)
        return logits, self.value(hidden)
