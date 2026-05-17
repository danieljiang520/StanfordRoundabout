import gymnasium as gym
import torch as th
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

import highway_env  # noqa: F401
import src

TRAIN = False

MAX_VEHICLES = 12
MODEL_DIR = "roundabout_dqn_gnn"
VIDEO_TRAFFIC_VEHICLES_COUNT = 1

ROUNDABOUT_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": MAX_VEHICLES,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-15, 15],
            "vy": [-15, 15],
        },
    },
}


class RoundaboutGNNExtractor(BaseFeaturesExtractor):
    """
    Masked graph feature extractor for padded kinematic vehicle observations.

    Stable-Baselines3 requires a fixed observation space, so the environment
    exposes up to MAX_VEHICLES rows and pads absent vehicles with presence=0.
    The shared node encoder, message passing, and masked pooling let the policy
    handle any number of present vehicles up to that fixed maximum without
    changing model shape.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        hidden_dim: int = 128,
        message_passing_steps: int = 2,
        presence_feature_idx: int = 0,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        if len(observation_space.shape) != 2:
            raise ValueError(
                "RoundaboutGNNExtractor expects observations shaped "
                "(vehicles_count, feature_count)."
            )

        self.presence_feature_idx = presence_feature_idx
        node_features = observation_space.shape[-1]

        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gnn_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "message": nn.Sequential(
                            nn.Linear(2 * hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                        ),
                        "update_net": nn.Sequential(
                            nn.Linear(2 * hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                        ),
                    }
                )
                for _ in range(message_passing_steps)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        present = observations[..., self.presence_feature_idx] > 0.5
        present = present.unsqueeze(-1)
        present_float = present.float()

        node_features = self.node_encoder(observations.float()) * present_float

        for layer in self.gnn_layers:
            receivers = node_features.unsqueeze(2).expand(
                -1, -1, node_features.shape[1], -1
            )
            senders = node_features.unsqueeze(1).expand(
                -1, node_features.shape[1], -1, -1
            )
            sender_mask = present_float.unsqueeze(1)
            messages = layer["message"](th.cat([receivers, senders], dim=-1))
            messages = messages * sender_mask
            denominator = sender_mask.sum(dim=2).clamp_min(1.0)
            graph_context = messages.sum(dim=2) / denominator
            update = layer["update_net"](
                th.cat([node_features, graph_context], dim=-1)
            )
            node_features = (node_features + update) * present_float

        ego_features = node_features[:, 0, :]
        mean_pool = self._masked_mean(node_features, present_float)
        max_pool = self._masked_max(node_features, present)
        return self.readout(th.cat([ego_features, mean_pool, max_pool], dim=-1))

    @staticmethod
    def _masked_mean(values: th.Tensor, mask: th.Tensor) -> th.Tensor:
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (values * mask).sum(dim=1) / denominator

    @staticmethod
    def _masked_max(values: th.Tensor, mask: th.Tensor) -> th.Tensor:
        masked_values = values.masked_fill(~mask, -th.inf)
        pooled = masked_values.max(dim=1).values
        return th.where(th.isfinite(pooled), pooled, th.zeros_like(pooled))


if __name__ == "__main__":
    # Create the environment
    env = gym.make("roundabout-v0", render_mode=None, config=ROUNDABOUT_CONFIG)
    obs, info = env.reset()

    # Create the model
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=RoundaboutGNNExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                hidden_dim=128,
                message_passing_steps=2,
            ),
            net_arch=[256, 256],
        ),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=f"{MODEL_DIR}/",
    )

    # Train the model
    if TRAIN:
        total_timesteps = int(2e5)
        print(f"Training GNN DQN for {total_timesteps:,} timesteps...")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(f"{MODEL_DIR}/model")
        del model
    env.close()

    video_config = {
        **ROUNDABOUT_CONFIG,
        "traffic_vehicles_count": VIDEO_TRAFFIC_VEHICLES_COUNT,
    }
    video_env = gym.make(
        "VariableRoundabout-v0", render_mode="rgb_array", config=video_config
    )

    # Run the trained model and record video
    model = DQN.load(f"{MODEL_DIR}/model", env=video_env)
    video_env = RecordVideo(
        video_env, video_folder=f"{MODEL_DIR}/videos", episode_trigger=lambda e: True
    )
    video_env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
    video_env.unwrapped.set_record_video_wrapper(video_env)
    print(
        "Recording videos with "
        f"{VIDEO_TRAFFIC_VEHICLES_COUNT} traffic vehicles "
        f"({MAX_VEHICLES} observed rows max)."
    )

    for videos in range(10):
        done = truncated = False
        obs, info = video_env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = video_env.step(action)
            # Render
            video_env.render()
    video_env.close()
