import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="CustomPendulum-v0",
    entry_point="gym_new_classic_envs.envs.custom_pendulum:CustomPendulumEnv",
)

register(
    id="Arm-v0",
    entry_point="gym_new_classic_envs.envs.arm:ArmEnv"
)

register(
    id="Mass-v0",
    entry_point="gym_new_classic_envs.envs.mass:MassEnv"
)

register(
    id="InvertedPendulum-v0",
    entry_point="gym_new_classic_envs.envs.inverted_pendulum:InvertedPendulumEnv"
)
