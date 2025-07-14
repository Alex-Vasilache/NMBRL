# Example of using the physical cartpole wrapper as a drop-in replacement
# for the DMC cartpole wrapper in dynamic_data_generator.py

# Original import:
# from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper

# New import for physical cartpole:
from world_models.physical_cartpole_wrapper import PhysicalCartpoleWrapper as wrapper

# Usage in dynamic_data_generator.py (line ~67):
# base_env = wrapper(
#     seed=config["global"]["seed"],
#     n_envs=1,
#     render_mode="human",
#     max_episode_steps=config["data_generator"]["max_episode_steps"],
# )

# That's it! The interface is identical, so no other changes are needed.
# The physical cartpole will now use the DMC reward function instead of
# the original swingup task reward.

# Key features of the physical cartpole wrapper:
# 1. Same interface as DMCCartpoleWrapper (inherits from VecNormalize)
# 2. Uses CartPoleEnv with cartpole_type="remote" for physical hardware
# 3. Implements DMC-style reward function from dm_control cartpole
# 4. Supports render_mode="human" for visualization
# 5. Handles sim_should_stop info for graceful shutdown when window is closed

print("Physical cartpole wrapper is ready to use!")
print(
    "Simply change the import in dynamic_data_generator.py to use the physical cartpole."
)
