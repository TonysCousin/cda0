import ray
from ray import air, tune
#import ray.rllib.algorithms.ppo as ppo
#import ray.rllib.algorithms.a2c as a2c
#import ray.rllib.algorithms.sac as sac
import ray.rllib.algorithms.ddpg as ddpg

from stop_logic import StopLogic
from stop_long  import StopLong
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper

ray.init()

# Define which learning algorithm we will use
algo = "DDPG"
params = ddpg.DEFAULT_CONFIG.copy() #a2c & td3 don't have defaults (td3 builds on ddpg)

# Define the custom environment for Ray
env_config = {}
env_config["time_step_size"]                = 0.5
env_config["debug"]                         = 0
env_config["training"]                      = True

# Algorithm configs
params["env"]                               = SimpleHighwayRampWrapper
params["env_config"]                        = env_config
params["framework"]                         = "torch"
params["num_gpus"]                          = 1 #for the local worker
params["num_cpus_per_worker"]               = 1 #also applies to the local worker and evaluation workers
params["num_gpus_per_worker"]               = 0 #this has to allow for evaluation workers also
params["num_workers"]                       = 12 #num remote workers (remember that there is a local worker also)
params["num_envs_per_worker"]               = 1
params["rollout_fragment_length"]           = 200 #timesteps
params["gamma"]                             = 0.999 #tune.choice([0.99, 0.999])
params["evaluation_interval"]               = 6
params["evaluation_duration"]               = 6
params["evaluation_duration_unit"]          = "episodes"
params["evaluation_parallel_to_training"]   = True #True requires evaluation_num_workers > 0
params["evaluation_num_workers"]            = 2
params["log_level"]                         = "WARN"
params["seed"]                              = 17

# ===== Params for DDPG =====================================================================

replay = params["replay_buffer_config"]
replay["type"]                              = "MultiAgentReplayBuffer"
replay["capacity"]                          = 1000000
replay["learning_starts"]                   =   10000

exp = params["exploration_config"]
exp["type"]                                 = "OrnsteinUhlenbeckNoise"
exp["random_timesteps"]                     = 100000
#exp["stddev"]                               = 0.5 #used for GaussianNoise only
exp["initial_scale"]                        = 1.0 #tune.choice([1.0, 0.05])
exp["final_scale"]                          = 0.1
exp["scale_timesteps"]                      = 4000000
exp["ou_sigma"]                             = 1.0
#exp.pop("ou_sigma")                         #these ou items need to be removed if not using OU noise
#exp.pop("ou_theta")
#exp.pop("ou_base_scale")

params["replay_buffer_config"]              = replay
params["exploration_config"]                = exp
params["actor_hiddens"]                     = [512, 64] #tune.choice([ [512, 64],
params["critic_hiddens"]                    = [768, 80] #tune.choice([[768, 80],
params["actor_lr"]                          = tune.loguniform(7e-8, 5e-7) #tune.choice([1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
params["critic_lr"]                         = tune.loguniform(3e-5, 1e-4) #tune.loguniform(3e-5, 2e-4)
params["tau"]                               = 0.005 #tune.choice([0.0005, 0.001, 0.005])
params["train_batch_size"]                  = tune.choice([8, 16, 32, 128, 1024])

# ===== Params for TD3 (added to the DDPG params) ===========================================
"""
params["twin_q"]                            = True
params["policy_delay"]                      = 2
params["smooth_target_policy"]              = True
params["l2_reg"]                            = 0.0
"""
# ===== Params for PPO ======================================================================
"""
params["lr"]                                = tune.loguniform(1e-6, 1e-4)
params["sgd_minibatch_size"]                = 256 #must be <= train_batch_size
params["train_batch_size"]                  = 256 #tune.choice([64, 64, 128, 256, 512, 1024])

# Add dict here for lots of model HPs
model_config = params["model"]
#model_config["fcnet_hiddens"]               = [300, 128, 64]
model_config["fcnet_hiddens"]               = tune.choice([ [300, 128, 64],
                                                            [200, 100, 20],
                                                            [384, 128, 32]
                                                          ])

model_config["fcnet_activation"]            = "relu" #tune.choice(["relu", "relu", "tanh"])
model_config["post_fcnet_activation"]       = tune.choice(["linear", "tanh"])
params["model"] = model_config
"""

# ===== Final setup =========================================================================

print("\n///// {} training params are:\n".format(algo))
for item in params:
    print("{}:  {}".format(item, params[item]))

tune_config = tune.TuneConfig(
                metric                      = "episode_reward_mean",
                mode                        = "max",
                num_samples                 = 15 #number of HP trials
              )
stopper = StopLong(max_timesteps           = 300,
                    max_iterations          = 2000,
                    min_iterations          = 300,
                    avg_over_latest         = 60,
                    success_threshold       = 1.1,
                    failure_threshold       = 0.0,
                    compl_std_dev           = 0.02
                   )
run_config = air.RunConfig(
                name                        = "cda0-l01-free",
                local_dir                   = "~/ray_results",
                stop                        = stopper,
                sync_config                 = tune.SyncConfig(syncer = None), #for single-node or shared checkpoint dir
                checkpoint_config           = air.CheckpointConfig(
                                                checkpoint_frequency        = 20,
                                                checkpoint_score_attribute  = "episode_reward_mean",
                                                num_to_keep                 = 4,
                                                checkpoint_at_end           = True
                )
             )

#checkpoint criteria: checkpoint_config=air.CheckpointConfig()

tuner = tune.Tuner(algo, param_space=params, tune_config=tune_config, run_config=run_config)
print("\n///// Tuner created.\n")

result = tuner.fit()
#print("\n///// tuner.fit() returned: ", type(result), " - ", result[0]) #we should only look at result[0] for some reason?
