import ray
from ray import air, tune
import ray.rllib.algorithms.ppo as ppo
#import ray.rllib.algorithms.a2c as a2c
#import ray.rllib.algorithms.sac as sac

from stop_logic import StopLogic
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper

ray.init()

# Define which learning algorithm we will use
algo = "PPO"
params = ppo.DEFAULT_CONFIG.copy() #a2c requires empty dict

# Define the custom environment for Ray
env_config = {  "time_step_size":   0.5,
                "debug":            0,
                #"init_ego_lane":    0 #left-most lane, which is just straight
             }

params["env"]                               = SimpleHighwayRampWrapper
params["env_config"]                        = env_config
params["framework"]                         = "torch"
params["num_gpus"]                          = 1 #for the local worker
params["num_cpus_per_worker"]               = 1 #also applies to the local worker and evaluation workers
params["num_gpus_per_worker"]               = 0 #this has to allow for evaluation workers also
params["num_workers"]                       = 14 #num remote workers (remember that there is a local worker also)
params["num_envs_per_worker"]               = 4
params["rollout_fragment_length"]           = 200 #timesteps
params["gamma"]                             = 0.99
params["lr"]                                = tune.loguniform(3e-6, 3e-4)
params["sgd_minibatch_size"]                = 64
params["train_batch_size"]                  = tune.choice([64, 128, 256, 512, 1024, 2048])
params["evaluation_interval"]               = 6
params["evaluation_duration"]               = 6
params["evaluation_duration_unit"]          = "episodes"
params["evaluation_parallel_to_training"]   = True #True requires evaluation_num_workers > 0
params["evaluation_num_workers"]            = 1
params["log_level"]                         = "WARN"
params["seed"]                              = tune.lograndint(1, 1048576)
# Add dict here for lots of model HPs
model_config = params["model"]
model_config["fcnet_hiddens"]               = [300, 128, 64]
"""
model_config["fcnet_hiddens"]               = tune.choice([ [300, 128, 64],
                                                            [256, 256]
                                                          ])
"""
model_config["fcnet_activation"]            = tune.choice(["relu", "tanh"])
params["model"] = model_config

print("\n///// {} training params are:\n".format(algo))
for item in params:
    print("{}:  {}".format(item, params[item]))

tune_config = tune.TuneConfig(
                metric      = "episode_reward_mean",
                mode        = "max",
                num_samples = 20 #number of HP trials
              )
stopper = StopLogic(max_timesteps       = 200,
                    max_iterations      = 500,
                    min_iterations      = 100,
                    avg_over_latest     = 18,
                    success_threshold   = 1.8,
                    failure_threshold   = 0.0,
                    compl_std_dev       = 0.02
                   )
run_config = air.RunConfig(
                name        = "cda0-l01-free",
                local_dir   = "~/ray_results",
                stop        = stopper,
                sync_config = tune.SyncConfig(syncer = None), #for single-node or shared checkpoint dir
                checkpoint_config   = air.CheckpointConfig(
                                        checkpoint_frequency        = 100,
                                        checkpoint_score_attribute  = "episode_reward_mean",
                                        num_to_keep                 = 2,
                                        checkpoint_at_end           = True
                )
             )

#checkpoint criteria: checkpoint_config=air.CheckpointConfig()

tuner = tune.Tuner(algo, param_space=params, tune_config=tune_config, run_config=run_config)
print("\n///// Tuner created.\n")

result = tuner.fit()
print("\n///// tuner.fit() returned: ", result) #we should only look at result[0] for some reason?
