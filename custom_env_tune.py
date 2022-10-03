import ray
from ray import air, tune
#import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.a2c as a2c
#import ray.rllib.algorithms.sac as sac

from stop_logic import StopLogic
from projects.cda0.simple_highway_with_ramp import SimpleHighwayRamp

ray.init()

# Define which learning algorithm we will use
algo = "A2C"
params = {} #a2c.DEFAULT_CONFIG.copy()

# Define the custom environment for Ray
env_config = {"corridor_length": 10}

params["env"]                               = SimpleHighwayRamp
params["env_config"]                        = env_config
params["framework"]                         = "torch"
params["num_gpus"]                          = 0 #for the local worker
params["num_cpus_per_worker"]               = 1 #also applies to the local worker and evaluation workers
params["num_gpus_per_worker"]               = 0.15 #this has to allow for evaluation workers also
params["num_workers"]                       = 4 #num remote workers (remember that there is a local worker also)
params["num_envs_per_worker"]               = 2
params["rollout_fragment_length"]           = 200 #timesteps
params["gamma"]                             = 0.99
params["lr"]                                = tune.loguniform(0.0001, 0.003)
params["train_batch_size"]                  = tune.choice([1000, 2000, 4000])
params["evaluation_interval"]               = 5
params["evaluation_duration"]               = 6
params["evaluation_duration_unit"]          = "episodes"
params["evaluation_parallel_to_training"]   = True #True requires evaluation_num_workers > 0
params["evaluation_num_workers"]            = 1
params["log_level"]                         = "WARN"
params["seed"]                              = tune.lograndint(1, 256000)
# Add dict here for lots of model HPs

print("\n///// {} training params are:\n".format(algo))
for item in params:
    print("{}:  {}".format(item, params[item]))

tune_config = tune.TuneConfig(
                metric      = "episode_reward_mean",
                mode        = "max",
                num_samples = 4 #number of HP trials
              )
stopper = StopLogic(max_timesteps       = 50000,
                    max_iterations      = 80,
                    min_iterations      = 30,
                    avg_over_latest     = 8,
                    success_threshold   = 0.8,
                    failure_threshold   = 0.0,
                    compl_std_dev       = 0.01
                   )
run_config = air.RunConfig(
                name        = "John-Tune-experiment",
                local_dir   = "~/ray_results",
                stop        = stopper,
                sync_config = tune.SyncConfig(syncer = None), #for single-node or shared checkpoint dir
                checkpoint_config   = air.CheckpointConfig(
                                        checkpoint_frequency        = 10,
                                        checkpoint_score_attribute  = "episode_reward_mean",
                                        num_to_keep                 = 1,
                                        checkpoint_at_end           = True
                )
             )

#checkpoint criteria: checkpoint_config=air.CheckpointConfig()

tuner = tune.Tuner(algo, param_space=params, tune_config=tune_config, run_config=run_config)
print("\n///// Tuner created.\n")

tuner.fit()
