import ray
from ray import air, tune
import ray.rllib.algorithms.ppo as ppo
#import ray.rllib.algorithms.a2c as a2c
#import ray.rllib.algorithms.sac as sac
#import ray.rllib.algorithms.ddpg as ddpg

from stop_logic import StopLogic
#from stop_long  import StopLong
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper

ray.init()

# Define which learning algorithm we will use
algo = "PPO"
params = ppo.DEFAULT_CONFIG.copy() #a2c & td3 don't have defaults (td3 builds on ddpg)

# Define the custom environment for Ray
env_config = {}
env_config["time_step_size"]                = 0.5
env_config["debug"]                         = 0
env_config["training"]                      = True
env_config["randomize_start_dist"]          = True #tune.choice([True, False])
env_config["neighbor_speed"]                = 29.1 #29.1 m/s is posted speed limit
env_config["neighbor_start_loc"]            = 320.0 #dist downtrac from beginning of lane 1 for n3, m
env_config["neighbor_first_timestep"]       = 1300000  #first total time step that neighbors are put into motion (curriculum learning)
#env_config["init_ego_lane"]                 = 0

# Algorithm configs
params["env"]                               = SimpleHighwayRampWrapper
params["env_config"]                        = env_config
params["framework"]                         = "torch"
params["num_gpus"]                          = 1 #for the local worker, which does the evaluation runs
params["num_cpus_per_worker"]               = 1 #also applies to the local worker and evaluation workers
params["num_gpus_per_worker"]               = 0 #this has to allow for evaluation workers also
params["num_workers"]                       = 1 #num remote workers (remember that there is a local worker also)
params["num_envs_per_worker"]               = 1
params["rollout_fragment_length"]           = 200 #timesteps pulled from a sampler
params["gamma"]                             = 0.999 #tune.choice([0.99, 0.999])
params["evaluation_interval"]               = 6
params["evaluation_duration"]               = 6
params["evaluation_duration_unit"]          = "episodes"
params["evaluation_parallel_to_training"]   = True #True requires evaluation_num_workers > 0
params["evaluation_num_workers"]            = 2
params["log_level"]                         = "WARN"
params["seed"]                              = tune.choice([2, 17, 666, 4334, 10003, 29771, 38710, 50848, 81199])
params["batch_mode"]                        = "complete_episodes"

# ===== Params for DDPG =====================================================================
"""
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
params["actor_lr"]                          = tune.choice([2e-8, 4e-8, 7e-8, 1e-5, 1e-4, 1e-3]) #tune.choice([1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
params["critic_lr"]                         = tune.choice([9e-5, 2e-4, 5e-4]) #tune.loguniform(3e-5, 2e-4)
params["tau"]                               = 0.005 #tune.choice([0.0005, 0.001, 0.005])
params["train_batch_size"]                  = tune.choice([4, 8, 16])
"""
# ===== Params for TD3 (added to the DDPG params) ===========================================
"""
params["twin_q"]                            = True
params["policy_delay"]                      = 2
params["smooth_target_policy"]              = True
params["l2_reg"]                            = 0.0
"""
# ===== Params for PPO ======================================================================

params["lr_schedule"]                       = [[0, 1.0e-4], [800000, 1.0e-5], [1400000, 1.0e-5], [7000000, 1.0e-6]]
params["sgd_minibatch_size"]                = 32 #must be <= train_batch_size (and divide into it)
params["train_batch_size"]                  = 2400 #must be = rollout_fragment_length * num_workers * num_envs_per_worker
#params["grad_clip"]                         = tune.uniform(0.1, 0.5)
#params["clip_param"]                        = None #tune.choice([0.2, 0.3, 0.6, 1.0])

# Add dict here for lots of model HPs
model_config = params["model"]
model_config["fcnet_hiddens"]               = tune.choice([[128, 50], [128, 50], [256, 64], [512, 64]])
model_config["fcnet_activation"]            = "relu" #tune.choice(["relu", "relu", "tanh"])
model_config["post_fcnet_activation"]       = "linear" #tune.choice(["linear", "tanh"])
params["model"] = model_config

explore_config = params["exploration_config"]
explore_config["type"]                      = "GaussianNoise" #default OrnsteinUhlenbeckNoise doesn't work well here
explore_config["stddev"]                    = tune.uniform(0.4, 0.75) #this param is specific to GaussianNoise
explore_config["random_timesteps"]          = 0 #tune.qrandint(0, 20000, 50000) #was 20000
explore_config["initial_scale"]             = 1.0
explore_config["final_scale"]               = 0.1 #tune.choice([1.0, 0.01])
explore_config["scale_timesteps"]           = 1600000  #tune.choice([100000, 400000]) #was 900k
params["exploration_config"] = explore_config


# ===== Final setup =========================================================================

print("\n///// {} training params are:\n".format(algo))
for item in params:
    print("{}:  {}".format(item, params[item]))

tune_config = tune.TuneConfig(
                metric                      = "episode_reward_mean",
                mode                        = "max",
                num_samples                 = 14 #number of HP trials
              )
stopper = StopLogic(max_timesteps           = 400,
                    max_iterations          = 4000,
                    phase_end_steps         = [1020000, 1720000, 10000000], #defines the phases; last entry needs to be >= num steps achievable in max_iterations
                    min_timesteps           = [1000000, 1700000,  3000000], #phase 1 ends when env neighbor_first_timestep is triggered
                    avg_over_latest         = 70,
                    success_threshold       = [5.0,  5.0,   1.0],
                    failure_threshold       = [0.0, -5.0, -10.0],
                    degrade_threshold       = 0.25,
                    compl_std_dev           = 0.05,
                    let_it_run              = [False, False, True] #stop in phase 0, 1 if it can't drive solo by its end
                   )
run_config = air.RunConfig(
                name                        = "cda0",
                local_dir                   = "~/ray_results",
                stop                        = stopper,
                sync_config                 = tune.SyncConfig(syncer = None), #for single-node or shared checkpoint dir
                checkpoint_config           = air.CheckpointConfig(
                                                checkpoint_frequency        = 10,
                                                checkpoint_score_attribute  = "episode_reward_mean",
                                                num_to_keep                 = 1, #if > 1 hard to tell which one is the best
                                                checkpoint_at_end           = True
                )
             )

#checkpoint criteria: checkpoint_config=air.CheckpointConfig()

tuner = tune.Tuner(algo, param_space=params, tune_config=tune_config, run_config=run_config)
print("\n///// Tuner created.\n")

result = tuner.fit()
#print("\n///// tuner.fit() returned: ", type(result), " - ", result[0]) #we should only look at result[0] for some reason?
