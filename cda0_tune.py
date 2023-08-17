import sys
import ray
from ray import air, tune
from ray.tune import Tuner
from ray.tune.tune_config import TuneConfig
from ray.tune.logger import pretty_print
from ray.air import RunConfig
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.sac as sac
import ray.rllib.algorithms.ddpg as ddpg

from stop_simple import StopSimple
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper
from cda_callbacks import CdaCallbacks

"""This program tunes (explores) hyperparameters to find a good set suitable for training.
    Usage is:
        cda0_tune <difficulty_level>
    If a difficulty level is not provided, it will default to 0.
"""

# Identify a baseline checkpoint from which to continue training
_checkpoint_path = None

# Completed level 0 on 5/30/23
#_checkpoint_path = "/home/starkj/projects/cda0/training/PPO/p256-256-128/L0-518f4/trial09/checkpoint_000086"

# Completed level 1 on 6/1/23
#_checkpoint_path = "/home/starkj/projects/cda0/training/PPO/p256-256-128/L1-abe44/trial03/checkpoint_000059"

# Completed level 3 with SAC on 6/17/23
#_checkpoint_path = "/home/starkj/projects/cda0/training/SAC/p256-256-v256-256/L3-3bbcf/trial03/checkpoint_001600"


def main(argv):

    difficulty_level = 5
    if len(argv) > 1:
        difficulty_level = min(max(int(argv[1]), 0), SimpleHighwayRampWrapper.NUM_DIFFICULTY_LEVELS)
    print("\n///// Tuning with initial environment difficulty level {}".format(difficulty_level))

    # Initialize per https://docs.ray.io/en/latest/workflows/management.html?highlight=local%20storage#storage-configuration
    ray.init() #storage = "~/ray_results/cda0")

    # Define which learning algorithm we will use and set up is default config params
    #algo = "DDPG"
    #cfg = ddpg.DDPGConfig()
    algo = "SAC"
    cfg = sac.SACConfig()
    cfg.framework("torch")
    cfg_dict = cfg.to_dict()

    # Define the stopper object that decides when to terminate training.
    # All list objects (min_timesteps, success_threshold, failure_threshold) must be of length equal to num phases in use.
    # Phase...............0             1           2           3           4           5
    success_threshold   = [9.5,         9.5,        9.5,        10.0,       8.0,        8.0]
    min_threshold       = [None,        None,       None,       None,       None,       0.0]
    fail_threshold      = [-12.0,       -12.0,      -12.0,      -12.0,      -12.0,      -12.0]
    avg_over_latest     = 400   #num most recent iters that are averaged to meet stopping criteria
    chkpt_int           = 10    #num iters between storing new checkpoints
    max_iterations      = 12000
    burn_in             = 500   #num iters before considering failure stopping
    num_trials          = 8

    # Define the stopping logic - this requires mean reward to stay at the threshold for multiple consiecutive
    # iterations, rather than just stopping on an outlier spike.
    stopper = StopSimple(max_iterations     = max_iterations,
                         avg_over_latest    = avg_over_latest,
                         avg_threshold      = success_threshold[difficulty_level],
                         min_threshold      = min_threshold[difficulty_level],
                         max_fail_threshold = fail_threshold[difficulty_level],
                         burn_in            = burn_in,
                        )

    # Define the custom environment for Ray
    env_config = {}
    env_config["difficulty_level"]              = difficulty_level
    env_config["time_step_size"]                = 0.5
    env_config["debug"]                         = 0
    env_config["verify_obs"]                    = True
    env_config["training"]                      = True
    env_config["randomize_start_dist"]          = False
    env_config["neighbor_speed"]                = 29.1  #29.1 m/s is posted speed limit; only applies for appropriate diff levels
    env_config["neighbor_start_loc"]            = 0.0   #dist downtrack from beginning of lane 1 for n3, m
    env_config["ignore_neighbor_crashes"]       = True  #if true, a crash between two neighbor vehicles won't stop the episode
    cfg.environment(env = SimpleHighwayRampWrapper, env_config = env_config)

    # Add exploration noise params
    #cfg.rl_module(_enable_rl_module_api = False) #disables the RL module API, which allows exploration config to be defined for ray 2.6

    explore_config = cfg_dict["exploration_config"]
    #print("///// Explore config:\n", pretty_print(explore_config))
    explore_config["type"]                      = "GaussianNoise" #default OrnsteinUhlenbeckNoise doesn't work well here
    explore_config["stddev"]                    = tune.uniform(0.1, 0.4) #this param is specific to GaussianNoise
    explore_config["random_timesteps"]          = 10000 #tune.qrandint(0, 20000, 50000) #was 20000
    explore_config["initial_scale"]             = 1.0
    explore_config["final_scale"]               = 0.1 #tune.choice([1.0, 0.01])
    explore_config["scale_timesteps"]           = tune.choice([2000000, 3000000, 4000000])
    exp_switch                                  = True #tune.choice([False, True, True])
    cfg.exploration(explore = exp_switch, exploration_config = explore_config)
    #cfg.exploration(explore = False)

    # Computing resources - Ray allocates 1 cpu per rollout worker and one cpu per env (2 cpus) per trial.
    # Use max_concurrent_trials in the TuneConfig area to limit the number of trials being run in parallel.
    # The number of workers does not equal the number of trials.
    # NOTE: if num_gpus = 0 then policies will always be built/evaluated on cpu, even if gpus are specified for workers;
    #       to get workers (only) to use gpu, num_gpus needs to be positive (e.g. 0.0001).
    # NOTE: local worker needs to do work for every trial, so needs to divide its time among simultaneous trials. Therefore,
    #       if gpu is to be used for local workder only, then the number of gpus available need to be divided among the
    #       number of possible simultaneous trials (as well as gpu memory).
    # This config will run 5 parallel trials on the Tensorbook.
    cfg.resources(  num_gpus                    = 0.5, #for the local worker, which does the learning & evaluation runs
                    num_cpus_for_local_worker   = 2,
                    num_cpus_per_worker         = 2, #also applies to the local worker and evaluation workers
                    num_gpus_per_worker         = 0  #this has to allow gpu left over for local worker & evaluation workers also
    )

    cfg.rollouts(   #num_rollout_workers         = 1, #num remote workers _per trial_ (remember that there is a local worker also)
                                                     # 0 forces rollouts to be done by local worker
                    num_envs_per_worker         = 1,
                    rollout_fragment_length     = 256, #timesteps pulled from a sampler
                    batch_mode                  = "complete_episodes",
    )

    cfg.fault_tolerance(
                    recreate_failed_workers     = True,
    )

    # Evaluation process
    """
    cfg.evaluation( evaluation_interval         = 10, #iterations between evals
                    evaluation_duration         = 15, #units specified next
                    evaluation_duration_unit    = "episodes",
                    evaluation_parallel_to_training = True, #True requires evaluation_num_workers > 0
                    evaluation_num_workers      = 1,
    )
    """

    # Debugging assistance
    cfg.debugging(  log_level                   = "WARN",
                    seed                        = 17, #tune.choice([2, 17, 666, 4334, 10003, 29771, 38710, 50848, 81199])
    )

    # Custom callbacks from the training algorithm - supports starting from a checkpoint
    cfg.callbacks(  CdaCallbacks)

    # ===== Training algorithm HPs for SAC ==================================================
    opt_config = cfg_dict["optimization"]
    opt_config["actor_learning_rate"]           = tune.loguniform(1e-6, 1e-4) #default 0.0003
    opt_config["critic_learning_rate"]          = tune.loguniform(1e-6, 1e-4) #default 0.0003
    opt_config["entropy_learning_rate"]         = tune.loguniform(1e-6, 1e-4) #default 0.0003

    policy_config = cfg_dict["policy_model_config"]
    policy_config["fcnet_hiddens"]              = [256, 256]
    policy_config["fcnet_activation"]           = "relu"

    q_config = cfg_dict["q_model_config"]
    q_config["fcnet_hiddens"]                   = [256, 256]
    q_config["fcnet_activation"]                = "relu"

    cfg.training(   twin_q                      = True,
                    gamma                       = 0.995,
                    train_batch_size            = 1024, #must be an int multiple of rollout_fragment_length * num_rollout_workers * num_envs_per_worker
                    initial_alpha               = 0.02, #tune.loguniform(0.002, 0.04),
                    tau                         = 0.005,
                    n_step                      = 1, #tune.choice([1, 2, 3]),
                    grad_clip                   = 1.0, #tune.uniform(0.5, 1.0),
                    optimization_config         = opt_config,
                    policy_model_config         = policy_config,
                    q_model_config              = q_config,
    )

    # ===== Final setup =========================================================================

    print("\n///// {} training params are:\n".format(algo))
    print(pretty_print(cfg.to_dict()))

    tune_config = TuneConfig(
                    metric                      = "episode_reward_mean",
                    mode                        = "max",
                    #scheduler                   = scheduler,
                    num_samples                 = num_trials,
                )

    run_config = RunConfig( #some commented-out items will allegedly be needed for Ray 2.6
                    name                        = "cda0",
                    local_dir                   = "~/ray_results", #for ray <= 2.5
                    #storage_path                = "~/ray_results", #required if not using remote storage for ray 2.6
                    stop                        = stopper,
                    sync_config                 = tune.SyncConfig(syncer = None), #for single-node or shared checkpoint dir, ray 2.5
                    #sync_config                 = tune.SyncConfig(syncer = None, upload_dir = None), #for single-node or shared checkpoint dir, ray 2.6
                    checkpoint_config           = air.CheckpointConfig(
                                                    checkpoint_frequency        = chkpt_int,
                                                    checkpoint_score_attribute  = "episode_reward_mean",
                                                    num_to_keep                 = 2, #if > 1 hard to tell which one is the best
                                                    checkpoint_at_end           = True
                    )
                )

    # Execute the HP tuning job, beginning with a previous checkpoint, if one was specified in the CdaCallbacks.
    tuner = Tuner(algo, param_space = cfg.to_dict(), tune_config = tune_config, run_config = run_config)
    #tuner = Tuner.restore(path="/home/starkj/ray_results/cda0")
    print("\n///// Tuner created.\n")

    result = tuner.fit()
    best_result = result.get_best_result(metric = "episode_reward_mean", mode = "max")
    print("\n///// Fit completed...")
    print("      Best checkpoint is ", best_result.checkpoint)
    print("      Best metrics: ", pretty_print(best_result.metrics))

    ray.shutdown()


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
