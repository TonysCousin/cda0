import sys
import ray
from ray import air, tune
from ray.tune import Tuner
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.tune_config import TuneConfig
from ray.tune.logger import pretty_print
from ray.air import RunConfig
import ray.rllib.algorithms.ppo as ppo
#import ray.rllib.algorithms.a2c as a2c
#import ray.rllib.algorithms.sac as sac
#import ray.rllib.algorithms.ddpg as ddpg

from stop_simple import StopSimple
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper
from simple_highway_with_ramp import curriculum_fn
from cda_callbacks import CdaCallbacks
from perturbation_control import PerturbationController

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


def main(argv):

    difficulty_level = 0
    if len(argv) > 1:
        difficulty_level = min(max(int(argv[1]), 0), SimpleHighwayRampWrapper.NUM_DIFFICULTY_LEVELS)
    print("\n///// Tuning with initial environment difficulty level {}".format(difficulty_level))

    ray.init()

    # Define which learning algorithm we will use and set up is default config params
    algo = "PPO"
    cfg = ppo.PPOConfig()
    cfg.framework("torch")
    cfg_dict = cfg.to_dict()

    # Define the stopper object that decides when to terminate training.
    # All list objects (min_timesteps, success_threshold, failure_threshold) must be of length equal to num phases in use.
    # let_it_run can be a single value if it applies to all phases.
    # Phase...............0             1           2           3           4
    min_timesteps       = [1500000,     1500000,    1000000,    1500000,    2000000]
    success_threshold   = [9.5,         9.5,        9.5,        15.0,        9.5]
    failure_threshold   = [6.0,         6.0,        6.0,        6.0,        6.0]
    let_it_run          = False #can be a scalar or list of same size as above lists
    chkpt_int           = 10    #num iters between storing new checkpoints
    burn_in_period      = 500   #num iterations before we consider stopping or promoting to next level
    perturb_int         = 200   #num iterations between perturbations (after burn-in period); must be multiple of chkpt_int
    max_iterations      = 1000
    num_trials          = 10

    # Set up a communication path with the CdaCallbacks to properly control PBT perturbation cycles
    PerturbationController(_checkpoint_path, num_trials)

    # Define the stopping logic for PBT runs - this requires mean reward to stay at the threshold for multiple consiecutive
    # iterations, rather than just stopping on an outlier spike.
    stopper = StopSimple(max_iterations     = max_iterations,
                         avg_over_latest    = 20,
                         success_threshold  = success_threshold[difficulty_level]
                        )

    # Define the custom environment for Ray
    env_config = {}
    env_config["difficulty_level"]              = difficulty_level
    #env_config["stopper"]                       = stopper #object must be instantiated above
    env_config["burn_in_iters"]                 = burn_in_period
    env_config["time_step_size"]                = 0.5
    env_config["debug"]                         = 0
    env_config["verify_obs"]                    = False
    env_config["training"]                      = True
    env_config["randomize_start_dist"]          = False
    env_config["neighbor_speed"]                = 29.1 #29.1 m/s is posted speed limit; only applies for appropriate diff levels
    env_config["neighbor_start_loc"]            = 0.0 #dist downtrack from beginning of lane 1 for n3, m
    #env_config["init_ego_lane"]                 = 0
    cfg.environment(env = SimpleHighwayRampWrapper, env_config = env_config, env_task_fn = curriculum_fn)

    # Add exploration noise params
    explore_config = cfg_dict["exploration_config"]
    #print("///// Explore config:\n", pretty_print(explore_config))
    explore_config["type"]                      = "GaussianNoise" #default OrnsteinUhlenbeckNoise doesn't work well here
    explore_config["stddev"]                    = tune.uniform(0.1, 0.5) #this param is specific to GaussianNoise
    explore_config["random_timesteps"]          = 0 #tune.qrandint(0, 20000, 50000) #was 20000
    explore_config["initial_scale"]             = 1.0
    explore_config["final_scale"]               = 0.1 #tune.choice([1.0, 0.01])
    explore_config["scale_timesteps"]           = 500000  #tune.choice([100000, 400000]) #was 900k
    cfg.exploration(explore = True, exploration_config = explore_config)
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
                    num_cpus_for_local_worker   = 1,
                    num_cpus_per_trainer_worker = 1, #also applies to the local worker and evaluation workers
                    num_gpus_per_trainer_worker = 0  #this has to allow gpu left over for local worker & evaluation workers also
    )

    cfg.rollouts(   #num_rollout_workers         = 1, #num remote workers _per trial_ (remember that there is a local worker also)
                                                     # 0 forces rollouts to be done by local worker
                    num_envs_per_worker         = 1,
                    rollout_fragment_length     = 256, #timesteps pulled from a sampler
                    batch_mode                  = "complete_episodes",
                    recreate_failed_workers     = True,
    )

    # Evaluation process
    cfg.evaluation( evaluation_interval         = 10, #iterations
                    evaluation_duration         = 15, #units specified next
                    evaluation_duration_unit    = "episodes",
                    evaluation_parallel_to_training = True, #True requires evaluation_num_workers > 0
                    evaluation_num_workers      = 2,
    )

    # Debugging assistance
    cfg.debugging(  log_level                   = "WARN",
                    seed                        = 17, #tune.choice([2, 17, 666, 4334, 10003, 29771, 38710, 50848, 81199])
    )

    # Custom callbacks from the training algorithm
    cfg.callbacks(  CdaCallbacks)

    # ===== Training algorithm HPs for PPO =================================================

    # NOTE: lr_schedule is only defined for policy gradient algos
    # NOTE: all items below lr_schedule are PPO-specific
    cfg.training(   gamma                       = 0.999, #tune.choice([0.99, 0.999, 0.9999]),
                    train_batch_size            = 1024, #must be an int multiple of rollout_fragment_length * num_rollout_workers * num_envs_per_worker
                    lr                          = tune.loguniform(1e-6, 2e-4),
                    #lr_schedule                 = [[0, 1.0e-4], [1600000, 1.0e-4], [1700000, 1.0e-5], [7000000, 1.0e-6]],
                    sgd_minibatch_size          = 128, #must be <= train_batch_size (and divide into it)
                    entropy_coeff               = tune.uniform(0.0005, 0.01),
                    kl_coeff                    = tune.uniform(0.3, 0.8),
                    #clip_actions                = True,
                    clip_param                  = tune.uniform(0.05, 0.4),
                    grad_clip                   = tune.uniform(10, 40),
    )

    # Add dict for model structure
    model_config = cfg_dict["model"]
    model_config["fcnet_hiddens"]               = [256, 256, 128]
    model_config["fcnet_activation"]            = "relu"
    cfg.training(model = model_config)

    """
    # ===== Training algorithm HPs for SAC ==================================================

    opt_config = cfg_dict["optimization"]
    opt_config["actor_learning_rate"]           = tune.loguniform(1e-6, 1e-3) #default 0.0003
    opt_config["critic_learning_rate"]          = tune.loguniform(1e-6, 1e-3) #default 0.0003
    opt_config["entropy_learning_rate"]         = tune.loguniform(1e-4, 1e-3) #default 0.0003

    policy_config = cfg_dict["policy_model_config"]
    policy_config["fcnet_hiddens"]              = [256, 128]
    policy_config["fcnet_activation"]           = "relu"

    q_config = cfg_dict["q_model_config"]
    q_config["fcnet_hiddens"]                   = [256, 128]
    q_config["fcnet_activation"]                = "relu"

    cfg.training(   twin_q                      = True,
                    gamma                       = 0.999,
                    train_batch_size            = 256, #must be an int multiple of rollout_fragment_length * num_rollout_workers * num_envs_per_worker
                    initial_alpha               = 1.0,
                    tau                         = 0.005,
                    n_step                      = 1,
                    optimization_config         = opt_config,
                    policy_model_config         = policy_config,
                    q_model_config              = q_config,
    )
    """

    # ===== Final setup =========================================================================

    #print("\n///// {} training params are:\n".format(algo))
    #print(pretty_print(cfg.to_dict()))

    scheduler = PopulationBasedTraining(
                    time_attr                   = "training_iteration", #type of interval for considering trial continuation
                    #metric                      = "episode_reward_mean",#duplicate the TuneConfig setup - this is the metric used to rank trials
                    #mode                        = "max",                #duplicate the TuneConfig setup - looking for largest metric
                    perturbation_interval       = perturb_int,          #number of iterations between continuation decisions on each trial
                    burn_in_period              = burn_in_period,       #num initial iterations before any perturbations occur
                    quantile_fraction           = 0.5,                  #fraction of trials to keep; must be in [0, 0.5]
                    resample_probability        = 0.5,                  #resampling vs mutation probability at each decision point
                    synch                       = False,                #True:  all trials must finish before each perturbation decision is made
                                                                        # if any trial errors then all trials hang!
                                                                        #False:  each trial finishes & decides based on available info at that time,
                                                                        # then immediately moves on. If True and one trial dies, then PBT hangs and all
                                                                        # remaining trials go into perpetual PAUSED state.
                    hyperparam_mutations={                              #resample distributions
                    #    "optimization/actor_learning_rate"       :   tune.loguniform(1e-6, 1e-3),
                    #    "optimization/critic_learning_rate"      :   tune.loguniform(1e-6, 1e-3),
                    #    "optimization/entropy_learning_rate"     :   tune.loguniform(1e-4, 1e-3),
                        "lr"                                    :   tune.loguniform(1e-6, 2e-4),
                        "entropy_coeff"                         :   tune.uniform(0.0005, 0.008),
                    #    "kl_coeff"                              :   tune.uniform(0.3, 0.8),
                        "clip_param"                            :   tune.uniform(0.05, 0.4),
                    },
    )

    tune_config = TuneConfig(
                    metric                      = "episode_reward_mean",
                    mode                        = "max",
                    scheduler                   = scheduler,
                    num_samples                 = num_trials,
                )

    run_config = RunConfig(
                    name                        = "cda0",
                    local_dir                   = "~/ray_results",
                    stop                        = stopper,
                    #stop                        = {"episode_reward_mean":       success_threshold[difficulty_level],
                    #                               "training_iteration":        max_iterations,
                    #                               },
                    sync_config                 = tune.SyncConfig(syncer = None), #for single-node or shared checkpoint dir
                    verbose                     = 3, #3 is default
                    checkpoint_config           = air.CheckpointConfig(
                                                    checkpoint_frequency        = chkpt_int,
                                                    checkpoint_score_attribute  = "episode_reward_mean",
                                                    num_to_keep                 = 1, #if > 1 hard to tell which one is the best
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
