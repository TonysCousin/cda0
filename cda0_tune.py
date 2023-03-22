import sys
import ray
from ray import air, tune
import ray.rllib.algorithms.ppo as ppo
from ray.tune import Tuner
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.tune_config import TuneConfig
from ray.tune.logger import pretty_print
from ray.air import RunConfig
#import ray.rllib.algorithms.a2c as a2c
#import ray.rllib.algorithms.sac as sac
#import ray.rllib.algorithms.ddpg as ddpg

from stop_logic import StopLogic
#from stop_long  import StopLong
from simple_highway_ramp_wrapper import SimpleHighwayRampWrapper
from simple_highway_with_ramp import curriculum_fn
from cda_callbacks import CdaCallbacks

"""This program tunes (explores) hyperparameters to find a good set suitable for training.
    Usage is:
        cda0_tune <difficulty_level>
    If a difficulty level is not provided, it will default to 0.

    NOTE that this program will begin training the existing (pre-trained) checkpoint that is hard-coded in
    the CdaCallbacks code (cda_callbacks.py). While hard-coding is not ideal, it is the only known way, at
    this time, to pass the information into RLlib.
"""

def main(argv):

    difficulty_level = 0
    if len(argv) > 1:
        difficulty_level = min(max(int(argv[1]), 0), SimpleHighwayRampWrapper.NUM_DIFFICULTY_LEVELS)
    print("\n///// Tuning with environment difficulty level {}".format(difficulty_level))

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
    success_threshold   = [9.5,         9.5,        9.5,        9.0,        1.0]
    failure_threshold   = [6.0,         6.0,        6.0,        5.0,       -10.0]
    let_it_run          = False #can be a scalar or list of same size as above lists
    burn_in_period      = 70 #num iterations before we consider stopping or promoting to next level
    max_iterations      = 1000

    stopper = StopLogic(max_ep_timesteps        = 400,
                        max_iterations          = max_iterations,
                        min_timesteps           = min_timesteps,
                        avg_over_latest         = burn_in_period,
                        success_threshold       = success_threshold,
                        failure_threshold       = failure_threshold,
                        degrade_threshold       = 0.25,
                        compl_std_dev           = 0.05,
                        let_it_run              = let_it_run,
                    )

    # Define the custom environment for Ray
    env_config = {}
    env_config["difficulty_level"]              = difficulty_level
    env_config["stopper"]                       = stopper #object must be instantiated above
    env_config["burn_in_iters"]                 = burn_in_period
    env_config["time_step_size"]                = 0.5
    env_config["debug"]                         = 0
    env_config["training"]                      = True
    env_config["randomize_start_dist"]          = True
    env_config["neighbor_speed"]                = 29.1 #29.1 m/s is posted speed limit; only applies for appropriate diff levels
    env_config["neighbor_start_loc"]            = 320.0 #dist downtrack from beginning of lane 1 for n3, m
    #env_config["init_ego_lane"]                 = 0
    cfg.environment(env = SimpleHighwayRampWrapper, env_config = env_config, env_task_fn = curriculum_fn)

    # Add dict for model structure
    model_config = cfg_dict["model"]
    model_config["fcnet_hiddens"]               = [256, 128]
    model_config["fcnet_activation"]            = "relu"
    model_config["post_fcnet_activation"]       = "linear" #tune.choice(["linear", "tanh"])
    cfg.training(model = model_config)

    # Add exploration noise params
    explore_config = cfg_dict["exploration_config"]
    explore_config["type"]                      = "GaussianNoise" #default OrnsteinUhlenbeckNoise doesn't work well here
    explore_config["stddev"]                    = tune.uniform(0.18, 0.4) #this param is specific to GaussianNoise
    explore_config["random_timesteps"]          = 0 #tune.qrandint(0, 20000, 50000) #was 20000
    explore_config["initial_scale"]             = 1.0
    explore_config["final_scale"]               = 0.2 #tune.choice([1.0, 0.01])
    explore_config["scale_timesteps"]           = 1200000  #tune.choice([100000, 400000]) #was 900k
    cfg.exploration(explore = True, exploration_config = explore_config)

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

    cfg.rollouts(   num_rollout_workers         = 4, #num remote workers _per trial_ (remember that there is a local worker also)
                                                     # 0 forces rollouts to be done by local worker
                    num_envs_per_worker         = 1,
                    rollout_fragment_length     = 256, #timesteps pulled from a sampler
                    batch_mode                  = "complete_episodes",
                    recreate_failed_workers     = True,
    )

    # Training algorithm HPs
    # NOTE: lr_schedule is only defined for policy gradient algos
    # NOTE: all items below lr_schedule are PPO-specific
    cfg.training(   gamma                       = 0.999, #tune.choice([0.99, 0.999, 0.9999]),
                    train_batch_size            = 1024, #must be = rollout_fragment_length * num_rollout_workers * num_envs_per_worker
                    lr                          = tune.loguniform(1e-5, 3e-4),
                    #lr_schedule                 = [[0, 1.0e-4], [1600000, 1.0e-4], [1700000, 1.0e-5], [7000000, 1.0e-6]],
                    sgd_minibatch_size          = 64, #must be <= train_batch_size (and divide into it)
                    entropy_coeff               = tune.uniform(0.001, 0.004),
                    kl_coeff                    = tune.uniform(0.35, 0.8),
                    #clip_actions                = True,
                    clip_param                  = tune.uniform(0.1, 0.3),
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

    # ===== Final setup =========================================================================

    #print("\n///// {} training params are:\n".format(algo))
    #print(pretty_print(cfg.to_dict()))

    chkpt_int                                   = 10                    #num iters between storing new checkpoints
    perturb_int                                 = 50                    #num iters between policy perturbations (must be a multiple of chkpt period)

    scheduler = PopulationBasedTraining(
                    time_attr                   = "training_iteration", #type of interval for considering trial continuation
                    metric                      = "episode_reward_mean",#duplicate the TuneConfig setup - this is the metric used to rank trials
                    mode                        = "max",                #duplicate the TuneConfig setup - looking for largest metric
                    perturbation_interval       = perturb_int,          #number of iterations between continuation decisions on each trial
                    burn_in_period              = burn_in_period,       #num initial iterations before any perturbations occur
                    quantile_fraction           = 0.5,                  #fraction of trials to keep; must be in [0, 0.5]
                    resample_probability        = 0.5,                  #resampling and mutation probability at each decision point
                    synch                       = False,                #True:  all trials must finish before each perturbation decision is made
                                                                        # if any trial errors then all trials hang!
                                                                        #False:  each trial finishes & decides based on available info at that time,
                                                                        # then immediately moves on. If True and one trial dies, then PBT hangs and all
                                                                        # remaining trials go into perpetual PAUSED state.
                    hyperparam_mutations={                              #resample distributions
                        "lr"                        :   tune.loguniform(1e-6, 1e-3),
                        #"exploration_config/stddev" :   tune.uniform(0.2, 0.7), #PPOConfig doesn't recognize any variation on this key
                        "kl_coeff"                  :   tune.uniform(0.35, 0.8),
                        "entropy_coeff"             :   tune.uniform(0.0, 0.008),
                    },
    )

    tune_config = TuneConfig(
                    metric                      = "episode_reward_mean",
                    mode                        = "max",
                    #scheduler                   = scheduler,
                    num_samples                 = 16 #number of HP trials
                    #max_concurrent_trials      = 8
                )

    run_config = RunConfig(
                    name                        = "cda0",
                    local_dir                   = "~/ray_results",
                    #stop                        = stopper,
                    stop                        = {"episode_reward_min":        failure_threshold[difficulty_level],
                                                   "episode_reward_mean":       success_threshold[difficulty_level],
                                                   "training_iteration":        max_iterations,
                                                   },
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
