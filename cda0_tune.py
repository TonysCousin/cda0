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

"""This program tunes (explores) hyperparameters to find a good set suitable for training.
    Usage is:
        cda0_tune [checkpoint]
    If a checkpoint directory is provided, it will use the learning algorithm & NN state stored there as a starting point.
    Note that for this to work, the algo and NN structure specified below need to be identical to those recorded in the
    checkpoint.

    If no checkpoint is specified, then this program will create a new NN with the structure specified below, and begin
    training it anew.
"""

def main(argv):

    checkpoint = None
    if len(argv) > 1:
        checkpoint = argv[1]

    ray.init()

    # Define which learning algorithm we will use and set up is default config params
    algo = "PPO"
    cfg = ppo.PPOConfig()
    cfg.framework("torch")
    cfg_dict = cfg.to_dict()

    # Define the stopper object that decides when to terminate training.
    # All list objects (min_timesteps, success_threshold, failure_threshold) must be of length equal to num phases in use.
    # let_it_run can be a single value if it applies to all phases.
    # Phase...............0             1           2           3
    min_timesteps       = [1500000,     1000000,    1500000,    2000000]
    success_threshold   = [5.0,         5.0,        5.0,        1.0]
    failure_threshold   = [0.0,         0.0,        -5.0,       -10.0]
    let_it_run          = False #can be a scalar or list of same size as above lists
    stopper = StopLogic(max_ep_timesteps        = 400,
                        max_iterations          = 3000,
                        min_timesteps           = min_timesteps,
                        avg_over_latest         = 70,
                        success_threshold       = success_threshold,
                        failure_threshold       = failure_threshold,
                        degrade_threshold       = 0.25,
                        compl_std_dev           = 0.05,
                        let_it_run              = let_it_run,
                    )

    # Define the custom environment for Ray
    env_config = {}
    env_config["stopper"]                       = stopper #object must be instantiated above
    env_config["time_step_size"]                = 0.5
    env_config["debug"]                         = 0
    env_config["training"]                      = True
    env_config["randomize_start_dist"]          = True #tune.choice([True, False])
    env_config["neighbor_speed"]                = 29.1 #29.1 m/s is posted speed limit
    env_config["neighbor_start_loc"]            = 320.0 #dist downtrack from beginning of lane 1 for n3, m
    #env_config["init_ego_lane"]                 = 0
    cfg.environment(env = SimpleHighwayRampWrapper, env_config = env_config, env_task_fn = curriculum_fn)

    # Add dict for model structure
    model_config = cfg_dict["model"]
    model_config["fcnet_hiddens"]               = [128, 50]
    model_config["fcnet_activation"]            = "relu"
    model_config["post_fcnet_activation"]       = tune.choice(["linear", "tanh"])
    cfg.training(model = model_config)

    # Add exploration noise params
    explore_config = cfg_dict["exploration_config"]
    explore_config["type"]                      = "GaussianNoise" #default OrnsteinUhlenbeckNoise doesn't work well here
    explore_config["stddev"]                    = tune.uniform(0.2, 0.75) #this param is specific to GaussianNoise
    explore_config["random_timesteps"]          = 0 #tune.qrandint(0, 20000, 50000) #was 20000
    explore_config["initial_scale"]             = 1.0
    explore_config["final_scale"]               = 0.1 #tune.choice([1.0, 0.01])
    explore_config["scale_timesteps"]           = 5000000  #tune.choice([100000, 400000]) #was 900k
    cfg.exploration(explore = True, exploration_config = explore_config)

    # Computing resources
    cfg.resources(  num_gpus                    = 0, #for the local worker, which does the evaluation runs
                    num_cpus_for_local_worker   = 1,
                    num_cpus_per_worker         = 1, #also applies to the local worker and evaluation workers
                    num_gpus_per_worker         = 0  #this has to allow gpu left over for local worker & evaluation workers also
    )

    cfg.rollouts(   num_rollout_workers         = 1, #num remote workers (remember that there is a local worker also)
                    num_envs_per_worker         = 1,
                    rollout_fragment_length     = 200, #timesteps pulled from a sampler
                    batch_mode                  = "complete_episodes",
    )

    # Training algorithm HPs
    # NOTE: lr_schedule is only defined for policy gradient algos
    # NOTE: all items below lr_schedule are PPO-specific
    cfg.training(   gamma                       = tune.choice([0.99, 0.999, 0.9999]),
                    train_batch_size            = 200, #must be = rollout_fragment_length * num_rollout_workers * num_envs_per_worker
                    lr                          = tune.loguniform(1e-6, 1e-3),
                    #lr_schedule                 = [[0, 1.0e-4], [1600000, 1.0e-4], [1700000, 1.0e-5], [7000000, 1.0e-6]],
                    sgd_minibatch_size          = 32, #must be <= train_batch_size (and divide into it)
                    #grad_clip                  = tune.uniform(0.1, 0.5),
                    #clip_param                 = None #tune.choice([0.2, 0.3, 0.6, 1.0]),
    )

    # Evaluation process
    cfg.evaluation( evaluation_interval         = 6, #iterations
                    evaluation_duration         = 6, #units specified next
                    evaluation_duration_unit    = "episodes",
                    evaluation_parallel_to_training = False, #True requires evaluation_num_workers > 0
                    evaluation_num_workers      = 0,
    )

    # Debugging assistance
    cfg.debugging(  log_level                   = "WARN",
                    seed                        = 17, #tune.choice([2, 17, 666, 4334, 10003, 29771, 38710, 50848, 81199])
    )

    # checkpoint behavior
    #cfg.checkpointing(export_native_model_files = True) #raw torch NN weights will be stored in checkpoints

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

    # ===== Final setup =========================================================================

    print("\n///// {} training params are:\n".format(algo))
    print(pretty_print(cfg.to_dict()))

    chkpt_int                                   = 10                    #num iters between checkpoints
    perturb_int                                 = 20                   #num iters between policy perturbations (must be a multiple of chkpt period)

    scheduler = PopulationBasedTraining(
                    time_attr                   = "training_iteration", #type of interval for considering trial continuation
                    perturbation_interval       = perturb_int,          #number of iterations between continuation decisions on each trial
                    quantile_fraction           = 0.5,                  #fraction of trials to keep; must be in [0, 0.5]
                    resample_probability        = 0.5,                  #resampling and mutation probability at each decision point
                    synch                       = True,                 #True:  all trials must finish before a checkpoint/perturbation decision is made
                                                                        #False:  each trial finishes & decides based on available info at that time,
                                                                        # then immediately moves on
                    hyperparam_mutations={                              #resample distributions
                        "lr"                        :   tune.loguniform(1e-6, 1e-3),
                        "exploration_config/stddev" :   tune.uniform(0.2, 0.7),
                    },
    )

    tune_config = TuneConfig(
                    metric                      = "episode_reward_mean",
                    mode                        = "max",
                    scheduler                   = scheduler,
                    num_samples                 = 16 #number of HP trials
                )

    run_config = RunConfig(
                    name                        = "cda0",
                    local_dir                   = "~/ray_results",
                    stop                        = stopper,
                    sync_config                 = tune.SyncConfig(syncer = None), #for single-node or shared checkpoint dir
                    checkpoint_config           = air.CheckpointConfig(
                                                    checkpoint_frequency        = chkpt_int,
                                                    checkpoint_score_attribute  = "episode_reward_mean",
                                                    num_to_keep                 = 1, #if > 1 hard to tell which one is the best
                                                    checkpoint_at_end           = True
                    )
                )

    # Execute the HP tuning job, beginning with a previous checkpoint, if one was specified
    tuner = Tuner(algo, param_space = cfg.to_dict(), tune_config = tune_config, run_config = run_config)
    print("\n///// Tuner created.\n")

    if checkpoint != None:
        tuner.restore(checkpoint)
        print("\n///// Successfully loaded starting checkpoint: {}".format(checkpoint))



    result = tuner.fit()
    #print("\n///// tuner.fit() returned: ", type(result), " - ", result[0]) #we should only look at result[0] for some reason?
    print(pretty_print(result))

    ray.shutdown()


######################################################################################################
######################################################################################################

if __name__ == "__main__":
   main(sys.argv)
