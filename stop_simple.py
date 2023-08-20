from collections import deque
from concurrent.futures.process import _threads_wakeups
import math
from statistics import mean, stdev
from ray.tune import Stopper
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

class StopSimple(Stopper):
    """Determines when to stop a trial, based on the running average of the mean reward (not just the "instantaneous" value
        for a single iteration).
        NOTE: if both avg_threshold and min_threshold are specified, then the trial has to meet both of those conditions
        simultaneously, over the specified period in order to consider it successful.
    """

    def __init__(self,
                 max_iterations     : int   = None,     #max total iterations alloweed over all phases combined
                 avg_over_latest    : int   = 10,       #num most recent iterations over which statistics will be calculated
                 avg_threshold      : float = 1.0,      #mean reward above which we can call it a win
                 min_threshold      : float = None,     #min reward above which we can call it a win
                 max_fail_threshold : float = None,     #max reward below which we can say it's a clear failure
                 burn_in            : int   = 0,        #num iterations before considering a failure stop
                ):

        self.max_iterations = max_iterations
        self.most_recent = avg_over_latest #num recent trials to consider when deciding to stop
        self.success_avg_threshold = avg_threshold
        self.success_min_threshold = min_threshold
        self.failure_max_threshold = max_fail_threshold
        self.burn_in = burn_in
        self.trials = {}


    def __call__(self,
                    trial_id    : str,  #ID of the trial being considered for stopping
                    result      : dict  #collection of results of the trial so far
                ) -> bool:              #return: should the trial be terminated?

        """ Will be called after each iteration to decide if learning should be stopped for this trial."""

        # Determine the total iteration count
        total_iters = result["training_iteration"] #was "iterations_since_restore"

        # If this trial is already underway and being tracked, then
        if trial_id in self.trials:

            # Capture the values of max, min and mean rewards for this iteration
            ep_mean = result["episode_reward_mean"]
            mean_rew = -100.0
            if not math.isnan(ep_mean):
                mean_rew = ep_mean
            self.trials[trial_id]["mean_rewards"].append(mean_rew)

            max_rew = -100.0
            if not math.isnan(result["episode_reward_max"]):
                max_rew = result["episode_reward_max"]
            self.trials[trial_id]["max_rewards"].append(max_rew)

            min_rew = -100.0
            if not math.isnan(result["episode_reward_min"]):
                min_rew = result["episode_reward_min"]
            self.trials[trial_id]["min_rewards"].append(min_rew)

            if ep_mean < self.trials[trial_id]["worst_mean"]:
                self.trials[trial_id]["worst_mean"] = ep_mean
            if ep_mean > self.trials[trial_id]["best_mean"]:
                self.trials[trial_id]["best_mean"] = ep_mean

            # If the deque of N most recent rewards is not yet full then increment its count
            if self.trials[trial_id]["num_entries"] < self.most_recent:
                self.trials[trial_id]["num_entries"] += 1

            # Else the deque is full so we can start analyzing stop criteria
            else:
                # If avg of mean rewards over recent history is above the succcess threshold
                avg_of_mean = mean(self.trials[trial_id]["mean_rewards"])
                if avg_of_mean >= self.success_avg_threshold:

                    # If min threshold is defined then if it is exceeded also call it a success
                    if self.success_min_threshold is not None:
                        avg_of_min = mean(self.trials[trial_id]["min_rewards"])
                        if avg_of_min >= self.success_min_threshold:
                            print("\n///// Stopping trial - SUCCESS!  Recent rmean = {}, rmin = {}\n".format(avg_of_mean, avg_of_min))
                            return True

                    # Else, no min threshold required so it is a success
                    else:
                        print("\n///// Stopping trial - SUCCESS!  Recent rmean = {}\n".format(avg_of_mean))
                        return True

                # Fail if avg of max rewards over recent history is below the failure threshold
                if self.failure_max_threshold is not None  and  total_iters >= self.burn_in:
                    avg_of_max = mean(self.trials[trial_id]["max_rewards"])
                    if avg_of_max < self.failure_max_threshold:
                        print("\n///// Stopping trial - FAILURE!  Recent rmax = {}\n".format(avg_of_max))
                        return True

                # Stop if max iterations reached
                if self.max_iterations is not None  and  total_iters >= self.max_iterations:
                    print("\n///// Stopping trial - reached max iterations.")
                    return True

        # Else, it is a brand new trial
        else:
            mean_rew = deque(maxlen = self.most_recent)
            mean_rew.append(result["episode_reward_mean"])
            max_rew = deque(maxlen = self.most_recent)
            max_rew.append(result["episode_reward_max"])
            min_rew = deque(maxlen = self.most_recent)
            min_rew.append(result["episode_reward_min"])
            self.trials[trial_id] = {"stop": False, "num_entries": 1, \
                                     "mean_rewards": mean_rew, "max_rewards": max_rew, "min_rewards": min_rew, "worst_mean": math.inf,
                                     "best_mean": -math.inf}
            print("///// StopLogic adding new trial: {}".format(trial_id))

        return False


    def set_environment_model(self,
                              env: TaskSettableEnv  #a reference to the environment model
                             ) -> None:
        """This is required to be called by the environment if multi-phase learning is to be done."""

        pass


    def get_success_thresholds(self) -> list:
        """Returns a list of the success thresholds defined for the various phases."""
        if type(self.success_avg_threshold) == list:
            return self.success_avg_threshold
        else:
            return [self.success_avg_threshold]


    """Not sure when this is called"""
    def stop_all(self):
        #print("\n\n///// In StopLogic.stop_all /////\n")
        return False
