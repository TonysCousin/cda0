from collections import deque
from concurrent.futures.process import _threads_wakeups
import math
from statistics import mean, stdev
from ray.tune import Stopper
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

class StopSimple(Stopper):
    """Determines when to stop a trial, based on the running average of the mean reward (not just the "instantaneous" value
        for a single iteration).
    """

    def __init__(self,
                 max_iterations     : int   = None,     #max total iterations alloweed over all phases combined
                 avg_over_latest    : int   = 10,       #num most recent iterations over which statistics will be calculated
                 success_threshold  : float = 1.0,      #reward above which we can call it a win; use a list for multi-phase
                ):

        self.max_iterations = max_iterations
        self.most_recent = avg_over_latest #num recent trials to consider when deciding to stop
        self.success_avg_threshold = success_threshold
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
                # Stop if avg of mean rewards over recent history is above the succcess threshold
                avg_of_mean = mean(self.trials[trial_id]["mean_rewards"])
                if avg_of_mean >= self.success_avg_threshold:
                    print("\n///// Stopping trial - SUCCESS!\n")
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