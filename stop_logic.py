from collections import deque
import math
from statistics import mean, stdev
from ray.tune import Stopper

#TODO: consider adding a condition where eval mean reward starts to drop, as long as its peak was above
#the acceptable threshold, and as long as the training mean reward is/was also above threshold
#TODO: add condition for eval mean reward below 0
#TODO: add condition where either training or eval mean reward continues to drop significantly below winning threshold

# Trial stopping decision
class StopLogic(Stopper):
    def __init__(self,
                 max_timesteps      : int   = None,
                 max_iterations     : int   = None,
                 min_iterations     : int   = 0,        #num iterations that must be completed before considering a stop
                 avg_over_latest    : int   = 10,
                 success_threshold  : float = 1.0,      #reward above which we can call it a win
                 failure_threshold  : float = -1.0,     #reward below which we can early terminate (after min required iters)
                 compl_std_dev      : float = 0.01      #std deviation of reward below which we can terminate (success or failure)
                ):

        self.max_iterations = None
        if max_iterations is not None:
            if max_iterations > 1.2*min_iterations:
                self.max_iterations = max_iterations
            else:
                self.max_iterations = 1.2*min_iterations

        self.max_timesteps = max_timesteps
        self.required_min_iters = min_iterations
        self.most_recent = avg_over_latest #num recent trials to consider when deciding to stop
        self.success_avg_threshold = success_threshold
        self.failure_avg_threshold = failure_threshold
        self.completion_std_threshold = compl_std_dev

        # Each entry will have key = trial_id and value = dict containing the following:
        #   "stop" (bool) - should this trial be stopped?
        #   "num_entries" (int) - num meaningful entries in the deque
        #   "rewards" (deque) - the most recent N mean rewards
        self.trials = {}


    def __call__(self,
                    trial_id    : str,  #ID of the trial being considered for stopping
                    result      : dict  #collection of results of the trial so far
                ) -> bool:              #return: should the trial be terminated?

        """ Will be called after each iteration to decide if learning should be stopped for this trial."""

        #print("\n///// StopLogic - result = ")
        #for item in result:
        #    print("{}: {}".format(item, result[item]))
        #print("///// - end of result\n")
        total_iters = result["iterations_since_restore"]
        threshold = self.required_min_iters
        if result["episode_reward_max"] > 10.0: #TODO: make this a config var or input arg
            threshold *= 1.2

        # If this trial is already underway and being tracked, then
        if trial_id in self.trials:
            rew = -100.0
            if not math.isnan(result["episode_reward_mean"]):
                rew = result["episode_reward_mean"]
            self.trials[trial_id]["rewards"].append(rew)
            #print("///// Appending reward ", rew)

            if self.trials[trial_id]["num_entries"] < self.most_recent:
                self.trials[trial_id]["num_entries"] += 1
                #print("\n///// StopLogic: trial {} has completed {} iterations.".format(trial_id, self.trials[trial_id]["num_entries"]))

            elif total_iters > threshold:

                if self.max_iterations is not None  and  total_iters >= self.max_iterations:
                    print("\n///// Stopping trial - reached max iterations.")
                    return True

                if self.max_timesteps is not None  and  result["timesteps_since_restore"] >= self.max_timesteps:
                    print("\n///// Stopping trial - reached max timesteps.")
                    print("")
                    return True

                std = stdev(self.trials[trial_id]["rewards"])
                if std <= self.completion_std_threshold:
                    print("\n///// Stopping trial due to no further change")
                    return True

                avg = mean(list(self.trials[trial_id]["rewards"]))
                #print("\n///// StopLogic: trial {} has recent avg reward = {:.2f}".format(trial_id, avg))
                if avg > self.success_avg_threshold: #success!
                    print("\n///// Stopping trial due to success!\n")
                    return True
                elif avg < self.failure_avg_threshold: #failure
                    print("\n///// Stopping trial due to failure\n")
                    return True

        # Else, it is a brand new trial
        else:
            init = deque(maxlen=self.most_recent)
            init.append(result["episode_reward_mean"])
            self.trials[trial_id] = {"stop": False, "num_entries": 1, "rewards": init}
            #print("\n///// Creating new trial entry: ", self.trials)

        return False

    """Not sure when this is called"""
    def stop_all(self):
        #print("\n\n///// In StopLogic.stop_all /////\n")
        return False
