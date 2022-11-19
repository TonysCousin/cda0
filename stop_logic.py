from collections import deque
from concurrent.futures.process import _threads_wakeups
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
                 avg_over_latest    : int   = 10,       #num most recent iterations over which statistics will be calculated
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
        self.required_min_iters = min_iterations #num required before declaring failure
        self.most_recent = avg_over_latest #num recent trials to consider when deciding to stop
        self.success_avg_threshold = success_threshold
        self.failure_avg_threshold = failure_threshold
        self.completion_std_threshold = compl_std_dev
        print("\n///// StopLogic initialized with max_timesteps = {}, max_iterations = {}, min_iterations = {}"
                .format(self.max_timesteps, self.max_iterations, self.required_min_iters))
        print("      most_recent = {}, success_avg_thresh = {:.2f}, failure_avg_thresh = {:.2f}, compl_std_thresh ={:.3f}"
                .format(self.most_recent, self.success_avg_threshold, self.failure_avg_threshold, self.completion_std_threshold))

        # Each entry will have key = trial_id and value = dict containing the following:
        #   "stop" (bool) - should this trial be stopped?
        #   "num_entries" (int) - num meaningful entries in the deque
        #   "mean_rewards" (deque) - the most recent N mean rewards
        #   "max_rewards" (deque) - the most recent N max rewards
        self.trials = {}

        # Other internal variables
        self.threshold_latch = False


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
        if result["episode_reward_max"] > -0.4  and   not self.threshold_latch: #TODO: make this a config var or input arg
            self.required_min_iters *= 1.2
            self.threshold_latch = True

        # If this trial is already underway and being tracked, then
        if trial_id in self.trials:

            # Capture the values of max, min and mean rewards for this iteration
            mean_rew = -100.0
            if not math.isnan(result["episode_reward_mean"]):
                mean_rew = result["episode_reward_mean"]
            self.trials[trial_id]["mean_rewards"].append(mean_rew)
            max_rew = -100.0
            if not math.isnan(result["episode_reward_max"]):
                max_rew = result["episode_reward_max"]
            self.trials[trial_id]["max_rewards"].append(max_rew)
            min_rew = -100.0
            if not math.isnan(result["episode_reward_min"]):
                min_rew = result["episode_reward_min"]
            self.trials[trial_id]["min_rewards"].append(min_rew)
            #print("///// Appending reward ", mean_rew, max_rew)

            # If the deque of N most recent rewards is not yet full then increment its count
            if self.trials[trial_id]["num_entries"] < self.most_recent:
                self.trials[trial_id]["num_entries"] += 1
                #print("\n///// StopLogic: trial {} has completed {} iterations.".format(trial_id, self.trials[trial_id]["num_entries"]))

            # Else the deque is full so we can start analyzing stop criteria
            else:
                # Stop if avg of mean rewards over recent history is above the succcess threshold and its standard deviation is small
                avg_of_mean = mean(self.trials[trial_id]["mean_rewards"])
                std_of_mean = stdev(self.trials[trial_id]["mean_rewards"])
                #print("///// StopLogic: iter #{}, avg reward = {:.2f}, std of mean = {:.3f}".format(total_iters, avg, std_of_mean))
                if avg_of_mean >= self.success_avg_threshold  and  std_of_mean <= self.completion_std_threshold:
                    print("\n///// Stopping trial due to success!\n")
                    return True

                # If we have seen more iterations than the min required for failure then
                if total_iters > self.required_min_iters:

                    # Stop if max iterations reached
                    if self.max_iterations is not None  and  total_iters >= self.max_iterations:
                        print("\n///// Stopping trial - reached max iterations.")
                        return True

                    # Stop if max timesteps reached
                    if self.max_timesteps is not None  and  result["timesteps_since_restore"] >= self.max_timesteps:
                        print("\n///// Stopping trial - reached max timesteps.")
                        print("")
                        return True

                    # Stop if mean and max rewards haven't significantly changed in recent history
                    std_of_max  = stdev(self.trials[trial_id]["max_rewards"])
                    if std_of_mean <= self.completion_std_threshold  and  std_of_max <= self.completion_std_threshold \
                       and  avg_of_mean >= self.success_avg_threshold:
                        print("\n///// Stopping trial due to winner with no further change")
                        return True

                    # If the avg mean reward over recent history is below the failure threshold then
                    if avg_of_mean < self.failure_avg_threshold:
                        done = False

                        # If the max reward is below success threshold and not climbing significantly, then stop as a failure
                        dq_max = self.trials[trial_id]["max_rewards"]
                        avg_of_max = mean(dq_max)
                        slope_max = self._get_slope(dq_max)
                        if avg_of_max < self.success_avg_threshold:
                            if slope_max <= 0.0:
                                print("\n///// Stopping trial - no improvement in {} iters.\n".format(self.most_recent))
                                done = True

                        # If the mean curve is heading down and the max is not increasing then stop as a failure
                        if slope_max <= 0.0  and  self._get_slope(self.trials[trial_id]["mean_rewards"]) < 0.0:
                            print("\n///// Stopping trial - mean reward bad & getting worse, max is not improving in latest {} iters."
                                    .format(self.most_recent))
                            done = True

                        # If the mean is a lot closer to the min than to the max then stop as failure
                        avg_of_min = mean(list(self.trials[trial_id]["min_rewards"]))
                        if avg_of_mean - avg_of_min < 0.25*(avg_of_max - avg_of_min):
                            print("\n///// Stopping trial - no improvement and min reward is dominating in latest {} iters."
                                    .format(self.most_recent))
                            done = True

                        if done:
                            print("      means = ", self.trials[trial_id]["mean_rewards"])
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
                                     "mean_rewards": mean_rew, "max_rewards": max_rew, "min_rewards": min_rew}

        return False

    """Not sure when this is called"""
    def stop_all(self):
        #print("\n\n///// In StopLogic.stop_all /////\n")
        return False

    def _get_slope(self, dq):
        begin = 0.0
        end = 0.0
        dq_size = len(dq)
        if dq_size < 4:
            begin = dq[0]
            end = dq[-1]
        elif dq_size < 21:
            begin = 0.333333 * mean([dq[i] for i in range(3)])
            end   = 0.333333 * mean([dq[i] for i in range(-3, 0)])
        else:
            begin = 0.1 * mean([dq[i] for i in range(10)])
            end   = 0.1 * mean([dq[i] for i in range(-10, 0)])

        return end - begin
