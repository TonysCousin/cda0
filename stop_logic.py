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
    """Determines when to stop a trial, either for success or failure.  It allows some rudimentary curriculum learning
        by defining multiple phases, where each phase can have its own duration (in time steps), reward thresholds
        for success and failure, and stopping criteria.  If only a single phase is desired (no curriculum), then these
        quantities can bevspecified as scalars.  If a multi-phase curriculum is desired, then these 4 quantities must
        be specified with lists (each list containing an entry for each phase).
    """

    def __init__(self,
                 max_timesteps      : int   = None,
                 max_iterations     : int   = None,     #max total iterations alloweed over all phases combined
                 phase_end_steps    : int   = 0,        #can be a scalar, indicating no phases are being used (value ignored),
                                                        # or a list with each entry being the num (cumulative) time steps till
                                                        # the end of that phase. Final entry is ignored and assumed to be infinity.
                 min_timesteps      : int   = 0,        #num timesteps that must be completed before considering a stop;
                                                        # use a list for multi-phase; list values are cumulative
                 avg_over_latest    : int   = 10,       #num most recent iterations over which statistics will be calculated
                 success_threshold  : float = 1.0,      #reward above which we can call it a win; use a list for multi-phase
                 failure_threshold  : float = -1.0,     #reward below which we can early terminate (after min required timesteps);
                                                        # use a list for multi-phase
                 degrade_threshold  : float = 0.25,     #fraction of mean reward range below its peak that mean is allowed to degrade
                 compl_std_dev      : float = 0.01,     #std deviation of reward below which we can terminate (success or failure)
                 let_it_run         : bool  = False     #should we allow the trial to run to the max iterations, regardless of rewards?
                                                        # use a list for multi-phase or default for all phases is False
                ):

        # Check for proper multi-phase inputs
        self.num_phases = 1
        self.cur_phase = 0
        if type(phase_end_steps) == list:
            self.num_phases = len(phase_end_steps)
            assert type(min_timesteps) == list, "///// StopLogic: min_timesteps needs to be a list but is a scalar."
            assert type(success_threshold) == list, "///// StopLogic: success_threshold needs to be a list but is a scalar."
            assert type(failure_threshold) == list, "///// StopLogic: failure_threshold needs to be a list but is a scalar."
            assert len(min_timesteps) == self.num_phases, "///// StopLogic: min_timesteps list is different length from phase_end_steps."
            assert len(success_threshold) == self.num_phases, "///// StopLogic: success_threshold list is different length from phase_end_steps."
            assert len(failure_threshold) == self.num_phases, "///// StopLogic: failure_threshold list is different length from phase_end_steps."
            if type(let_it_run) == list:
                assert len(let_it_run) == self.num_phases, "///// StopLogic: let_it_run list is different length from phase_end_steps."
            else:
                let_it_run = [] * let_it_run
        else:
            phase_end_steps = math.inf


        self.max_timesteps = max_timesteps
        self.max_iterations = max_iterations
        self.phase_end_steps = phase_end_steps
        self.required_min_timesteps = min_timesteps #num required before declaring failure
        self.most_recent = avg_over_latest #num recent trials to consider when deciding to stop
        self.success_avg_threshold = success_threshold
        self.failure_avg_threshold = failure_threshold
        self.degrade_threshold = degrade_threshold
        self.completion_std_threshold = compl_std_dev
        self.let_it_run = let_it_run
        print("\n///// StopLogic initialized with max_timesteps = {}, max_iterations = {}, {} phases."
                .format(self.max_timesteps, self.max_iterations, self.num_phases))
        print("      phase_end_steps = {}".format(self.phase_end_steps))
        print("      min_timesteps   = {}".format(self.required_min_timesteps))
        print("      most_recent = {}, degrade_threshold = {:.2f}, compl_std_thresh = {:.3f}"
                .format(self.most_recent, self.degrade_threshold, self.completion_std_threshold))
        print("      success_avg_thresh = {}, failure_avg_thresh = {}, let_it_run = {}"
                .format(self.success_avg_threshold, self.failure_avg_threshold, self.let_it_run))

        # Each entry will have key = trial_id and value = dict containing the following:
        #   "stop" (bool) - should this trial be stopped?
        #   "num_entries" (int) - num meaningful entries in the deque
        #   "mean_rewards" (deque) - the most recent N mean rewards
        #   "max_rewards" (deque) - the most recent N max rewards
        #   "min_rewards" (deque) - the most recent N min rewards
        #   "worst_mean" (float) - the lowest mean reward achieved at any point in the trial so far
        #   "best_mean" (float) - the highest mean reward achieved at any point in the trial so far
        self.trials = {}

        # Other internal variables
        self.threshold_latch = False
        self.prev_phase = 0
        self.crossed_min_timesteps = False


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
        total_steps = result["timesteps_total"]

        # Determine if this is a multi-phase trial and what phase we are in, then assign local variables for the phase-dependent items
        phase = 0
        min_timesteps = 0
        success_avg_thresh = -math.inf
        failure_avg_thresh = -math.inf
        let_it_run = False
        if self.num_phases == 1:
            min_timesteps = self.required_min_timesteps
            success_avg_thresh = self.success_avg_threshold
            failure_avg_thresh = self.failure_avg_threshold
            let_it_run = self.let_it_run
        else:
            for item in self.phase_end_steps:
                if total_steps <= item:
                    break
                phase += 1
            phase = min(phase, len(self.phase_end_steps) - 1)
            if phase != self.prev_phase:
                print("///// StopLogic: Beginning phase {}".format(phase))
                self.crossed_min_timesteps = False
            self.prev_phase = phase
            min_timesteps = self.required_min_timesteps[phase]
            success_avg_thresh = self.success_avg_threshold[phase]
            failure_avg_thresh = self.failure_avg_threshold[phase]
            let_it_run = self.let_it_run[phase]

        # If we see a respectable reward at any point, then extend the guaranteed min timesteps for all phases (need to do this after
        # the phase counter has been evaluated so that we don't bounce back to a previous value)
        #if result["episode_reward_max"] > -0.4  and   not self.threshold_latch: #TODO: make this a config var or input arg
        #    min_timesteps *= 1.2
        #    if self.num_phases > 1:
        #        self.required_min_timesteps = [1.2*item for item in self.required_min_timesteps]
        #    self.threshold_latch = True

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
            #print("///// Appending reward ", mean_rew, max_rew)
            if ep_mean < self.trials[trial_id]["worst_mean"]:
                self.trials[trial_id]["worst_mean"] = ep_mean
            if ep_mean > self.trials[trial_id]["best_mean"]:
                self.trials[trial_id]["best_mean"] = ep_mean

            # If the deque of N most recent rewards is not yet full then increment its count
            if self.trials[trial_id]["num_entries"] < self.most_recent:
                self.trials[trial_id]["num_entries"] += 1
                #print("\n///// StopLogic: trial {} has completed {} iterations.".format(trial_id, self.trials[trial_id]["num_entries"]))

            # Else the deque is full so we can start analyzing stop criteria
            else:
                # Stop if we are in the final phase, avg of mean rewards over recent history is above the succcess threshold and
                # its standard deviation is small.
                avg_of_mean = mean(self.trials[trial_id]["mean_rewards"])
                std_of_mean = stdev(self.trials[trial_id]["mean_rewards"])
                #print("///// StopLogic: iter #{}, avg reward = {:.2f}, std of mean = {:.3f}".format(total_iters, avg, std_of_mean))
                if phase == self.num_phases-1  and  avg_of_mean >= success_avg_thresh  and  std_of_mean <= self.completion_std_threshold:
                    print("\n///// Stopping trial - SUCCESS!\n")
                    return True

                if total_steps > min_timesteps:
                    if not self.crossed_min_timesteps:
                        print("///// StopLogic: Beyond min time steps for phase {}".format(phase))
                        self.crossed_min_timesteps = True

                    # Stop if max iterations reached
                    if self.max_iterations is not None  and  total_iters >= self.max_iterations:
                        print("\n///// Stopping trial - reached max iterations.")
                        return True

                    # Stop if max timesteps reached
                    if self.max_timesteps is not None  and  result["timesteps_since_restore"] >= self.max_timesteps:
                        print("\n///// Stopping trial - reached max timesteps of {}.".format(self.max_timesteps))
                        return True

                    # Stop if mean and max rewards haven't significantly changed in recent history
                    std_of_max  = stdev(self.trials[trial_id]["max_rewards"])
                    if std_of_mean <= self.completion_std_threshold  and  std_of_max <= self.completion_std_threshold:
                        if avg_of_mean >= success_avg_thresh:
                            print("\n///// Stopping trial - winner with no further change. Mean avg = {:.1f}, mean std = {:.2f}"
                                    .format(avg_of_mean, std_of_mean))
                        else:
                            print("\n///// Stopping trial - loser with no further change. Mean avg = {:.1f}, mean std = {:.2f}"
                                    .format(avg_of_mean, std_of_mean))
                        return True

                    # If user chooses to let it run, regardless of reward trends, then let it run
                    if let_it_run:
                        return False

                    # If the avg mean reward over recent history is below the failure threshold then
                    if avg_of_mean < failure_avg_thresh:
                        done = False
                        slope_mean = self._get_slope(self.trials[trial_id]["mean_rewards"])
                        avg_of_min = mean(list(self.trials[trial_id]["min_rewards"]))

                        # If the max reward is below success threshold and not climbing significantly, then stop as a failure
                        dq_max = self.trials[trial_id]["max_rewards"]
                        avg_of_max = mean(dq_max)
                        slope_max = self._get_slope(dq_max)
                        if avg_of_max < success_avg_thresh:
                            if avg_of_max <= failure_avg_thresh  or  (slope_max < 0.0  and  slope_mean < 0.04):
                                print("\n///// Stopping trial - max is toast in {} iters with little hope of turnaround.\n".format(self.most_recent))
                                done = True

                        # If the avg mean is well below the best achieved so far and going south, then stop as failure
                        delta = max((self.trials[trial_id]["best_mean"] - self.trials[trial_id]["worst_mean"]) * self.degrade_threshold, 10.0)
                        thresh_value = self.trials[trial_id]["best_mean"] - delta
                        if avg_of_mean < thresh_value  and  slope_mean < 0.0:
                            print("\n///// Stopping trial - mean reward is failing and {:.0f}% below its peak of {:.1f}."
                                    .format(100*self.degrade_threshold, self.trials[trial_id]["best_mean"]))
                            done = True

                        # If the mean curve is heading down and the max is not increasing then stop as a failure
                        if slope_max < -0.04  and  slope_mean < -0.04:
                            print("\n///// Stopping trial - mean reward bad & getting worse, max is not improving in latest {} iters."
                                    .format(self.most_recent))
                            done = True

                        # If the mean is a lot closer to the min than to the max and no sign of improving then stop as failure
                        if avg_of_mean - avg_of_min < 0.25*(avg_of_max - avg_of_min)  and  slope_mean <= 0.0:
                            print("\n///// Stopping trial - no improvement and min reward is dominating in latest {} iters."
                                    .format(self.most_recent))
                            done = True

                        if done:
                            print("///// Trial {}, mean avg = {:.1f}, mean slope = {:.2f}, max avg = {:.1f}, max slope = {:.2f}"
                                    .format(trial_id, avg_of_mean, slope_mean, avg_of_max, slope_max))
                            print("      Phase = {}, steps complete = {}, min timesteps = {}, success threshold = {:.2f}, failure threshold = {:.2f}"
                                    .format(phase, total_steps, min_timesteps, success_avg_thresh, failure_avg_thresh))
                            print("      latest means:")
                            for i in range(len(self.trials[trial_id]["mean_rewards"]) // 5):
                                print("      {:3d}: ".format(5*i), end="")
                                for j in range(5):
                                    print("{:5.1f}, ".format(self.trials[trial_id]["mean_rewards"][5*i + j]), end="")
                                print(" ")
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
