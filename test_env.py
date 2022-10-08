"""Unit tester for the SimpleHighwayRamp environment model."""

from simple_highway_with_ramp import SimpleHighwayRamp

config = {}

# Case 1
print("\n+++++ Begin case 1")
env = SimpleHighwayRamp(config)
obs = env.reset()
print("Case 1: reset obs = ", obs)
