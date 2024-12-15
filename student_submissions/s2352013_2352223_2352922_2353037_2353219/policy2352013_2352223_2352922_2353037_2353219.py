from policy import Policy
from . import Genetic, Greedy

class Policy2352013_2352223_2352922_2353037_2353219(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = Greedy.Greedy()
        elif policy_id == 2:
            self.policy = Genetic.Genetic()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
