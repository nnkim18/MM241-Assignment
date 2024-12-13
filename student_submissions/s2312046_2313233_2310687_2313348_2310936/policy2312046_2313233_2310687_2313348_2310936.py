from policy import Policy
from s2312046_2313233_2310687_2313348_2310936.ppo_policy import PPO
from s2312046_2313233_2310687_2313348_2310936.heuristic_policy import CPolicy


class policy2312046_2313233_2310687_2313348_2310936(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy = None
        if policy_id == 1:
            self.policy = CPolicy()
        elif policy_id == 2:
            self.policy = PPO()

    def get_action(self, observation, info):
        return self.policy.get_action(observation,info)
