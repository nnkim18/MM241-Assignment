from policy import Policy
from student_submissions.s2313293_2313950_2310280.policy2313293_2313950_2310280 import Policy2313293_2313950_2310280

class Policy2210xxx(Policy):
    def __init__(self, policy_id = 1):
        self.policy = Policy2313293_2313950_2310280(policy_id)

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)