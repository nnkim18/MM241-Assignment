# from student_submissions.s2210xxx.policy_genetic import policy_genetic
from policy import Policy
from student_submissions.s2210xxx.floorceil import FloorCeil
from student_submissions.s2210xxx.genetic import GeneticPolicy


class Policy2210xxx(Policy):
    def __init__(self, policy_id):
        super().__init__()  # Call parent class constructor
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Initialize policy attribute
        self.policy = None

        if policy_id == 1:
            self.policy = FloorCeil()  # Create instance of BFDH policy
        elif policy_id == 2:
            # Initialize policy 2 here
            self.policy = GeneticPolicy()

    def get_action(self, observation, info):
        if self.policy is not None:
            return self.policy.get_action(observation, info)
        return None  # Or implement default behavior

    # Student code here
    # You can add more functions if needed
