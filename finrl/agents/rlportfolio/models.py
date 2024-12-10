"""
DRL models to solve the portfolio optimization task with reinforcement learning.
This agent was developed to work with environments like PortfolioOptimizationEnv.
"""

from __future__ import annotations

from rlportfolio.algorithm import PolicyGradient, EpisodicPolicyGradient
from rlportfolio.policy import EIIE, EIIERecurrent, EI3, GPM, GPMSimplified

MODELS = {"pg": PolicyGradient, "epg": EpisodicPolicyGradient}
POLICIES = {"eiie": EIIE, "eiie_recurrent": EIIERecurrent, "ei3": EI3, "gpm": GPM, "gpm_simplified": GPMSimplified}

class DRLAgent:
    """Implementation for DRL algorithms for portfolio optimization.

    Note:
        During testing, the agent is optimized through online learning.
        The parameters of the policy is updated repeatedly after a constant
        period of time. To disable it, set learning rate to 0.

    Attributes:
        env: Gym environment class.
    """

    def __init__(self, env):
        """Agent initialization.

        Args:
            env: Gym environment to be used in training.
        """
        self.env = env

    def get_model(
        self, model_name, policy="eiie", model_kwargs=None, policy_kwargs=None, device="cpu"
    ):
        """Setups DRL model.

        Args:
            model_name: Name of the model according to MODELS list.
            device: Device used to instantiate neural networks.
            model_kwargs: Arguments to be passed to model class.
            policy_kwargs: Arguments to be passed to policy class.

        Note:
            model_kwargs and policy_kwargs are dictionaries. The keys must be strings
            with the same names as the class arguments. Example for model_kwargs::

            { "lr": 0.01, "policy": EIIE }

        Returns:
            An instance of the model.
        """
        if model_name not in MODELS:
            raise NotImplementedError("The model requested is not implemented.")
        
        if policy not in POLICIES:
            raise NotImplementedError("The policy requested is not implemented.")

        model = MODELS[model_name]
        model_kwargs = {} if model_kwargs is None else model_kwargs
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        # specify policy in model_kwargs
        model_kwargs["policy"] = POLICIES[policy]

        # add device settings
        model_kwargs["device"] = device
        policy_kwargs["device"] = device

        # add policy_kwargs inside model_kwargs
        model_kwargs["policy_kwargs"] = policy_kwargs

        return model(self.env, **model_kwargs)

    @staticmethod
    def train_model(model, steps_or_eps, **kwargs):
        """Trains portfolio optimization model.

        Args:
            model: Instance of the model.
            steps_or_eps: Number of steps or episodes (depending on the model chosen)
                to train the agent.
            **kwargs: Keyword arguments of the training function of the chosen model.

        Note:
            Check RLPortfolio's documentation in order to understand the possible
            keyword arguments. https://rlportfolio.readthedocs.io

        Returns:
            An instance of the trained model.
        """
        model.train(steps_or_eps, **kwargs)
        return model

    @staticmethod
    def DRL_validation(
        model,
        test_env,
        **kwargs,
    ):
        """Tests a model in a testing environment.

        Args:
            model: Instance of the model.
            test_env: Gymnasium environment to be used in testing.
            **kwargs: Keyword arguments of the test function of the chosen model.

        Note:
            Check RLPortfolio's documentation in order to understand the possible
            keyword arguments. https://rlportfolio.readthedocs.io
        
        Returns:
            A dictionary with metrics of the test sequence.
        """
        model.test(test_env, **kwargs)
