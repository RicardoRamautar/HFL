import math

class LRScheduler():
    _REQUIRED_PARAMS = {
        "exp": {"base_lr", "gamma"},
        "step": {"base_lr", "gamma", "step_size"},
        "cosine": {"base_lr", "min_lr", "total_rounds"},
        "cyclic": {"base_lr", "max_lr", "step_size"}
    }

    def __init__(self, policy: str, **kwargs):
        self.policies = list(self._REQUIRED_PARAMS.keys())
        
        self.policy = policy
        self.cfg = kwargs

        if policy not in self._REQUIRED_PARAMS:
            raise ValueError(
                f"Specified policy should be one of the "
                f"following: {self.policies}, but got '{self.policy}'."
            )

        missing_params = self._REQUIRED_PARAMS[self.policy] - self.cfg.keys()
        if missing_params:
            raise ValueError(
                f"Missing params for policy '{self.policy}': {missing_params}."
            )
        

    def lr_at(self, round_idx: int) -> float:
        if self.policy == "step":
            k = round_idx // self.cfg["step_size"]
            return self.cfg["base_lr"] * (self.cfg["gamma"] ** k)

        if self.policy == "exp":
            return self.cfg["base_lr"] * (self.cfg["gamma"] ** round_idx)

        if self.policy == "cosine":
            return self.cfg['min_lr'] + 0.5 * \
                (self.cfg['base_lr'] - self.cfg['min_lr']) * \
                (1 + math.cos(round_idx * math.pi / self.cfg['total_rounds']))

        if self.policy == "cyclic":
            cycle = math.floor(1 + round_idx / (2 * self.cfg['step_size']))
            x = abs(round_idx / self.cfg['step_size'] - 2 * cycle + 1)
            return  self.cfg['base_lr'] + (self.cfg['max_lr'] - self.cfg['base_lr']) * max(0, (1 - x))

        raise ValueError(
            f"Stored policy ('{self.policy}') is unknown."
            f"Ensure it is one of the following: {self.policies}."
        )
            

