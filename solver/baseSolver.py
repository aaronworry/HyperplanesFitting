import numpy as np
import time

class BaseSolver():
    def __init__(self, max_time: float = 1000,
        max_iterations: int = 1000,
        min_gradient_norm: float = 1e-6,
        min_step_size: float = 1e-10,
        max_cost_evaluations: int = 5000,
        verbosity: int = 2,
        log_verbosity: int = 0):
        
        self._max_time = max_time
        self._max_iterations = max_iterations
        self._min_gradient_norm = min_gradient_norm
        self._min_step_size = min_step_size
        self._max_cost_evaluations = max_cost_evaluations
        self._verbosity = verbosity
        self._log_verbosity = log_verbosity

        self._log = None
        
    def run(self):
        pass
        
    def _check_stopping_criterion(
        self,
        start_time,
        iteration=-1,
        gradient_norm=np.inf,
        step_size=np.inf,
        cost_evaluations=-1,
    ):
        run_time = time.time() - start_time
        reason = None
        if time.time() >= start_time + self._max_time:
            reason = (
                f"Terminated - max time reached after {iteration} iterations."
            )
        elif iteration >= self._max_iterations:
            reason = (
                "Terminated - max iterations reached after "
                f"{run_time:.2f} seconds."
            )
        elif gradient_norm < self._min_gradient_norm:
            reason = (
                f"Terminated - min grad norm reached after {iteration} "
                f"iterations, {run_time:.2f} seconds."
            )
        elif step_size < self._min_step_size:
            reason = (
                f"Terminated - min step_size reached after {iteration} "
                f"iterations, {run_time:.2f} seconds."
            )
        elif cost_evaluations >= self._max_cost_evaluations:
            reason = (
                "Terminated - max cost evals reached after "
                f"{run_time:.2f} seconds."
            )
        return reason
        