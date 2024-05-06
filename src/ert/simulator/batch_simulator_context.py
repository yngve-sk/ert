from __future__ import annotations

import logging
import time
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ert.scheduler.job import State as JobState
from ert.scheduler.scheduler import Scheduler

from .simulation_context import SimulationContext

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from ert.enkf_main import EnKFMain
    from ert.storage import Ensemble

Status = namedtuple("Status", "waiting pending running complete failed")


class BatchContext(SimulationContext):
    def __init__(
        self,
        result_keys: "Iterable[str]",
        ert: "EnKFMain",
        fs: Ensemble,
        mask: npt.NDArray[np.bool_],
        itr: int,
        case_data: List[Tuple[Any, Any]],
    ):
        """
        Handle which can be used to query status and results for batch simulation.
        """
        super().__init__(ert, fs, mask, itr, case_data)
        self.result_keys = result_keys
        self.ert_config = ert.ert_config

    def join(self) -> None:
        """
        Will block until the simulation is complete.
        """
        while self.running():
            time.sleep(1)

    def running(self) -> bool:
        return self.isRunning()

    @property
    def status(self) -> Status:
        """
        Will return the state of the simulations.

        NB: Killed realizations are not reported.
        """
        if isinstance(self._job_queue, Scheduler):
            states = self._job_queue.count_states()
            return Status(
                running=states[JobState.RUNNING],
                waiting=states[JobState.WAITING],
                pending=states[JobState.PENDING],
                complete=states[JobState.COMPLETED],
                failed=states[JobState.FAILED],
            )
        return Status(
            running=self.getNumRunning(),
            waiting=self.getNumWaiting(),
            pending=self.getNumPending(),
            complete=self.getNumSuccess(),
            failed=self.getNumFailed(),
        )

    def results(self) -> List[Optional[Dict[str, "npt.NDArray[np.float64]"]]]:
        """Will return the results of the simulations.

        Observe that this function will raise RuntimeError if the simulations
        have not been completed. To be certain that the simulations have
        completed you can call the join() method which will block until all
        simulations have completed.

        The function will return all the results which were configured with the
        @results when the simulator was created. The results will come as a
        list of dictionaries of arrays of double values, i.e. if the @results
        argument was:

             results = ["CMODE", "order"]

        when the simulator was created the results will be returned as:


          [ {"CMODE" : [1,2,3], "order" : [1,1,3]},
            {"CMODE" : [1,4,1], "order" : [0,7,8]},
            None,
            {"CMODE" : [6,1,0], "order" : [0,0,8]} ]

        For a simulation which consist of a total of four simulations, where the
        None value indicates that the simulator was unable to compute a request.
        The order of the list corresponds to case_data provided in the start
        call.

        """
        if self.running():
            raise RuntimeError(
                "Simulations are still running - need to wait before gettting results"
            )

        self.get_ensemble().unify_responses()
        res: List[Optional[Dict[str, "npt.NDArray[np.float64]"]]] = []
        for sim_id in range(len(self)):
            if not self.didRealizationSucceed(sim_id):
                logging.error(f"Simulation {sim_id} failed.")
                res.append(None)
                continue
            d = {}
            for key in self.result_keys:
                data = self.get_ensemble().load_responses(key, (sim_id,))
                d[key] = data["values"].dropna("index").values.flatten()
            res.append(d)

        return res
