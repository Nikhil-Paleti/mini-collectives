# src/minicoll/communicator.py
from mpi4py import MPI
import numpy as np
from .broadcast import bcast_tree

class Communicator:
    """
    Minimal communicator wrapper that tracks per-rank bytes transferred.
    """
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    # --- passthroughs
    def Get_size(self) -> int: return self.comm.Get_size()
    def Get_rank(self) -> int: return self.comm.Get_rank()
    def Barrier(self): self.comm.Barrier()
    def Allreduce(self, src, dst, op=MPI.SUM): self.comm.Allreduce(src, dst, op)

    # --- helpers
    @staticmethod
    def _nbytes(buf: np.ndarray) -> int:
        return int(buf.nbytes)

    # --- our custom broadcast
    def myBcast(self, buf: np.ndarray, root: int = 0, method: str = "tree") -> int:
        """
        In-place broadcast into 'buf' from 'root' to all ranks.
        Returns bytes this rank sent/received; accumulates into self.total_bytes_transferred.
        """
        if method == "tree":
            bytes_rank = bcast_tree(self.comm, buf, root)
        else:
            raise ValueError(f"Unknown method '{method}' for broadcast")

        self.total_bytes_transferred += bytes_rank
        return bytes_rank