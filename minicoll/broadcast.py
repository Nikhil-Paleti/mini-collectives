# src/minicoll/broadcast.py
from mpi4py import MPI
import numpy as np
import math

def _nbytes(a: np.ndarray) -> int:
    return int(a.nbytes)

def bcast_tree(comm: MPI.Comm, buf: np.ndarray, root: int = 0) -> int:

    rank = comm.Get_rank()
    size = comm.Get_size()
    if size == 1:
        return 0

    # Map to virtual rank space where root == 0
    vrank = (rank - root) % size

    steps = math.ceil(math.log2(size))
    tag = 101  # keep distinct from other collectives
    bytes_rank = 0
    msg_nbytes = _nbytes(buf)

    for k in range(steps):
        span  = 1 << k                  # distance this round: 1,2,4,8,...
        block = span << 1               # group size: 2*span
        pos   = vrank % block           # position within the group

        if pos < span:
            # lower half -> sender (if partner exists inside [0..P-1])
            dst_vr = vrank + span
            if dst_vr < size:
                dst = (dst_vr + root) % size
                comm.Send(buf, dest=dst, tag=tag)
                bytes_rank += msg_nbytes
        else:
            # upper half -> receiver
            src_vr = vrank - span
            src = (src_vr + root) % size
            comm.Recv(buf, source=src, tag=tag)
            bytes_rank += msg_nbytes

    return bytes_rank