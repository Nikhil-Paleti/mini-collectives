# mini-collectives

A minimal hands-on project to re-implement **MPI collectives** (Allreduce, Allgather, Reduce-Scatter, etc.)
from scratch using `mpi4py`, with a focus on understanding how distributed **ML communication** works.

```bash
brew install open-mpi

# Recreate the exact environment
uv sync

uv run mpiexec -n 4 python -m scripts.demo_all
uv run mpiexec -n 4 python -m scripts.bench > results.csv
```

## Cost Model (α–β)

For message size **m** bytes:

```
T = α + β·m
```

- **α** → per-message latency  
- **β** → inverse bandwidth (s/byte)

Collective time ≈ (number of steps)·α + (bytes per slowest rank)·β.

---

## Collectives

| Collective | Used in ML for | Approx. cost per GPU (× message size) | Appears in | Description / When used |
|---|---|---:|---|---|
| **Allreduce** | Gradient synchronization | **≈ 2×** (Reduce-Scatter + All-Gather) | **DDP**, ZeRO-1/2, some TP ops | Core operation each backward bucket. Per-GPU volume ≈ 2 · S · (N-1)/N. |
| **Reduce** | Accumulate to a root | **≈ 1×** (non-root); root gets **N×** | Legacy PS | Rare in modern fully-sharded training; mostly historical. |
| **Broadcast (`Bcast`)** | Distribute weights | **≈ 1×** | Init/reload, elastic training | Root → all, typically at startup or checkpoint load. |
| **Allgather** | Gather sharded params / activations | **≈ 1×** | **FSDP / ZeRO-3**, sequence parallel | Before compute to materialize full tensors from shards. |
| **Reduce-Scatter** | Sharded gradient reduction | **≈ 1×** | **FSDP / ZeRO-3**, sequence parallel | After backward to leave each rank with its shard’s sum. |
| **Alltoall** | Token / expert routing; tensor / sequence swaps | **≈ 1×** | **MoE**, Tensor / Sequence Parallel | Each rank communicates with every other; dominates MoE cost. |
| **Scatter** | Split data from root | **≈ 1×** (root sends) | Data bootstrap | Usually replaced by DDP Sampler; not on hot path. |
| **Gather** | Metrics / validation aggregation | **≈ 1×** (root receives) | Logging / evaluation | Not performance-critical. |

**MoE note:** real workloads often have imbalanced token counts per expert → variable message sizes → practically an `Alltoallv`.  
You can approximate this with `Alltoall` by padding to equal lengths.


### Practical Notes

- **Overlap & bucketing:** Frameworks bucket gradients (e.g., 25–50 MB) and overlap `allreduce` / `reduce-scatter` with backward compute to hide latency.  
- **Precision:** Communicate in **fp16 / bf16** whenever numerically safe; halves communication volume vs fp32.  
- **Chunking / pipelining:** Large tensors are divided into chunks to pipeline ring/tree steps and keep NICs saturated.  
- **Topology awareness:** Intra-node (NVLink / SHM) and inter-node (IB / ROCE) latencies differ; α–β model is a first-order guide, but contention and GPU placement strongly affect real throughput.
---

# mpi4py notes:


### Tags and Communicator Info

Tags (31, 32) act like channel IDs.  
Without different tags, if multiple messages are in flight, MPI could match them out of order.  
MPI guarantees that for each (source, destination, tag) pair, messages arrive in send order, but different tags are independent streams.  

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    # Phase 1: "reduce" phase (tag 31)
    a = np.array([100], dtype=np.int32)
    comm.Send(a, dest=1, tag=31)
    print("Rank 0 → Phase 1 sent:", a[0])

    # Phase 2: "broadcast" phase (tag 32)
    b = np.array([200], dtype=np.int32)
    comm.Send(b, dest=1, tag=32)
    print("Rank 0 → Phase 2 sent:", b[0])

elif rank == 1:
    # Matching receives with the correct tags
    x = np.empty(1, dtype=np.int32)
    y = np.empty(1, dtype=np.int32)

    comm.Recv(x, source=0, tag=31)  # Must match the first tag
    print("Rank 1 → Phase 1 received:", x[0])

    comm.Recv(y, source=0, tag=32)  # Second message with a different tag
    print("Rank 1 → Phase 2 received:", y[0])
```

---

### Blocking `Send` / `Recv`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    x = np.arange(8, dtype=np.int32)
    comm.Send(x, dest=1, tag=10)
    print("This line only runs after Send completes")
else:
    y = np.empty(8, dtype=np.int32)
    comm.Recv(y, source=0, tag=10)
    print("This line only runs after the data is received")
```

---

### Non-blocking `Isend` / `Irecv` + `Waitall`

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

buf  = np.full(1024, rank, dtype=np.float32)
recv = np.empty_like(buf)

left  = (rank - 1) % comm.Get_size()
right = (rank + 1) % comm.Get_size()

reqs = [
    comm.Isend(buf, dest=right, tag=20),
    comm.Irecv(recv, source=left,  tag=20)
]

MPI.Request.Waitall(reqs)
print(f"Rank {rank} received first element {recv[0]}")
```

---

### `Sendrecv` (deadlock-free pairwise exchange)

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

send = np.array([rank], dtype=np.int32)
recv = np.empty_like(send)

dst = (rank + 1) % comm.Get_size()
src = (rank - 1 + comm.Get_size()) % comm.Get_size()

comm.Sendrecv(sendbuf=send, dest=dst, sendtag=30,
              recvbuf=recv, source=src, recvtag=30)

print(f"Rank {rank} sent {send[0]} and received {recv[0]}")
```