"""Two real CPU ranks must equal the serial global-batch gradient."""
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tunnelweave.train import reduce_gradients


def _worker(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    p = torch.nn.Parameter(torch.tensor(2.))
    q = torch.nn.Parameter(torch.tensor(3.))
    unused = torch.nn.Parameter(torch.tensor(4.))
    # Four event losses per rank, each divided by the GLOBAL batch size of eight.
    loss = sum((p * (i + rank * 4 + 1)) ** 2 / 8 for i in range(4))
    if rank == 1:
        loss = loss + q.square() / 8
    loss.backward()
    reduce_gradients([p, q, unused], 2)
    torch.save({'p': p.grad, 'q': q.grad, 'unused': unused.grad}, Path(output) / f'rank{rank}.pt')
    dist.destroy_process_group()


def test_two_rank_sum_matches_serial_and_preserves_unused(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    p = torch.nn.Parameter(torch.tensor(2.))
    q = torch.nn.Parameter(torch.tensor(3.))
    (sum((p * (i + 1)) ** 2 / 8 for i in range(8)) + q.square() / 8).backward()
    for rank in range(2):
        result = torch.load(tmp_path / f'rank{rank}.pt', weights_only=True)
        assert torch.equal(result['p'], p.grad)
        assert torch.equal(result['q'], q.grad)
        assert result['unused'] is None
