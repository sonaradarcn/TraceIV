"""Per-step cost of the update rules for a tanh cell with ``n`` units (one input, bias), one learner, float64.

Each rule is implemented with the storage it needs and nothing else:

    rule         what is propagated                                       time                 memory
    forward      state update only                                        n^2                  n
    immediate    g_i = delta_i D_i phi                                    n^2                  -
    snap1        one eligibility vector per unit                          n^2                  n q
    uoro         rank-one sensitivity estimate                            n^2                  n + P
    probes:k     SnAp-1 + k forward tangents of the exact sensitivity     (k+1) n^2 + k P      n q + k (n + P)
    hybrid:u     SnAp-1 + exact sensitivity for the weights of u units    u n^3                n q + u n q
    tbptt:K      backpropagation through the last K steps                 K n^2                K (2n + q)
    rtrl         full sensitivity                                         n^4                  n P

Only the recursions are timed.  ``probes:k`` leaves out the resampling and orthogonalisation of the
tangents and the tracking update of the estimate, ``hybrid:u`` and the others leave out the parameter
update; memory is the size of the stored trace, not the peak of the process.

    python -m experiments.restoration.step_cost_benchmark --device cpu --threads 1
"""

from __future__ import annotations

import time

import torch

from experiments.common import int_list, make_parser, stored_arguments, write_json
from trace_instruments import DTYPE

RESULTS_SUBDIR = ("restoration", "step_cost")
WARMUP_STEPS = 20


def benchmark(rule, n, device, steps):
    """Microseconds per step and stored trace size (MiB) of one rule."""
    g = torch.Generator().manual_seed(0)
    q = n + 2
    P = n * q
    W = (0.9 * torch.randn(n, n, generator=g, dtype=DTYPE) / n**0.5).to(device)
    V = torch.randn(n, 2, generator=g, dtype=DTYPE).to(device)
    h = torch.zeros(n, dtype=DTYPE, device=device)
    delta = torch.randn(n, generator=g, dtype=DTYPE).to(device)
    xs = torch.randn(steps + WARMUP_STEPS, 1, generator=g, dtype=DTYPE).to(device)
    one = torch.ones(1, dtype=DTYPE, device=device)
    memory = 0
    if rule == "snap1" or rule.startswith(("probes", "hybrid")):
        z = torch.zeros(n, q, dtype=DTYPE, device=device)
        memory += z.numel()
    if rule == "uoro":
        uoro_state = torch.zeros(n, dtype=DTYPE, device=device)
        uoro_parameters = torch.zeros(P, dtype=DTYPE, device=device)
        memory += n + P
    if rule == "rtrl":
        Z = torch.zeros(n, P, dtype=DTYPE, device=device)
        memory += Z.numel()
        idx = torch.arange(n, device=device)
    if rule.startswith("probes"):
        k = int(rule.split(":")[1])
        U = torch.linalg.qr(torch.randn(P, k, generator=g, dtype=DTYPE)).Q.to(device)
        tangents = torch.zeros(n, k, dtype=DTYPE, device=device)
        statistic = torch.zeros(k, dtype=DTYPE, device=device)
        memory += U.numel() + tangents.numel()
    if rule.startswith("hybrid"):
        u = int(rule.split(":")[1])
        Z_exact = torch.zeros(n, u, q, dtype=DTYPE, device=device)
        memory += Z_exact.numel()
    if rule.startswith("tbptt"):
        K = int(rule.split(":")[1])
        buffer = []
        memory += K * (2 * n + q)
    if str(device).startswith("cuda"):

        def synchronize():
            torch.cuda.synchronize(device)

    else:

        def synchronize():
            return None

    with torch.no_grad():
        for t in range(steps + WARMUP_STEPS):
            if t == WARMUP_STEPS:
                synchronize()
                start = time.perf_counter()
            phi = torch.cat([h, xs[t], one])
            h = torch.tanh(W @ h + V @ torch.cat([xs[t], one]))
            D = 1 - h * h
            if rule == "forward":
                continue
            if rule == "immediate":
                gradient = (delta * D)[:, None] * phi[None, :]
            elif rule == "uoro":
                nu = torch.sign(torch.randn(n, dtype=DTYPE, device=device))
                propagated = D * (W @ uoro_state)
                nu_B = ((nu * D)[:, None] * phi[None, :]).reshape(P)
                r0 = ((uoro_parameters.norm() + 1e-12) / (propagated.norm() + 1e-12)).sqrt()
                r1 = ((nu_B.norm() + 1e-12) / (nu.norm() + 1e-12)).sqrt()
                uoro_state = r0 * propagated + r1 * nu
                uoro_parameters = uoro_parameters / r0 + nu_B / r1
                gradient = uoro_parameters * (uoro_state @ delta)
            elif rule.startswith("tbptt"):
                buffer.append((D, phi, h))
                buffer = buffer[-K:]
                e = delta
                gradient = torch.zeros(n, q, dtype=DTYPE, device=device)
                for D_past, phi_past, _ in reversed(buffer):
                    e = e * D_past
                    gradient += e[:, None] * phi_past[None, :]
                    e = W.T @ e
            elif rule == "rtrl":
                Z = (D[:, None] * W) @ Z
                Z.view(n, n, q)[idx, idx] += D[:, None] * phi[None, :]
                gradient = delta @ Z
            else:
                z = (D * torch.diagonal(W))[:, None] * z + D[:, None] * phi[None, :]
                gradient = delta[:, None] * z
                if rule.startswith("probes"):
                    Um = U.view(n, q, k)
                    tangents = D[:, None] * (W @ tangents) + D[:, None] * torch.einsum("a,iak->ik", phi, Um)
                    statistic += delta @ tangents - torch.einsum("i,ia,iak->k", delta, z, Um)
                elif rule.startswith("hybrid"):
                    Z_exact = torch.einsum("ij,jua->iua", D[:, None] * W, Z_exact)
                    Z_exact[n - u :, torch.arange(u), :] += D[n - u :, None] * phi[None, :]
                    gradient_exact = torch.einsum("i,iua->ua", delta, Z_exact)  # noqa: F841 (timed, not used)
        synchronize()
    return (time.perf_counter() - start) / steps * 1e6, memory * 8 / 2**20


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR)
    parser.add_argument("--num-units", default="16,32,64,128,256", help="comma-separated network sizes")
    parser.add_argument("--steps", type=int, default=300, help="timed steps (after a warm-up)")
    parser.add_argument("--rules", default="forward,immediate,snap1,uoro,tbptt:8,probes:4,probes:16,hybrid:2,rtrl")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; 0 keeps the torch default")
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        hardware = torch.cuda.get_device_name(args.device)
    else:
        hardware = f"CPU, {torch.get_num_threads()} thread(s)"
    record = dict(args=stored_arguments(args), hardware=hardware, rows=[])
    for n in int_list(args.num_units):
        for rule in args.rules.split(","):
            microseconds, mebibytes = benchmark(rule, n, args.device, args.steps)
            record["rows"].append(
                dict(
                    num_units=n,
                    num_parameters=n * (n + 2),
                    rule=rule,
                    microseconds_per_step=microseconds,
                    trace_mebibytes=mebibytes,
                )
            )
            print(
                f"n={n:4d} P={n * (n + 2):6d} {rule:10s} {microseconds:10.1f} us/step  trace {mebibytes:9.3f} MiB",
                flush=True,
            )
    write_json(args.output_dir / f"cost_{args.device.replace(':', '')}.json", record)


if __name__ == "__main__":
    main()
