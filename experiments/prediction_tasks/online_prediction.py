"""Prediction streams with a trained read-out, for a tanh or a gated cell of any width.

The whole step-size grid is inside one run: lanes are seeds x step sizes x rules, and all lanes of a seed
share data and initialisation.

Phase 1 (from a random start).  Every lane is trained online and then evaluated on held-out data with
frozen parameters.

Phase 2 (handover).  The step size of RTRL is fixed as the one with the smallest median online loss over
the last tenth of training; every lane (all rules, all step sizes) then restarts from the end point of the
RTRL lane of its seed at that step size and continues with its own rule and its own step size.

    python -m experiments.prediction_tasks.online_prediction --cell tanh --num-units 16
    python -m experiments.prediction_tasks.online_prediction --cell gru_reset_after --tasks delay:5,narma10
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.common import float_list, make_parser, stored_arguments, write_json
from trace_instruments import DTYPE, rules
from trace_instruments.online_learner import OnlineLearner
from trace_instruments.prediction_streams import POWER, TASKS, PredictionStream, task_tag

RESULTS_SUBDIR = ("prediction_tasks",)
RULES = [rules.RTRL, rules.SNAP1, rules.IMMEDIATE, "leak:0.5", rules.UORO, "tbptt:8"]
NUM_LOG_BLOCKS = 50
LAST_TENTH = 5  # blocks of the loss curve that make up the last tenth of a phase

REAL_SERIES = ("sunspot", POWER)  # fixed records, split 80/20
EVALUATION_BURN_IN = 200  # steps discarded by OnlineLearner.evaluate before it scores


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--num-units", type=int, default=16)
    parser.add_argument("--cell", default="tanh", help="tanh or gru_reset_after")
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--step-sizes", default="0.0003,0.001,0.003,0.01", help="grid for the recurrent weights")
    parser.add_argument("--readout-step-size", type=float, default=0.01)
    parser.add_argument("--training-steps", type=int, default=200_000)
    parser.add_argument("--handover-steps", type=int, default=100_000)
    parser.add_argument(
        "--train-length",
        type=int,
        default=50_000,
        help="length of the training series (synthetic tasks; the real records are split 80/20 whatever this says)",
    )
    parser.add_argument(
        "--eval-length", type=int, default=10_000, help="length of the held-out series (synthetic tasks only)"
    )
    args = parser.parse_args(argv)
    synthetic = [task for task in args.tasks.split(",") if task not in REAL_SERIES]
    if synthetic and args.eval_length <= EVALUATION_BURN_IN:  # the real records ignore the lengths
        parser.error(f"--eval-length must exceed the burn-in of the evaluation, {EVALUATION_BURN_IN} steps")
    step_sizes = float_list(args.step_sizes)
    num_rules, num_step_sizes = len(RULES), len(step_sizes)
    replicas = num_rules * num_step_sizes
    num_seeds = args.num_seeds
    shape = (num_seeds, num_step_sizes, num_rules)

    def per_lane(metrics):
        return {key: value.view(shape).cpu().numpy() for key, value in metrics.items()}

    for task in args.tasks.split(","):
        start = time.time()
        stream = PredictionStream(
            task,
            list(range(num_seeds)),
            replicas,
            args.device,
            train_length=args.train_length,
            eval_length=args.eval_length,
        )
        lane_rules = [RULES[i % num_rules] for i in range(stream.B)]
        lane_steps = torch.tensor(
            [step_sizes[(i // num_rules) % num_step_sizes] for i in range(stream.B)], dtype=DTYPE, device=args.device
        )
        learner = OnlineLearner(
            stream,
            args.num_units,
            lane_rules,
            lane_steps,
            args.device,
            seed=0,
            readout_step_size=args.readout_step_size,
            replicas=replicas,
            clip=0.0,
            cell=args.cell,
        )

        # ---- phase 1: from a random start
        curve = learner.train(args.training_steps, log_every=args.training_steps // NUM_LOG_BLOCKS)
        scratch = per_lane(learner.evaluate())
        online = curve.view(num_seeds, num_step_sizes, num_rules, -1)[..., -LAST_TENTH:].mean(-1).numpy()
        rtrl_online = np.where(np.isfinite(scratch["mse"][:, :, 0]), online[:, :, 0], np.inf)
        handover_index = int(np.argmin(np.median(rtrl_online, 0)))

        # ---- phase 2: restart every lane from the RTRL lane (rule index 0) of its seed at that step size
        def from_rtrl_lane(v):
            lanes = v.view(num_seeds, num_step_sizes, num_rules, *v.shape[1:])
            source = lanes[:, handover_index : handover_index + 1, :1]
            return source.expand(num_seeds, num_step_sizes, num_rules, *v.shape[1:]).reshape(v.shape).clone()

        theta0 = from_rtrl_lane(learner.theta)
        learner.theta, learner.R, learner.c0 = theta0.clone(), from_rtrl_lane(learner.R), from_rtrl_lane(learner.c0)
        learner.alive = from_rtrl_lane(learner.alive)
        learner.reset_state()
        stream.t = 0
        handover_point = per_lane(learner.evaluate())  # each rule's own mean update at the handover point
        learner.reset_state()
        curve2 = learner.train(args.handover_steps, log_every=args.handover_steps // NUM_LOG_BLOCKS)
        online2 = curve2.view(num_seeds, num_step_sizes, num_rules, -1)[..., -LAST_TENTH:].mean(-1).numpy()
        handover = per_lane(learner.evaluate())
        handover["parameter_movement"] = (learner.theta - theta0).norm(dim=1).view(shape).cpu().numpy()

        record = dict(
            args=stored_arguments(args),
            task=task,
            rules=RULES,
            step_sizes=step_sizes,
            num_parameters=learner.cell.P,
            seconds=time.time() - start,
            handover_step_index=handover_index,
            online_loss_scratch=online.tolist(),
            online_loss_handover=online2.tolist(),
            scratch={key: value.tolist() for key, value in scratch.items()},
            handover={key: value.tolist() for key, value in handover.items()},
            handover_point={key: value.tolist() for key, value in handover_point.items()},
        )
        write_json(args.output_dir / f"{args.cell}_n{args.num_units}_{task_tag(task)}.json", record)
        print(
            f"{task} cell={args.cell} n={args.num_units} P={learner.cell.P}  [{time.time() - start:.0f}s]", flush=True
        )
        median_online = np.nanmedian(online, 0)
        selected = np.nanargmin(np.where(np.isnan(median_online), np.inf, median_online), 0)  # per rule
        for r, rule in enumerate(RULES):
            mse = scratch["mse"][:, selected[r], r]
            mse_rtrl = scratch["mse"][:, selected[0], 0]
            ratio = handover["mse"][:, handover_index, r] / handover["mse"][:, handover_index, 0]
            print(
                f"   {rule:9s} alpha={step_sizes[selected[r]]:<7g} MSE median={np.nanmedian(mse):.3e} "
                f"ratio to RTRL={np.nanmedian(mse / mse_rtrl):.2f} div={int(np.isnan(mse).sum())} "
                f"| handover at RTRL's step, ratio median={np.nanmedian(ratio):.2f} "
                f"[{np.nanmin(ratio):.2f},{np.nanmax(ratio):.2f}] div={int(np.isnan(ratio).sum())}",
                flush=True,
            )


if __name__ == "__main__":
    main()
