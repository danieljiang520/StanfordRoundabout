print('Are you running this in a tmux? (y/n): ', end='')
in_tmux = input() == 'y'
if not in_tmux:
    print('Rerun this script in a tmux, to avoid early terminations due to ssh disconnections. Quitting now.')
    quit()
print('Do you want to use WandB for logging training progress (highly recommended)? (y/n): ', end='')
use_wandb = input() == 'y'
if use_wandb:
    import wandb
    wandb.login()
    print('USING WANDB - THE URL TO ACCESS LOGS WILL BE PRINTED SHORTLY')
else:
    print('NOT USING WANDB - LOGS WILL STILL BE SAVED LOCALLY')

import sys
import subprocess
from pathlib import Path

import datetime
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
repo_dir = Path(__file__).resolve().parent
run_experiment = repo_dir / 'libraries' / 'DeepReach_MPC' / 'run_experiment.py'

# cmd = [
#     sys.executable,
#     str(run_experiment),
#     '--mode',
#     'train',
#     '--experiment_name',
#     timestamp,
#     '--dynamics_class',
#     'TwoCar8D',
#     '--tMax',
#     '1',
#     '--pretrain',
#     '--pretrain_iters',
#     '1000',
#     '--num_epochs',
#     '104000',
#     '--counter_end',
#     '100000',
#     '--num_nl',
#     '512',
#     '--collisionR',   ## Current  collision radius is 5.0. This is probably too large
#     '5.0',
#     '--wheelbase',   ## Wheelbase for two-car system
#     '2.7',
#     '--set_mode',
#     'avoid',
#     '--lr',
#     '2e-5',
#     '--num_MPC_batches',
#     '20',
#     '--MPC_batch_size',
#     '5000'
# ]

cmd = [  ##Sanity Run
    sys.executable,
    str(run_experiment),
    '--mode', 'train',
    '--experiment_name', timestamp,
    '--dynamics_class', 'TwoCar8D',

    '--tMax', '1',
    '--pretrain',
    '--pretrain_iters', '2000',

    '--num_epochs', '30000',
    '--counter_end', '20000',

    '--num_nl', '512',
    '--collisionR', '1.0',
    '--wheelbase', '5',
    '--set_mode', 'avoid',

    '--lr', '5e-5',

    '--num_MPC_batches', '10',
    '--MPC_batch_size', '2000'
]

if use_wandb:
    cmd.extend([
        '--use_wandb',
        '--wandb_project',
        'aa276',
        '--wandb_name',
        timestamp,
        '--wandb_group',
        'two_car_8d'
    ])
# subprocess.run(cmd, cwd=repo_dir, check=True)
result = subprocess.run(
    cmd,
    cwd=repo_dir,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

print(result.stdout)

if result.returncode != 0:
    print(f"run_experiment.py failed with exit code {result.returncode}")
    raise SystemExit(result.returncode)
