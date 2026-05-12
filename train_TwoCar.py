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

import datetime
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

cmd = [
    sys.executable,
    '../libraries/DeepReach_MPC/run_experiment.py',
    '--mode',
    'train',
    '--experiment_name',
    timestamp,
    '--dynamics_class',
    'TwoCar8D',
    '--tMax',
    '1',
    '--pretrain',
    '--pretrain_iters',
    '1000',
    '--num_epochs',
    '104000',
    '--counter_end',
    '100000',
    '--num_nl',
    '512',
    '--collisionR',   ## Current  collision radius is 5.0. This is probably too large
    '5.0',
    '--wheelbase',   ## Wheelbase for two-car system
    '2.7',
    '--set_mode',
    'avoid',
    '--lr',
    '2e-5',
    '--num_MPC_batches',
    '20',
    '--MPC_batch_size',
    '5000'
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
subprocess.run(cmd)