#!/bin/bash

#SBATCH --job-name=trm

#SBATCH --output=saida_%j.log

#SBATCH --error=erro_%j.log

#SBATCH --time=03:00:00

#SBATCH --partition=h100n3

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G


python3 pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" evaluators="[]" epochs=50000 eval_interval=5000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=pretrain_sudoku  ema=True global_batch_size=64