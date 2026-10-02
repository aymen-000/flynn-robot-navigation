from __future__ import annotations





from src.flynn.agents.connectome_rnn_agent import ConnectomeAgent
from src.flynn.config import (
    ACTIVATION, BATCH_CHUNK, CELL_TYPES_CSV, DESCENDING_NEURONS_CSV, EDGE_PATH, INPUT_SCALE_INIT, LEAK_ALPHA,
    PHOTORECEPTOR_LEFT_CSV, PHOTORECEPTOR_RIGHT_CSV, ROW_TILE_SIZE, TACTILE_LEFT_CSV, TACTILE_RIGHT_CSV,
    TARGET_RHO, WIND_SENSING_CSV,
)
from src.flynn.core.utils import build_connectome_cell

TRAINING_REGIMES = {
    "before_training": dict(train=False),

    "frozen_core": dict(train=True, train_rnn_weights=False, train_rnn_bias=False),

    "finetuned": dict(train=True, train_rnn_weights=True, train_rnn_bias=True),
}


def build_fresh_agent(device, dtype, train_rnn_weights: bool = False, train_readout_head: bool = True):

    cell, pr_positions, input_splits, id2idx = build_connectome_cell(
        edge_path=EDGE_PATH, device=device, dtype=dtype,
        photoreceptor_left_csv=PHOTORECEPTOR_LEFT_CSV, photoreceptor_right_csv=PHOTORECEPTOR_RIGHT_CSV,
        tactile_left_csv=TACTILE_LEFT_CSV, tactile_right_csv=TACTILE_RIGHT_CSV,
        descending_neurons_csv=DESCENDING_NEURONS_CSV, cell_types_csv=CELL_TYPES_CSV,
        wind_sensing_csv=WIND_SENSING_CSV, target_rho=TARGET_RHO, leak_alpha=LEAK_ALPHA,
        activation=ACTIVATION, train_rnn_weights=train_rnn_weights, train_readout_head=train_readout_head,
        batch_chunk=BATCH_CHUNK, row_tile_size=ROW_TILE_SIZE,
    )
    agent = ConnectomeAgent(
        cell, photoreceptor_positions=pr_positions, input_splits=input_splits,
        dtype=dtype, input_scale_init=INPUT_SCALE_INIT,
        use_value_head=False, learn_policy_std=True,
    ).to(device)
    return agent, dict(N=cell.N, Nin=cell.Nin, Nout=cell.Nout)
