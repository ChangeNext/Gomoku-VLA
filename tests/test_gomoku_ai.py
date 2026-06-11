import unittest

import numpy as np

from board import GomokuBoard
from gomoku_ai import (
    MCTSConfig,
    ReplayBuffer,
    SelfPlayConfig,
    UniformPolicyValueModel,
    action_to_index,
    encode_board,
    generate_self_play_game,
    legal_action_mask,
    run_mcts,
)
from gomoku_ai.replay_buffer import augment_state_policy
from gomoku_ai.torch_model import GomokuPolicyValueNet
from gomoku_ai.train import train_epochs

import torch


class GomokuAITest(unittest.TestCase):
    def test_encode_board_uses_current_player_perspective(self) -> None:
        board = GomokuBoard(size=5)
        board.place(2, 2)
        encoded = encode_board(board)
        self.assertEqual(encoded.shape, (3, 5, 5))
        self.assertEqual(encoded[1, 2, 2], 1.0)
        self.assertEqual(encoded[2, 0, 0], 0.0)

    def test_legal_action_mask_excludes_occupied_cells(self) -> None:
        board = GomokuBoard(size=5)
        board.place(1, 3)
        mask = legal_action_mask(board)
        self.assertFalse(mask[action_to_index(1, 3, 5)])
        self.assertEqual(int(mask.sum()), 24)

    def test_mcts_returns_legal_probability_distribution(self) -> None:
        board = GomokuBoard(size=5)
        board.place(0, 0)
        policy = run_mcts(board, UniformPolicyValueModel(), MCTSConfig(simulations=8))
        self.assertEqual(policy.shape, (25,))
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        self.assertEqual(policy[action_to_index(0, 0, 5)], 0.0)

    def test_self_play_generates_training_samples(self) -> None:
        samples = generate_self_play_game(
            UniformPolicyValueModel(),
            SelfPlayConfig(board_size=5, win_length=4, mcts=MCTSConfig(simulations=4), max_moves=6),
        )
        self.assertGreater(len(samples), 0)
        self.assertLessEqual(len(samples), 6)
        self.assertEqual(samples[0].state.shape, (3, 5, 5))
        self.assertEqual(samples[0].policy_target.shape, (25,))
        self.assertTrue(np.isin(samples[0].value_target, [-1.0, 0.0, 1.0]))

    def test_torch_network_forward_shapes(self) -> None:
        network = GomokuPolicyValueNet(board_size=5, channels=8)
        states = torch.zeros((2, 3, 5, 5), dtype=torch.float32)
        policy_logits, values = network(states)
        self.assertEqual(tuple(policy_logits.shape), (2, 25))
        self.assertEqual(tuple(values.shape), (2,))

    def test_train_epochs_updates_from_replay_buffer(self) -> None:
        samples = generate_self_play_game(
            UniformPolicyValueModel(),
            SelfPlayConfig(board_size=5, win_length=4, mcts=MCTSConfig(simulations=2), max_moves=4),
        )
        replay = ReplayBuffer()
        replay.add_game(samples)
        network = GomokuPolicyValueNet(board_size=5, channels=8)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        loss = train_epochs(
            network,
            optimizer,
            replay,
            epochs=1,
            batches_per_epoch=1,
            batch_size=4,
            device=torch.device("cpu"),
        )
        self.assertGreater(loss, 0.0)

    def test_augmentation_preserves_state_policy_alignment(self) -> None:
        state = np.zeros((3, 3, 3), dtype=np.float32)
        state[0, 0, 1] = 1.0
        policy = np.zeros(9, dtype=np.float32)
        policy[action_to_index(0, 1, 3)] = 1.0

        for _ in range(20):
            augmented_state, augmented_policy = augment_state_policy(state, policy)
            state_position = np.argwhere(augmented_state[0] == 1.0)
            policy_position = np.argwhere(augmented_policy.reshape(3, 3) == 1.0)
            self.assertEqual(state_position.tolist(), policy_position.tolist())
            self.assertAlmostEqual(float(augmented_policy.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
