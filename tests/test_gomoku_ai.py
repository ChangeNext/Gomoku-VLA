import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from board import GomokuBoard, Player
from gomoku_ai import (
    CheckpointPolicy,
    MCTSConfig,
    MovePrediction,
    ReplayBuffer,
    SelfPlayConfig,
    UniformPolicyValueModel,
    action_to_index,
    encode_board,
    generate_self_play_game,
    generate_self_play_games,
    legal_action_mask,
    predict_move,
    run_mcts,
    run_mcts_batch,
    select_tactical_move,
)
from gomoku_ai.replay_buffer import augment_state_policy
from gomoku_ai.torch_model import GomokuPolicyValueNet, TorchPolicyValueModel, load_checkpoint, save_checkpoint
from gomoku_ai.train import (
    AlphaZeroTrainingConfig,
    plot_training_history,
    run_training,
    train_epochs,
    write_training_history_csv,
)
from scripts.evaluate_checkpoint import play_match_game

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

    def test_legal_action_mask_excludes_renju_forbidden_moves(self) -> None:
        board = GomokuBoard(rule_set="renju")
        for row, col in ((7, 6), (7, 8), (6, 7), (8, 7)):
            board.grid[row][col] = 1
            board.move_count += 1

        mask = legal_action_mask(board)
        self.assertFalse(mask[action_to_index(7, 7, 15)])

    def test_mcts_returns_legal_probability_distribution(self) -> None:
        board = GomokuBoard(size=5)
        board.place(0, 0)
        policy = run_mcts(board, UniformPolicyValueModel(), MCTSConfig(simulations=8))
        self.assertEqual(policy.shape, (25,))
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        self.assertEqual(policy[action_to_index(0, 0, 5)], 0.0)

    def test_batched_mcts_returns_policy_for_each_board(self) -> None:
        boards = [GomokuBoard(size=5), GomokuBoard(size=5)]
        boards[0].place(0, 0)
        boards[1].place(1, 1)
        policies = run_mcts_batch(boards, UniformPolicyValueModel(), MCTSConfig(simulations=4))
        self.assertEqual(len(policies), 2)
        for policy in policies:
            self.assertEqual(policy.shape, (25,))
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        self.assertEqual(policies[0][action_to_index(0, 0, 5)], 0.0)
        self.assertEqual(policies[1][action_to_index(1, 1, 5)], 0.0)

    def test_self_play_generates_training_samples(self) -> None:
        samples = generate_self_play_game(
            UniformPolicyValueModel(),
            SelfPlayConfig(board_size=5, win_length=4, rule_set="free", mcts=MCTSConfig(simulations=4), max_moves=6),
        )
        self.assertGreater(len(samples), 0)
        self.assertLessEqual(len(samples), 6)
        self.assertEqual(samples[0].state.shape, (3, 5, 5))
        self.assertEqual(samples[0].policy_target.shape, (25,))
        self.assertTrue(np.isin(samples[0].value_target, [-1.0, 0.0, 1.0]))

    def test_batched_self_play_generates_multiple_games(self) -> None:
        games = generate_self_play_games(
            UniformPolicyValueModel(),
            game_count=3,
            config=SelfPlayConfig(board_size=5, win_length=4, rule_set="free", mcts=MCTSConfig(simulations=2), max_moves=4),
        )
        self.assertEqual(len(games), 3)
        self.assertTrue(all(0 < len(samples) <= 4 for samples in games))

    def test_torch_network_forward_shapes(self) -> None:
        network = GomokuPolicyValueNet(board_size=5, channels=8, input_channels=3)
        states = torch.zeros((2, 3, 5, 5), dtype=torch.float32)
        policy_logits, values = network(states)
        self.assertEqual(tuple(policy_logits.shape), (2, 25))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(network.architecture, "resnet")

    def test_checkpoint_round_trip_preserves_architecture_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            network = GomokuPolicyValueNet(board_size=5, channels=8, res_blocks=2)
            save_checkpoint(network, path, metadata={"training_config": {"device": "cpu"}})
            loaded = load_checkpoint(path)

            self.assertEqual(loaded.board_size, 5)
            self.assertEqual(loaded.channels, 8)
            self.assertEqual(loaded.res_blocks, 2)
            self.assertEqual(loaded.architecture, "resnet")

    def test_legacy_checkpoint_loads_without_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.pt"
            network = GomokuPolicyValueNet(board_size=5, channels=8, architecture="legacy_cnn")
            torch.save({"board_size": 5, "state_dict": network.state_dict()}, path)

            loaded = load_checkpoint(path)
            self.assertEqual(loaded.architecture, "legacy_cnn")
            self.assertEqual(loaded.channels, 8)

    def test_train_epochs_updates_from_replay_buffer(self) -> None:
        samples = generate_self_play_game(
            UniformPolicyValueModel(),
            SelfPlayConfig(board_size=5, win_length=4, rule_set="free", mcts=MCTSConfig(simulations=2), max_moves=4),
        )
        replay = ReplayBuffer()
        replay.add_game(samples)
        network = GomokuPolicyValueNet(board_size=5, channels=8, input_channels=3)
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

    def test_training_run_writes_structured_outputs_and_replay(self) -> None:
        with TemporaryDirectory() as tmpdir:
            history = run_training(
                AlphaZeroTrainingConfig(
                    board_size=5,
                    win_length=4,
                    rule_set="free",
                    enforce_center_opening=False,
                    iterations=1,
                    games_per_iteration=1,
                    mcts_simulations=2,
                    epochs=1,
                    batches_per_epoch=1,
                    batch_size=4,
                    runs_dir=tmpdir,
                    run_name="smoke",
                    channels=8,
                    res_blocks=1,
                )
            )
            run_dir = Path(tmpdir) / "smoke"
            self.assertEqual(len(history), 1)
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "checkpoints" / "latest.pt").exists())
            self.assertTrue((run_dir / "metrics" / "history.csv").exists())
            self.assertTrue((run_dir / "plots" / "training.png").exists())
            self.assertTrue((run_dir / "replay" / "replay_buffer.pkl").exists())

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

    def test_replay_buffer_save_and_load_preserves_samples(self) -> None:
        samples = generate_self_play_game(
            UniformPolicyValueModel(),
            SelfPlayConfig(board_size=5, win_length=4, rule_set="free", mcts=MCTSConfig(simulations=2), max_moves=4),
        )
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.pkl"
            replay = ReplayBuffer(capacity=16)
            replay.add_game(samples)
            replay.save(path)

            loaded = ReplayBuffer.load(path)
            self.assertEqual(len(loaded), len(replay))
            self.assertEqual(loaded.capacity, 16)

    def test_tactical_move_prefers_win_and_blocks_loss(self) -> None:
        winning_board = GomokuBoard(size=5, win_length=4)
        for col in range(3):
            winning_board.place(0, col)
            winning_board.place(1, col)
        self.assertEqual(select_tactical_move(winning_board), (0, 3))

        blocking_board = GomokuBoard(size=5, win_length=4)
        blocking_board.place(4, 4)
        for col in range(3):
            blocking_board.place(0, col)
            if col < 2:
                blocking_board.place(1, col)
        self.assertEqual(select_tactical_move(blocking_board), (0, 3))

    def test_predict_move_returns_policy_value_and_move(self) -> None:
        board = GomokuBoard(size=5, win_length=4)
        prediction = predict_move(
            board,
            UniformPolicyValueModel(),
            MCTSConfig(simulations=4, temperature=0.0),
        )

        self.assertIsInstance(prediction, MovePrediction)
        self.assertIn(prediction.move, board.legal_moves())
        self.assertEqual(prediction.policy.shape, (25,))
        self.assertAlmostEqual(float(prediction.policy.sum()), 1.0, places=5)
        self.assertEqual(prediction.action_index, action_to_index(prediction.row, prediction.col, 5))
        self.assertEqual(prediction.value, 0.0)

    def test_predict_move_can_sample_from_policy_distribution(self) -> None:
        class FakeRng:
            def choice(self, size: int, p: np.ndarray) -> int:
                self.size = size
                self.probabilities = p
                return 4

        board = GomokuBoard(size=5, win_length=4)
        policy = np.zeros(25, dtype=np.float32)
        policy[0] = 0.8
        policy[4] = 0.2
        fake_rng = FakeRng()
        with patch("gomoku_ai.inference.run_mcts", return_value=policy):
            prediction = predict_move(
                board,
                UniformPolicyValueModel(),
                MCTSConfig(simulations=1, temperature=1.0),
                use_tactics=False,
                sample_move=True,
                rng=fake_rng,
            )

        self.assertEqual(prediction.action_index, 4)
        self.assertEqual(prediction.move, (0, 4))
        self.assertEqual(fake_rng.size, 25)
        self.assertAlmostEqual(float(fake_rng.probabilities.sum()), 1.0)

    def test_predict_move_uses_tactical_override(self) -> None:
        board = GomokuBoard(size=5, win_length=4)
        for col in range(3):
            board.place(0, col)
            board.place(1, col)

        prediction = predict_move(board, UniformPolicyValueModel(), MCTSConfig(simulations=1))

        self.assertEqual(prediction.move, (0, 3))
        self.assertTrue(prediction.used_tactical_move)
        self.assertEqual(float(prediction.policy[action_to_index(0, 3, 5)]), 1.0)

    def test_checkpoint_policy_loads_metadata_and_predicts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            network = GomokuPolicyValueNet(
                board_size=5,
                channels=8,
                res_blocks=1,
                rule_set="renju",
                enforce_center_opening=True,
            )
            save_checkpoint(network, path)

            policy = CheckpointPolicy(path, device="cpu", simulations=2, use_tactics=False)
            board = policy.new_board()
            prediction = policy.predict(board)

            self.assertEqual(policy.board_size, 5)
            self.assertEqual(policy.rule_set, "renju")
            self.assertTrue(policy.enforce_center_opening)
            self.assertEqual(board.rule_set, "renju")
            self.assertTrue(board.enforce_center_opening)
            self.assertIn(prediction.move, board.legal_moves())

    def test_checkpoint_policy_switches_to_late_temperature_after_opening(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            network = GomokuPolicyValueNet(board_size=5, channels=8, res_blocks=1)
            save_checkpoint(network, path)

            policy = CheckpointPolicy(
                path,
                device="cpu",
                simulations=2,
                temperature=1.0,
                temperature_moves=1,
                late_temperature=0.1,
                sample_moves=True,
                use_tactics=False,
            )
            board = policy.new_board(enforce_center_opening=False)
            board.place(0, 0)
            captured: dict[str, float] = {}

            def fake_run_mcts(board: GomokuBoard, model: UniformPolicyValueModel, config: MCTSConfig) -> np.ndarray:
                captured["temperature"] = config.temperature
                output = np.zeros(board.size * board.size, dtype=np.float32)
                output[action_to_index(1, 1, board.size)] = 1.0
                return output

            with patch("gomoku_ai.inference.run_mcts", side_effect=fake_run_mcts):
                prediction = policy.predict(board)

            self.assertEqual(prediction.move, (1, 1))
            self.assertEqual(captured["temperature"], 0.1)

    def test_tactical_move_skips_renju_forbidden_opponent_moves(self) -> None:
        board = GomokuBoard(rule_set="renju")
        board.current_player = Player.WHITE
        for row, col in ((7, 6), (7, 8), (6, 7), (8, 7)):
            board.grid[row][col] = 1
            board.move_count += 1

        self.assertNotEqual(select_tactical_move(board), (7, 7))

    def test_evaluator_game_runs_between_checkpoints(self) -> None:
        candidate = GomokuPolicyValueNet(board_size=3, channels=4, res_blocks=1)
        baseline = GomokuPolicyValueNet(board_size=3, channels=4, res_blocks=1)
        winner, moves = play_match_game(
            candidate=TorchPolicyValueModel(candidate),
            baseline=TorchPolicyValueModel(baseline),
            board_size=3,
            win_length=3,
            rule_set="free",
            enforce_center_opening=False,
            simulations=2,
            candidate_is_black=True,
            opening_random_moves=1,
            rng=__import__("random").Random(0),
        )
        self.assertIn(winner, {"candidate", "baseline", "draw"})
        self.assertGreater(moves, 0)

    def test_training_history_writes_csv_and_plot(self) -> None:
        history = [
            {
                "iteration": 1.0,
                "samples_added": 10.0,
                "replay_size": 10.0,
                "train_steps": 2.0,
                "loss": 3.0,
            },
            {
                "iteration": 2.0,
                "samples_added": 12.0,
                "replay_size": 22.0,
                "train_steps": 2.0,
                "loss": 2.5,
            },
        ]
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "history.csv"
            plot_path = Path(tmpdir) / "history.png"
            write_training_history_csv(history, csv_path)
            plot_training_history(history, plot_path)

            self.assertIn(
                "iteration,samples_added,replay_size,train_steps,loss,policy_loss,value_loss,policy_target_entropy",
                csv_path.read_text(),
            )
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_training_rejects_unavailable_cuda_device(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA device was requested"):
                run_training(AlphaZeroTrainingConfig(device="cuda", rule_set="free", enforce_center_opening=False))


if __name__ == "__main__":
    unittest.main()
