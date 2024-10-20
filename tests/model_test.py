from src.fsrs_optimizer import *


class Test_Model:
    def test_next_stability(self):
        model = FSRS(DEFAULT_PARAMETER)
        stability = torch.tensor([5.0] * 4)
        difficulty = torch.tensor([1.0, 2.0, 3.0, 4.0])
        retention = torch.tensor([0.9, 0.8, 0.7, 0.6])
        rating = torch.tensor([1, 2, 3, 4])
        state = torch.stack([stability, difficulty]).unsqueeze(0)
        s_recall = model.stability_after_success(state, retention, rating)
        assert torch.allclose(
            s_recall, torch.tensor([25.7761, 14.1219, 60.4044, 208.9760]), atol=1e-4
        )
        s_forget = model.stability_after_failure(state, retention)
        assert torch.allclose(
            s_forget, torch.tensor([1.7029, 1.9799, 2.3760, 2.8885]), atol=1e-4
        )
        s_short_term = model.stability_short_term(state, rating)
        assert torch.allclose(
            s_short_term, torch.tensor([2.5051, 4.1992, 7.0389, 11.7988]), atol=1e-4
        )

    def test_next_difficulty(self):
        model = FSRS(DEFAULT_PARAMETER)
        stability = torch.tensor([5.0] * 4)
        difficulty = torch.tensor([5.0] * 4)
        rating = torch.tensor([1, 2, 3, 4])
        state = torch.stack([stability, difficulty]).unsqueeze(0)
        d_recall = model.next_d(state, rating)
        assert torch.allclose(
            d_recall,
            torch.tensor([6.6070, 5.7994, 4.9918, 4.1842]),
            atol=1e-4,
        )

    def test_power_forgetting_curve(self):
        delta_t = torch.tensor([0, 1, 2, 3, 4, 5])
        stability = torch.tensor([1, 2, 3, 4, 4, 2])
        retention = power_forgetting_curve(delta_t, stability)
        assert torch.allclose(
            retention,
            torch.tensor([1.0, 0.946059, 0.9299294, 0.9221679, 0.90000004, 0.79394597]),
            atol=1e-4,
        )

    def test_forward(self):
        model = FSRS(DEFAULT_PARAMETER)
        delta_ts = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            ]
        )
        ratings = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            ]
        )
        inputs = torch.stack([delta_ts, ratings], dim=2)
        _, state = model.forward(inputs)
        stability = state[:, 0]
        difficulty = state[:, 1]
        assert torch.allclose(
            stability,
            torch.tensor([0.2619, 1.7073, 5.8691, 25.0123, 0.3403, 2.1482]),
            atol=1e-4,
        )
        assert torch.allclose(
            difficulty,
            torch.tensor([8.0827, 7.0405, 5.2729, 2.1301, 8.0827, 7.0405]),
            atol=1e-4,
        )
