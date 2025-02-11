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
            torch.tensor([0.2619, 1.7074, 5.8691, 25.0124, 0.2859, 2.1482]),
            atol=1e-4,
        )
        assert torch.allclose(
            difficulty,
            torch.tensor([8.0827, 7.0405, 5.2729, 2.1301, 8.0827, 7.0405]),
            atol=1e-4,
        )

    def test_loss_and_grad(self):
        model = FSRS(DEFAULT_PARAMETER)
        init_w = torch.tensor(DEFAULT_PARAMETER)
        params_stddev = DEFAULT_PARAMS_STDDEV_TENSOR
        optimizer = torch.optim.Adam(model.parameters(), lr=4e-2)
        loss_fn = nn.BCELoss(reduction="none")
        t_histories = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 3.0],
                [1.0, 3.0, 3.0, 5.0],
                [3.0, 6.0, 6.0, 12.0],
            ]
        )
        r_histories = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 2.0, 4.0],
                [1.0, 4.0, 4.0, 3.0],
                [4.0, 3.0, 3.0, 3.0],
                [3.0, 1.0, 3.0, 3.0],
                [2.0, 3.0, 3.0, 4.0],
            ]
        )
        delta_ts = torch.tensor([4.0, 11.0, 12.0, 23.0])
        labels = torch.tensor([1, 1, 1, 0], dtype=torch.float32, requires_grad=False)
        inputs = torch.stack([t_histories, r_histories], dim=2)
        seq_lens = inputs.shape[0]
        real_batch_size = inputs.shape[1]
        outputs, _ = model.forward(inputs)
        stabilities = outputs[seq_lens - 1, torch.arange(real_batch_size), 0]
        retentions = power_forgetting_curve(delta_ts, stabilities)
        loss = loss_fn(retentions, labels).sum()
        assert round(loss.item(), 4) == 4.4467
        loss.backward()
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    -0.0583,
                    -0.0068,
                    -0.0026,
                    0.0105,
                    -0.0513,
                    1.3643,
                    0.0837,
                    -0.9502,
                    0.5345,
                    -2.8929,
                    0.5142,
                    -0.0131,
                    0.0419,
                    -0.1183,
                    -0.0009,
                    -0.1445,
                    0.2024,
                    0.2141,
                    0.0323,
                    -0.1561,
                ]
            ),
            atol=1e-4,
        )
        optimizer.step()
        assert torch.allclose(
            model.w,
            torch.tensor(
                [
                    0.44255,
                    1.22385,
                    3.2129998,
                    15.65105,
                    7.2349,
                    0.4945,
                    1.4204,
                    0.0446,
                    1.5057501,
                    0.1592,
                    0.97925,
                    1.9794999,
                    0.07000001,
                    0.33605,
                    2.3097994,
                    0.2715,
                    2.9498,
                    0.47655,
                    0.62210006,
                    0.54,
                ]
            ),
            atol=1e-4,
        )

        optimizer.zero_grad()
        penalty = (
            torch.sum(torch.square(model.w - init_w) / torch.square(params_stddev))
            * 512
            / 1000
            * 2
        )
        assert round(penalty.item(), 4) == 0.6485
        penalty.backward()
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    0.0018749383,
                    0.00090389,
                    0.00026177685,
                    -0.00010645759,
                    0.27080965,
                    -1.0448978,
                    -0.18249036,
                    5.688889,
                    -0.5119995,
                    2.528395,
                    -0.7086509,
                    1.1237301,
                    -12.799997,
                    4.179591,
                    0.25213587,
                    1.3107198,
                    -0.07721739,
                    -1.1237309,
                    -0.5385926,
                    0.0819,
                ]
            ),
            atol=1e-4,
        )

        optimizer.zero_grad()
        t_histories = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 3.0],
                [1.0, 3.0, 3.0, 5.0],
                [3.0, 6.0, 6.0, 12.0],
            ]
        )
        r_histories = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 2.0, 4.0],
                [1.0, 4.0, 4.0, 3.0],
                [4.0, 3.0, 3.0, 3.0],
                [3.0, 1.0, 3.0, 3.0],
                [2.0, 3.0, 3.0, 4.0],
            ]
        )
        delta_ts = torch.tensor([4.0, 11.0, 12.0, 23.0])
        labels = torch.tensor([1, 1, 1, 0], dtype=torch.float32, requires_grad=False)
        inputs = torch.stack([t_histories, r_histories], dim=2)
        outputs, _ = model.forward(inputs)
        stabilities = outputs[seq_lens - 1, torch.arange(real_batch_size), 0]
        retentions = power_forgetting_curve(delta_ts, stabilities)
        loss = loss_fn(retentions, labels).sum()
        assert round(loss.item(), 6) == 4.171336
        loss.backward()
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    -0.038874,
                    -0.005887,
                    -0.002892,
                    0.012191,
                    -0.056227,
                    1.145189,
                    0.067944,
                    -0.688681,
                    0.486917,
                    -2.536179,
                    0.48966,
                    -0.011007,
                    0.03588,
                    -0.091596,
                    -0.000902,
                    -0.127628,
                    0.190617,
                    0.259652,
                    0.050606,
                    -0.116076,
                ]
            ),
            atol=1e-4,
        )
        optimizer.step()
        assert torch.allclose(
            model.w,
            torch.tensor(
                [
                    0.481363,
                    1.263588,
                    3.253052,
                    15.611004,
                    7.274954,
                    0.454833,
                    1.380828,
                    0.083771,
                    1.46589,
                    0.198977,
                    0.939313,
                    2.019176,
                    0.03028,
                    0.375467,
                    2.349772,
                    0.311294,
                    2.90988,
                    0.436534,
                    0.58259,
                    0.579271,
                ]
            ),
            atol=1e-4,
        )
