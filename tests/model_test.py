from fsrs_optimizer import *


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
            s_recall,
            torch.tensor([25.602541, 28.226582, 58.656002, 127.226685]),
            atol=1e-4,
        )
        s_forget = model.stability_after_failure(state, retention)
        assert torch.allclose(
            s_forget,
            torch.tensor([1.0525396, 1.1894329, 1.3680838, 1.584989]),
            atol=1e-4,
        )
        s_short_term = model.stability_short_term(state, rating)
        assert torch.allclose(
            s_short_term,
            torch.tensor([1.596818, 2.7470093, 5.0, 8.12961]),
            atol=1e-4,
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
            torch.tensor([8.341763, 6.6659956, 4.990228, 3.3144615]),
            atol=1e-4,
        )

    def test_power_forgetting_curve(self):
        delta_t = torch.tensor([0, 1, 2, 3, 4, 5])
        stability = torch.tensor([1, 2, 3, 4, 4, 2])
        retention = power_forgetting_curve(delta_t, stability)
        assert torch.allclose(
            retention,
            torch.tensor([1.0, 0.9403443, 0.9253786, 0.9185229, 0.9, 0.8261359]),
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
            torch.tensor(
                [
                    0.10088589,
                    3.2494123,
                    7.3153,
                    18.014914,
                    0.112798266,
                    4.4694576,
                ]
            ),
            atol=1e-4,
        )
        assert torch.allclose(
            difficulty,
            torch.tensor([8.806304, 6.7404594, 2.1112142, 1.0, 8.806304, 6.7404594]),
            atol=1e-4,
        )

    def test_loss_and_grad(self):
        model = FSRS(DEFAULT_PARAMETER)
        clipper = ParameterClipper()
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
        retentions = power_forgetting_curve(delta_ts, stabilities, -model.w[20])
        loss = loss_fn(retentions, labels).sum()
        assert torch.allclose(loss, torch.tensor(4.047898), atol=1e-4)
        loss.backward()
        assert model.w.grad is not None
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    -0.095688485,
                    -0.0051607806,
                    -0.00080300873,
                    0.007462064,
                    0.03677408,
                    -0.084962785,
                    0.059571628,
                    -2.1566951,
                    0.5738574,
                    -2.8749206,
                    0.7123072,
                    -0.028993709,
                    0.0099172965,
                    -0.2189217,
                    -0.0017800558,
                    -0.089381434,
                    0.299141,
                    0.0708902,
                    -0.01219162,
                    -0.25424173,
                    0.27452517,
                ]
            ),
            atol=1e-4,
        )
        optimizer.step()
        assert torch.allclose(
            model.w,
            torch.tensor(
                [
                    0.252,
                    1.3331,
                    2.3464994,
                    8.2556,
                    6.3733,
                    0.87340003,
                    2.9794,
                    0.040999997,
                    1.8322,
                    0.20660001,
                    0.756,
                    1.5235,
                    0.021400042,
                    0.3029,
                    1.6882998,
                    0.64140004,
                    1.8329,
                    0.5025,
                    0.13119997,
                    0.1058,
                    0.1142,
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
        assert torch.allclose(penalty, torch.tensor(0.6771115), atol=1e-4)
        penalty.backward()
        assert model.w.grad is not None
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    0.0019813816,
                    0.00087788026,
                    0.00026506148,
                    -0.000105618295,
                    -0.25213888,
                    1.0448985,
                    -0.22755535,
                    5.688889,
                    -0.5385926,
                    2.5283954,
                    -0.75225013,
                    0.9102214,
                    -10.113569,
                    3.1999993,
                    0.2521374,
                    1.3107208,
                    -0.07721739,
                    -0.85244584,
                    0.79999936,
                    4.1795917,
                    -1.1237311,
                ]
            ),
            atol=1e-5,
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
        retentions = power_forgetting_curve(delta_ts, stabilities, -model.w[20])
        loss = loss_fn(retentions, labels).sum()
        assert torch.allclose(loss, torch.tensor(3.76888), atol=1e-4)
        loss.backward()
        assert model.w.grad is not None
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    -0.040530164,
                    -0.0041278866,
                    -0.0006833144,
                    0.007239434,
                    0.009416521,
                    -0.12156768,
                    0.039193563,
                    -0.86553144,
                    0.57743585,
                    -2.571437,
                    0.76415884,
                    -0.024242667,
                    0.0,
                    -0.16912507,
                    -0.0017008218,
                    -0.061857328,
                    0.28093633,
                    0.06636292,
                    0.0057900245,
                    -0.19041246,
                    0.6214733,
                ]
            ),
            atol=1e-4,
        )
        optimizer.step()
        model.apply(clipper)
        assert torch.allclose(
            model.w,
            torch.tensor(
                [
                    0.2882918,
                    1.3726242,
                    2.3862023,
                    8.215636,
                    6.339949,
                    0.9131501,
                    2.940647,
                    0.07696302,
                    1.7921939,
                    0.2464219,
                    0.71595156,
                    1.5631561,
                    0.001,
                    0.34230903,
                    1.7282416,
                    0.68038,
                    1.7929853,
                    0.46259063,
                    0.1426339,
                    0.14509763,
                    0.1,
                ]
            ),
            atol=1e-4,
        )
