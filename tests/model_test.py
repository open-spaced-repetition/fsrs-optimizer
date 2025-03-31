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
            s_recall,
            torch.tensor([25.361727, 13.676782, 59.194153, 205.02472]),
            atol=1e-4,
        )
        s_forget = model.stability_after_failure(state, retention)
        assert torch.allclose(
            s_forget,
            torch.tensor([1.929576, 2.2484288, 2.705061, 3.2972968]),
            atol=1e-4,
        )
        s_short_term = model.stability_short_term(state, rating)
        assert torch.allclose(
            s_short_term, torch.tensor([1.2750568, 2.4917638, 5.0, 9.516155]), atol=1e-4
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
            torch.tensor([7.1631646, 6.0708623, 4.9785595, 3.8862574]),
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
            torch.tensor(
                [0.23664382, 1.9285083, 6.27545, 26.054081, 0.23664382, 2.3679762]
            ),
            atol=1e-4,
        )
        assert torch.allclose(
            difficulty,
            torch.tensor([8.301044, 7.0668244, 4.9201818, 1.0, 8.301044, 7.0668244]),
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
        assert torch.allclose(loss, torch.tensor(4.5020204), atol=1e-4)
        loss.backward()
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    -0.04798106,
                    -0.0040113665,
                    -0.0011027358,
                    0.005099956,
                    0.04079412,
                    -0.057942636,
                    0.030063614,
                    -1.0625131,
                    0.60149425,
                    -3.0468569,
                    0.5935735,
                    -0.013909986,
                    0.038726028,
                    -0.11341163,
                    -0.00089108385,
                    -0.15296058,
                    0.22184022,
                    0.10291521,
                    0.004057862,
                    -0.33532578,
                ]
            ),
            atol=1e-4,
        )
        optimizer.step()
        assert torch.allclose(
            model.w,
            torch.tensor(
                [
                    0.3495,
                    1.4591999,
                    3.5492997,
                    15.9419,
                    7.0129,
                    0.6076,
                    1.9436,
                    0.0488,
                    1.4855001,
                    0.14739999,
                    0.9610999,
                    1.9166,
                    0.07110001,
                    0.3733,
                    2.3393996,
                    0.2649,
                    2.9604,
                    0.63,
                    0.36060008,
                    0.2232,
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
        assert torch.allclose(penalty, torch.tensor(0.6546369), atol=1e-4)
        penalty.backward()
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    0.0019813809,
                    0.00087788026,
                    0.00026506305,
                    -0.000105618295,
                    -0.25213888,
                    1.044897,
                    -0.22755535,
                    5.688889,
                    -0.5385926,
                    2.5283945,
                    -0.75225013,
                    0.9102214,
                    -10.113578,
                    3.1999993,
                    0.25213587,
                    1.3107198,
                    -0.07721739,
                    -0.85244584,
                    -0.79999804,
                    4.179591,
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
        retentions = power_forgetting_curve(delta_ts, stabilities)
        loss = loss_fn(retentions, labels).sum()
        assert torch.allclose(loss, torch.tensor(4.259659), atol=1e-4)
        loss.backward()
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    -0.025341026,
                    -0.0030165915,
                    -0.00084971637,
                    0.0044398247,
                    0.023752848,
                    0.022402849,
                    0.022914382,
                    -0.7292311,
                    0.59442425,
                    -2.7857566,
                    0.61564076,
                    -0.011727474,
                    0.030692013,
                    -0.08884726,
                    -0.00088726176,
                    -0.12694591,
                    0.22048862,
                    0.096262515,
                    0.015309425,
                    -0.28274217,
                ]
            ),
            atol=1e-4,
        )
        optimizer.step()
        assert torch.allclose(
            model.w,
            torch.tensor(
                [
                    0.38710356,
                    1.4985125,
                    3.5886993,
                    15.90214,
                    6.9747577,
                    0.62185836,
                    1.9042385,
                    0.08774383,
                    1.4455132,
                    0.18726659,
                    0.92106867,
                    1.9562788,
                    0.031605165,
                    0.41275662,
                    2.3793945,
                    0.3045352,
                    2.9204068,
                    0.59009206,
                    0.32496357,
                    0.26287907,
                ]
            ),
            atol=1e-4,
        )
