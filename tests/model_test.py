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
            torch.tensor(
                [
                    25.578495025634766,
                    13.550500869750977,
                    59.86878967285156,
                    207.70382690429688,
                ]
            ),
            atol=1e-4,
        )
        s_forget = model.stability_after_failure(state, retention)
        assert torch.allclose(
            s_forget,
            torch.tensor(
                [
                    1.7469292879104614,
                    2.0312795639038086,
                    2.4401676654815674,
                    2.9707436561584473,
                ]
            ),
            atol=1e-4,
        )
        s_short_term = model.stability_short_term(state, rating)
        assert torch.allclose(
            s_short_term,
            torch.tensor(
                [
                    1.1298232078552246,
                    2.4004619121551514,
                    5.100105285644531,
                    10.835862159729004,
                ]
            ),
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
            torch.tensor(
                [
                    7.2961106300354,
                    6.139369487762451,
                    4.98262882232666,
                    3.82588791847229,
                ]
            ),
            atol=1e-4,
        )

    def test_power_forgetting_curve(self):
        delta_t = torch.tensor([0, 1, 2, 3, 4, 5])
        stability = torch.tensor([1, 2, 3, 4, 4, 2])
        retention = power_forgetting_curve(delta_t, stability)
        assert torch.allclose(
            retention,
            torch.tensor(
                [
                    1.0,
                    0.9421982765197754,
                    0.9268093109130859,
                    0.91965252161026,
                    0.9,
                    0.8178008198738098,
                ]
            ),
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
                    0.16648849844932556,
                    1.6992956399917603,
                    6.414825439453125,
                    28.05109977722168,
                    0.16896963119506836,
                    2.0530757904052734,
                ]
            ),
            atol=1e-4,
        )
        assert torch.allclose(
            difficulty,
            torch.tensor(
                [
                    8.362964630126953,
                    7.086328506469727,
                    4.868056774139404,
                    1.0,
                    8.362964630126953,
                    7.086328506469727,
                ]
            ),
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
        assert torch.allclose(loss, torch.tensor(4.514678), atol=1e-4)
        loss.backward()
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    -0.09797611,
                    -0.0072790897,
                    -0.0013130547,
                    0.005998566,
                    0.040757798,
                    -0.059734505,
                    0.030936658,
                    -1.0551243,
                    0.5905801,
                    -3.1485195,
                    0.5726495,
                    -0.020666562,
                    0.055198845,
                    -0.17501271,
                    -0.0013422093,
                    -0.15273242,
                    0.21408933,
                    0.11237629,
                    -0.0053924676,
                    -0.4327011,
                ]
            ),
            atol=1e-4,
        )
        optimizer.step()
        assert torch.allclose(
            model.w,
            torch.tensor(
                [
                    0.2572,
                    1.2170999,
                    3.3001997,
                    16.1107,
                    6.9714003,
                    0.61,
                    2.0566,
                    0.0469,
                    1.4861001,
                    0.15200001,
                    0.97779995,
                    1.8889999,
                    0.07330002,
                    0.3527,
                    2.3333998,
                    0.2591,
                    2.9604,
                    0.7136,
                    0.37319994,
                    0.1837,
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
        assert torch.allclose(penalty, torch.tensor(0.654637), atol=1e-4)
        penalty.backward()
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    0.0019813816,
                    0.00087788026,
                    0.00026506305,
                    -0.00010561578,
                    -0.25213888,
                    1.0448985,
                    -0.22755535,
                    5.688889,
                    -0.5385926,
                    2.5283954,
                    -0.75225013,
                    0.9102214,
                    -10.113576,
                    3.1999993,
                    0.2521374,
                    1.3107198,
                    -0.07721739,
                    -0.85244584,
                    0.79999864,
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
        assert torch.allclose(loss, torch.tensor(4.263608), atol=1e-4)
        loss.backward()
        assert torch.allclose(
            model.w.grad,
            torch.tensor(
                [
                    -0.050412238,
                    -0.005721763,
                    -0.001122501,
                    0.005703689,
                    0.02302191,
                    0.018940464,
                    0.022635777,
                    -0.69668454,
                    0.5801687,
                    -2.8480258,
                    0.5906929,
                    -0.017377893,
                    0.043433353,
                    -0.13754296,
                    -0.0012391449,
                    -0.12037965,
                    0.20936891,
                    0.11671813,
                    0.017347742,
                    -0.37295583,
                ]
            ),
            atol=1e-4,
        )
        optimizer.step()
        assert torch.allclose(
            model.w,
            torch.tensor(
                [
                    0.29465374,
                    1.2565677,
                    3.3399148,
                    16.070766,
                    6.9334216,
                    0.6265469,
                    2.0173912,
                    0.08566975,
                    1.4461204,
                    0.19184539,
                    0.9377726,
                    1.9286716,
                    0.033828788,
                    0.39216724,
                    2.3732843,
                    0.29857677,
                    2.920426,
                    0.6735678,
                    0.35272712,
                    0.2234361,
                ]
            ),
            atol=1e-4,
        )
