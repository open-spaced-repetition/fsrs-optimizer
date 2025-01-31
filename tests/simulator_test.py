from src.fsrs_optimizer import *


class Test_Simulator:
    def test_simulate(self):
        (
            card_table,
            review_cnt_per_day,
            learn_cnt_per_day,
            memorized_cnt_per_day,
            cost_per_day,
            revlogs,
        ) = simulate(w=DEFAULT_PARAMETER, request_retention=0.9)
        assert memorized_cnt_per_day[-1] == 5960.836176338407

    def test_optimal_retention(self):
        default_params = {
            "w": DEFAULT_PARAMETER,
            "deck_size": 10000,
            "learn_span": 1000,
            "max_cost_perday": math.inf,
            "learn_limit_perday": 10,
            "review_limit_perday": math.inf,
            "max_ivl": 36500,
            "loss_aversion": 2.5,
        }
        r = optimal_retention(**default_params)
        assert r == 0.8254596913507394
