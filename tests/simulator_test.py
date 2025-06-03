from src.fsrs_optimizer import *

FSRS_RS_MEMORIZED = 5361.807


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
        deviation = abs(1 - (memorized_cnt_per_day[-1] / FSRS_RS_MEMORIZED))
        assert (
            deviation < 0.06
        ), f"{memorized_cnt_per_day[-1]:.2f} is not within 5% of the expected {FSRS_RS_MEMORIZED:.2f} ({deviation:.2%} deviation)"

    def test_optimal_retention(self):
        default_params = {
            "w": DEFAULT_PARAMETER,
            "deck_size": 10000,
            "learn_span": 1000,
            "max_cost_perday": math.inf,
            "learn_limit_perday": 10,
            "review_limit_perday": math.inf,
            "max_ivl": 36500,
        }
        r = optimal_retention(**default_params)
        assert r == 0.7
