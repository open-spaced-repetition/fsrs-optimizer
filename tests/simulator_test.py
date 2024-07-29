from src.fsrs_optimizer import *


class Test_Simulator:
    def test_simulate(self):
        (
            card_table,
            review_cnt_per_day,
            learn_cnt_per_day,
            memorized_cnt_per_day,
            cost_per_day,
        ) = simulate(w=DEFAULT_PARAMETER)
        card_table.to_csv("card_table.csv", index=False)
        print(review_cnt_per_day)
        print(learn_cnt_per_day)
        print(memorized_cnt_per_day)
        print(cost_per_day)
        # assert memorized_cnt_per_day[-1] == 3145.3779679589484

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
        assert r == 0.7791796050312
