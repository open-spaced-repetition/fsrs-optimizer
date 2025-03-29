from src.fsrs_optimizer import *

# Use `DEFAULT_PARAMETER` when updated
TEMP_PARAMETERS = [*DEFAULT_PARAMETER, 0.1832]

class Test_Simulator:
    def test_simulate(self):
        (
            card_table,
            review_cnt_per_day,
            learn_cnt_per_day,
            memorized_cnt_per_day,
            cost_per_day,
            revlogs,
        ) = simulate(w=TEMP_PARAMETERS, request_retention=0.9)
        assert memorized_cnt_per_day[-1] == 6064.7049740089315

    def test_optimal_retention(self):
        default_params = {
            "w": TEMP_PARAMETERS,
            "deck_size": 10000,
            "learn_span": 1000,
            "max_cost_perday": math.inf,
            "learn_limit_perday": 10,
            "review_limit_perday": math.inf,
            "max_ivl": 36500,
            "loss_aversion": 2.5,
        }
        r = optimal_retention(**default_params)
        assert r == 0.86305357865811
