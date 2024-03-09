from src.fsrs_optimizer import *


class Test_Simulator:
    def test_simulate(self):
        (_, _, _, memorized_cnt_per_day, _) = simulate(
            w=DEFAULT_WEIGHT, forget_cost=125
        )
        assert memorized_cnt_per_day[-1] == 3145.3779679589484

    def test_optimal_retention(self):
        default_params = {
            "w": DEFAULT_WEIGHT,
            "deck_size": 10000,
            "learn_span": 1000,
            "max_cost_perday": math.inf,
            "learn_limit_perday": 10,
            "review_limit_perday": math.inf,
            "max_ivl": 36500,
            "recall_costs": np.array([14, 10, 6]),
            "forget_cost": 50,
            "learn_cost": 20,
            "first_rating_prob": np.array([0.15, 0.2, 0.6, 0.05]),
            "review_rating_prob": np.array([0.3, 0.6, 0.1]),
        }
        r = optimal_retention(**default_params)
        assert r == 0.7791796050312