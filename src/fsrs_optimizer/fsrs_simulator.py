import math
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


DECAY = -0.1542
FACTOR = 0.9 ** (1 / DECAY) - 1
S_MIN = 0.001


def power_forgetting_curve(t, s, decay=DECAY):
    factor = 0.9 ** (1 / decay) - 1
    return (1 + factor * t / s) ** decay


def next_interval(
    s, r, float_ivl: bool = False, fuzz: bool = False, decay: float = DECAY
):
    factor = 0.9 ** (1 / decay) - 1
    ivl = s / factor * (r ** (1 / decay) - 1)
    if float_ivl:
        ivl = np.round(ivl, 6)
    else:
        ivl = np.maximum(1, np.round(ivl).astype(int))
        if fuzz:
            fuzz_mask = ivl >= 3
            ivl[fuzz_mask] = fuzz_interval(ivl[fuzz_mask])
    return ivl


FUZZ_RANGES = [
    {
        "start": 2.5,
        "end": 7.0,
        "factor": 0.15,
    },
    {
        "start": 7.0,
        "end": 20.0,
        "factor": 0.1,
    },
    {
        "start": 20.0,
        "end": math.inf,
        "factor": 0.05,
    },
]


def get_fuzz_range(interval):
    delta = np.ones_like(interval, dtype=float)
    for range in FUZZ_RANGES:
        delta += range["factor"] * np.maximum(
            np.minimum(interval, range["end"]) - range["start"], 0.0
        )
    min_ivl = np.round(interval - delta).astype(int)
    max_ivl = np.round(interval + delta).astype(int)
    min_ivl = np.maximum(2, min_ivl)
    min_ivl = np.minimum(min_ivl, max_ivl)
    return min_ivl, max_ivl


def fuzz_interval(interval):
    min_ivl, max_ivl = get_fuzz_range(interval)
    # max_ivl + 1 because randint upper bound is exclusive
    return np.random.randint(min_ivl, max_ivl + 1, size=min_ivl.shape)


columns = [
    "difficulty",
    "stability",
    "retrievability",
    "delta_t",
    "reps",
    "lapses",
    "last_date",
    "due",
    "ivl",
    "cost",
    "rand",
    "rating",
    "ease",
]
col = {key: i for i, key in enumerate(columns)}

DEFAULT_LEARN_COSTS = np.array([33.79, 24.3, 13.68, 6.5])
DEFAULT_REVIEW_COSTS = np.array([23.0, 11.68, 7.33, 5.6])
DEFAULT_FIRST_RATING_PROB = np.array([0.24, 0.094, 0.495, 0.171])
DEFAULT_REVIEW_RATING_PROB = np.array([0.224, 0.631, 0.145])
DEFAULT_LEARNING_STEP_COUNT = 2
DEFAULT_RELEARNING_STEP_COUNT = 1
DEFAULT_LEARNING_STEP_TRANSITIONS = np.array(
    [
        [0.3687, 0.0628, 0.5108, 0.0577],
        [0.0441, 0.4553, 0.4457, 0.0549],
        [0.0518, 0.0470, 0.8462, 0.0550],
    ],
)
DEFAULT_RELEARNING_STEP_TRANSITIONS = np.array(
    [
        [0.2157, 0.0643, 0.6595, 0.0605],
        [0.0500, 0.4638, 0.4475, 0.0387],
        [0.1056, 0.1434, 0.7266, 0.0244],
    ],
)
DEFAULT_STATE_RATING_COSTS = np.array(
    [
        [12.75, 12.26, 8.0, 6.38],
        [13.05, 11.74, 7.42, 5.6],
        [10.56, 10.0, 7.37, 5.4],
    ]
)


def simulate(
    w,
    request_retention=0.9,
    deck_size=10000,
    learn_span=365,
    max_cost_perday=1800,
    learn_limit_perday=math.inf,
    review_limit_perday=math.inf,
    max_ivl=36500,
    first_rating_prob=DEFAULT_FIRST_RATING_PROB,
    review_rating_prob=DEFAULT_REVIEW_RATING_PROB,
    learning_step_count=DEFAULT_LEARNING_STEP_COUNT,
    relearning_step_count=DEFAULT_RELEARNING_STEP_COUNT,
    learning_step_transitions=DEFAULT_LEARNING_STEP_TRANSITIONS,
    relearning_step_transitions=DEFAULT_RELEARNING_STEP_TRANSITIONS,
    state_rating_costs=DEFAULT_STATE_RATING_COSTS,
    seed=42,
    fuzz=False,
    scheduler_name="fsrs",
):
    np.random.seed(seed)
    card_table = np.zeros((len(columns), deck_size))
    card_table[col["due"]] = learn_span
    card_table[col["difficulty"]] = 1e-10
    card_table[col["stability"]] = 1e-10
    card_table[col["rating"]] = np.random.choice(
        [1, 2, 3, 4], deck_size, p=first_rating_prob
    )
    card_table[col["rating"]] = card_table[col["rating"]].astype(int)
    card_table[col["ease"]] = 0

    revlogs = {}
    review_cnt_per_day = np.zeros(learn_span)
    learn_cnt_per_day = np.zeros(learn_span)
    memorized_cnt_per_day = np.zeros(learn_span)
    cost_per_day = np.zeros(learn_span)

    # Anki scheduler constants
    GRADUATING_IVL = 1
    EASY_IVL = 4
    NEW_IVL = 0
    HARD_IVL = 1.2
    INTERVAL_MODIFIER = 1.0
    EASY_BONUS = 1.3
    MIN_IVL = 1

    def anki_sm2_scheduler(scheduled_interval, real_interval, ease, rating):
        # Handle new cards (ease == 0)
        is_new_card = ease == 0
        new_card_intervals = np.where(rating == 4, EASY_IVL, GRADUATING_IVL)
        new_card_eases = np.full_like(ease, 2.5)

        # Handle review cards
        delay = real_interval - scheduled_interval

        # Calculate intervals for each rating
        again_interval = np.minimum(
            np.maximum(
                np.round(scheduled_interval * NEW_IVL * INTERVAL_MODIFIER + 0.01),
                MIN_IVL,
            ),
            max_ivl,
        )
        hard_interval = np.minimum(
            np.maximum(
                np.round(scheduled_interval * HARD_IVL * INTERVAL_MODIFIER + 0.01),
                np.maximum(scheduled_interval + 1, MIN_IVL),
            ),
            max_ivl,
        )
        good_interval = np.minimum(
            np.maximum(
                np.round(
                    (scheduled_interval + delay / 2) * ease * INTERVAL_MODIFIER + 0.01
                ),
                np.maximum(hard_interval + 1, MIN_IVL),
            ),
            max_ivl,
        )
        easy_interval = np.minimum(
            np.maximum(
                np.round(real_interval * ease * INTERVAL_MODIFIER * EASY_BONUS + 0.01),
                np.maximum(good_interval + 1, MIN_IVL),
            ),
            max_ivl,
        )

        # Select intervals based on rating
        review_intervals = np.choose(
            rating - 1, [again_interval, hard_interval, good_interval, easy_interval]
        )

        # Calculate new eases
        review_eases = np.choose(
            rating - 1,
            [
                ease - 0.2,
                ease - 0.15,
                ease,
                ease + 0.15,
            ],
        )
        review_eases = np.maximum(review_eases, 1.3)

        # Combine new card and review card results
        intervals = np.where(is_new_card, new_card_intervals, review_intervals)
        eases = np.where(is_new_card, new_card_eases, review_eases)

        return intervals, eases

    def stability_after_success(s, r, d, rating):
        hard_penalty = np.where(rating == 2, w[15], 1)
        easy_bonus = np.where(rating == 4, w[16], 1)
        return np.maximum(
            S_MIN,
            s
            * (
                1
                + np.exp(w[8])
                * (11 - d)
                * np.power(s, -w[9])
                * (np.exp((1 - r) * w[10]) - 1)
                * hard_penalty
                * easy_bonus
            ),
        )

    def stability_after_failure(s, r, d):
        return np.maximum(
            S_MIN,
            np.minimum(
                w[11]
                * np.power(d, -w[12])
                * (np.power(s + 1, w[13]) - 1)
                * np.exp((1 - r) * w[14]),
                s / np.exp(w[17] * w[18]),
            ),
        )

    MAX_RELEARN_STEPS = 5

    # learn_state: 1: Learning, 2: Review, 3: Relearning
    def memory_state_short_term(
        s: np.ndarray, d: np.ndarray, init_rating: Optional[np.ndarray] = None
    ):
        if init_rating is not None:
            s = np.choose(init_rating - 1, w)
            d = np.clip(init_d(init_rating), 1, 10)
            costs = state_rating_costs[0]
            max_consecutive = learning_step_count - np.choose(
                init_rating - 1, [0, 0, 1, 1]
            )
            cost = np.choose(init_rating - 1, costs).sum()
        else:
            costs = state_rating_costs[2]
            max_consecutive = relearning_step_count
            cost = 0

        def step(s, next_weights):
            rating = np.random.choice([1, 2, 3, 4], p=next_weights)
            sinc = (math.e ** (w[17] * (rating - 3 + w[18]))) * (s ** -w[19])
            new_s = s * (sinc.clip(min=1) if rating >= 3 else sinc)

            return (new_s, rating)

        def loop(s, d, max_consecutive, init_rating):
            i = 0
            consecutive = 0
            step_transitions = (
                relearning_step_transitions
                if init_rating is None
                else learning_step_transitions
            )
            rating = init_rating or 1
            cost = 0
            while (
                i < MAX_RELEARN_STEPS and consecutive < max_consecutive and rating < 4
            ):
                (s, rating) = step(s, step_transitions[rating - 1])
                d = next_d(d, rating)
                cost += costs[rating - 1]
                i += 1
                if rating > 2:
                    consecutive += 1
                elif rating == 1:
                    consecutive = 0

            return s, d, cost

        if len(s) != 0:
            new_s, new_d, cost = np.vectorize(loop, otypes=["float", "float", "float"])(
                s, d, max_consecutive, init_rating
            )
        else:
            new_s, new_d, cost = [], [], []

        return new_s, new_d, cost

    def init_d(rating):
        return w[4] - np.exp(w[5] * (rating - 1)) + 1

    def linear_damping(delta_d, old_d):
        return delta_d * (10 - old_d) / 9

    def next_d(d, rating):
        delta_d = -w[6] * (rating - 3)
        new_d = d + linear_damping(delta_d, d)
        new_d = mean_reversion(init_d(4), new_d)
        return np.clip(new_d, 1, 10)

    def mean_reversion(init, current):
        return w[7] * init + (1 - w[7]) * current

    for today in range(learn_span):
        new_s = np.copy(card_table[col["stability"]])
        new_d = np.copy(card_table[col["difficulty"]])

        has_learned = card_table[col["stability"]] > 1e-10
        card_table[col["delta_t"]][has_learned] = (
            today - card_table[col["last_date"]][has_learned]
        )
        card_table[col["retrievability"]][has_learned] = power_forgetting_curve(
            card_table[col["delta_t"]][has_learned],
            card_table[col["stability"]][has_learned],
            -w[20],
        )
        card_table[col["cost"]] = 0
        need_review = card_table[col["due"]] <= today
        card_table[col["rand"]][need_review] = np.random.rand(np.sum(need_review))
        forget = card_table[col["rand"]] > card_table[col["retrievability"]]
        card_table[col["rating"]][need_review & forget] = 1
        card_table[col["rating"]][need_review & ~forget] = np.random.choice(
            [2, 3, 4], np.sum(need_review & ~forget), p=review_rating_prob
        )
        card_table[col["cost"]][need_review] = np.choose(
            card_table[col["rating"]][need_review].astype(int) - 1,
            state_rating_costs[1],
        )
        true_review = need_review & (np.cumsum(need_review) <= review_limit_perday)
        card_table[col["last_date"]][true_review] = today

        card_table[col["lapses"]][true_review & forget] += 1
        card_table[col["reps"]][true_review & ~forget] += 1

        new_s[true_review & forget] = stability_after_failure(
            card_table[col["stability"]][true_review & forget],
            card_table[col["retrievability"]][true_review & forget],
            card_table[col["difficulty"]][true_review & forget],
        )
        new_d[true_review & forget] = next_d(
            card_table[col["difficulty"]][true_review & forget],
            card_table[col["rating"]][true_review & forget],
        )
        (
            card_table[col["stability"]][true_review & forget],
            card_table[col["difficulty"]][true_review & forget],
            costs,
        ) = memory_state_short_term(
            new_s[true_review & forget],
            new_d[true_review & forget],
        )
        new_s[true_review & ~forget] = stability_after_success(
            new_s[true_review & ~forget],
            card_table[col["retrievability"]][true_review & ~forget],
            new_d[true_review & ~forget],
            card_table[col["rating"]][true_review & ~forget],
        )

        new_d[true_review & ~forget] = next_d(
            card_table[col["difficulty"]][true_review & ~forget],
            card_table[col["rating"]][true_review & ~forget],
        )

        card_table[col["cost"]][true_review & forget] = [
            a + b for a, b in zip(card_table[col["cost"]][true_review & forget], costs)
        ]

        need_learn = card_table[col["stability"]] == 1e-10
        card_table[col["cost"]][need_learn] = np.choose(
            card_table[col["rating"]][need_learn].astype(int) - 1,
            state_rating_costs[0],
        )
        true_learn = need_learn & (np.cumsum(need_learn) <= learn_limit_perday)
        card_table[col["last_date"]][true_learn] = today
        new_s[true_learn] = np.choose(
            card_table[col["rating"]][true_learn].astype(int) - 1, w[:4]
        )
        (
            new_s[true_learn],
            new_d[true_learn],
            costs,
        ) = memory_state_short_term(
            new_s[true_learn],
            new_d[true_learn],
            init_rating=card_table[col["rating"]][true_learn].astype(int),
        )

        card_table[col["cost"]][true_learn] = [
            a + b for a, b in zip(card_table[col["cost"]][true_learn], costs)
        ]

        below_cost_limit = np.cumsum(card_table[col["cost"]]) <= max_cost_perday
        reviewed = (true_review | true_learn) & below_cost_limit

        card_table[col["stability"]][reviewed] = new_s[reviewed]
        card_table[col["difficulty"]][reviewed] = new_d[reviewed]

        if scheduler_name == "fsrs":
            card_table[col["ivl"]][reviewed] = np.clip(
                next_interval(
                    card_table[col["stability"]][reviewed],
                    request_retention,
                    fuzz=fuzz,
                    decay=-w[20],
                ),
                1,
                max_ivl,
            )
            card_table[col["due"]][reviewed] = today + card_table[col["ivl"]][reviewed]
        else:  # anki scheduler
            scheduled_intervals = card_table[col["ivl"]][reviewed]
            eases = card_table[col["ease"]][reviewed]
            real_intervals = card_table[col["delta_t"]][reviewed]
            ratings = card_table[col["rating"]][reviewed].astype(int)

            delta_ts, new_eases = anki_sm2_scheduler(
                scheduled_intervals, real_intervals, eases, ratings
            )
            card_table[col["ivl"]][reviewed] = delta_ts
            card_table[col["due"]][reviewed] = today + delta_ts
            card_table[col["ease"]][reviewed] = new_eases

        revlogs[today] = {
            "card_id": np.where(reviewed)[0],
            "rating": card_table[col["rating"]][reviewed],
        }

        has_learned = card_table[col["stability"]] > 1e-10
        card_table[col["delta_t"]][has_learned] = (
            today - card_table[col["last_date"]][has_learned]
        )
        card_table[col["retrievability"]][has_learned] = power_forgetting_curve(
            card_table[col["delta_t"]][has_learned],
            card_table[col["stability"]][has_learned],
        )

        review_cnt_per_day[today] = np.sum(true_review & reviewed)
        learn_cnt_per_day[today] = np.sum(true_learn & reviewed)
        memorized_cnt_per_day[today] = card_table[col["retrievability"]].sum()
        cost_per_day[today] = card_table[col["cost"]][reviewed].sum()
    return (
        card_table,
        review_cnt_per_day,
        learn_cnt_per_day,
        memorized_cnt_per_day,
        cost_per_day,
        revlogs,
    )


def optimal_retention(**kwargs):
    return brent(**kwargs)


CMRR_TARGET_WORKLOAD_ONLY = True
CMRR_TARGET_MEMORIZED_PER_WORKLOAD = False
CMRR_TARGET_MEMORIZED_STABILITY_PER_WORKLOAD = "memorized_stability_per_workload"


def run_simulation(args):
    target, kwargs = args

    (card_table, _, _, memorized_cnt_per_day, cost_per_day, _) = simulate(**kwargs)

    if target == CMRR_TARGET_WORKLOAD_ONLY:
        return np.mean(cost_per_day)
    if target == CMRR_TARGET_MEMORIZED_PER_WORKLOAD:
        return np.sum(cost_per_day) / memorized_cnt_per_day[-1]
    if target == CMRR_TARGET_MEMORIZED_STABILITY_PER_WORKLOAD:
        return np.sum(cost_per_day) / np.sum(
            np.max(card_table[col["stability"]], 0) * card_table[col["retrievability"]]
        )


def sample(
    r,
    w,
    deck_size=10000,
    learn_span=365,
    max_cost_perday=1800,
    learn_limit_perday=math.inf,
    review_limit_perday=math.inf,
    max_ivl=36500,
    first_rating_prob=DEFAULT_FIRST_RATING_PROB,
    review_rating_prob=DEFAULT_REVIEW_RATING_PROB,
    learning_step_transitions=DEFAULT_LEARNING_STEP_TRANSITIONS,
    relearning_step_transitions=DEFAULT_RELEARNING_STEP_TRANSITIONS,
    state_rating_costs=DEFAULT_STATE_RATING_COSTS,
    workload_only=CMRR_TARGET_MEMORIZED_PER_WORKLOAD,
):
    results = []

    def best_sample_size(days_to_simulate):
        if days_to_simulate <= 30:
            return 45
        elif days_to_simulate >= 365:
            return 4
        else:
            a1, a2, a3 = 8.20e-07, 2.41e-03, 1.30e-02
            factor = a1 * np.power(days_to_simulate, 2) + a2 * days_to_simulate + a3
            default_sample_size = 4
            return int(default_sample_size / factor)

    SAMPLE_SIZE = best_sample_size(learn_span)

    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(SAMPLE_SIZE):
            kwargs = {
                "w": w,
                "request_retention": r,
                "deck_size": deck_size,
                "learn_span": learn_span,
                "max_cost_perday": max_cost_perday,
                "learn_limit_perday": learn_limit_perday,
                "review_limit_perday": review_limit_perday,
                "max_ivl": max_ivl,
                "first_rating_prob": first_rating_prob,
                "review_rating_prob": review_rating_prob,
                "learning_step_transitions": learning_step_transitions,
                "relearning_step_transitions": relearning_step_transitions,
                "state_rating_costs": state_rating_costs,
                "seed": 42 + i,
            }
            futures.append(executor.submit(run_simulation, (workload_only, kwargs)))

        for future in as_completed(futures):
            results.append(future.result())
    return np.mean(results)


def brent(tol=0.01, maxiter=20, **kwargs):
    mintol = 1.0e-11
    cg = 0.3819660

    xb = 0.70
    fb = sample(xb, **kwargs)
    funccalls = 1

    #################################
    # BEGIN
    #################################
    x = w = v = xb
    fw = fv = fx = fb
    a = 0.70
    b = 0.95
    deltax = 0.0
    iter = 0

    while iter < maxiter:
        tol1 = tol * np.abs(x) + mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        # check for convergence
        if np.abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break

        if np.abs(deltax) <= tol1:
            if x >= xmid:
                deltax = a - x  # do a golden section step
            else:
                deltax = b - x
            rat = cg * deltax
        else:  # do a parabolic step
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                p = -p
            tmp2 = np.abs(tmp2)
            dx_temp = deltax
            deltax = rat
            # check parabolic fit
            if (
                (p > tmp2 * (a - x))
                and (p < tmp2 * (b - x))
                and (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))
            ):
                rat = p * 1.0 / tmp2  # if parabolic step is useful
                u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    if xmid - x >= 0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if x >= xmid:
                    deltax = a - x  # if it's not do a golden section step
                else:
                    deltax = b - x
                rat = cg * deltax

        if np.abs(rat) < tol1:  # update by at least tol1
            if rat >= 0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu = sample(u, **kwargs)  # calculate new output value
        funccalls += 1

        if fu > fx:  # if it's bigger than current
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu

        iter += 1
    #################################
    # END
    #################################

    xmin = x
    fval = fx

    success = (
        iter < maxiter
        and not (np.isnan(fval) or np.isnan(xmin))
        and (0.70 <= xmin <= 0.95)
    )

    if success:
        return xmin
    else:
        raise Exception("The algorithm terminated without finding a valid value.")


def workload_graph(default_params, sampling_size=30):
    R = np.linspace(0.7, 0.999, sampling_size).tolist()
    default_params["max_cost_perday"] = math.inf
    default_params["learn_limit_perday"] = int(
        default_params["deck_size"] / default_params["learn_span"]
    )
    default_params["review_limit_perday"] = math.inf
    workload = [sample(r=r, workload_only=True, **default_params) for r in R]

    # this is for testing
    # workload = [min(x, 2.3 * min(workload)) for x in workload]
    min_w = min(workload)  # minimum workload
    max_w = max(workload)  # maximum workload
    min1_index = R.index(R[workload.index(min_w)])

    min_w2 = 0
    min_w3 = 0
    target2 = 1.5 * min_w
    target3 = 2 * min_w

    for i in range(len(workload) - 1):
        if (workload[i] <= target2) and (workload[i + 1] >= target2):
            if abs(workload[i] - target2) < abs(workload[i + 1] - target2):
                min_w2 = workload[i]
            else:
                min_w2 = workload[i + 1]

    for i in range(len(workload) - 1):
        if (workload[i] <= target3) and (workload[i + 1] >= target3):
            if abs(workload[i] - target3) < abs(workload[i + 1] - target3):
                min_w3 = workload[i]
            else:
                min_w3 = workload[i + 1]

    if min_w2 == 0:
        min2_index = len(R)
    else:
        min2_index = R.index(R[workload.index(min_w2)])

    min1_5_index = int(math.ceil((min2_index + 3 * min1_index) / 4))
    if min_w3 == 0:
        min3_index = len(R)
    else:
        min3_index = R.index(R[workload.index(min_w3)])

    fig = plt.figure(figsize=(16, 8))
    ax = fig.gca()
    if min1_index > 0:
        ax.fill_between(
            x=R[: min1_index + 1],
            y1=0,
            y2=workload[: min1_index + 1],
            color="red",
            alpha=1,
        )
        ax.fill_between(
            x=R[min1_index : min1_5_index + 1],
            y1=0,
            y2=workload[min1_index : min1_5_index + 1],
            color="gold",
            alpha=1,
        )
    else:
        # handle the case when there is no red area to the left
        ax.fill_between(
            x=R[: min1_5_index + 1],
            y1=0,
            y2=workload[: min1_5_index + 1],
            color="gold",
            alpha=1,
        )

    ax.fill_between(
        x=R[min1_5_index : min2_index + 1],
        y1=0,
        y2=workload[min1_5_index : min2_index + 1],
        color="limegreen",
        alpha=1,
    )
    ax.fill_between(
        x=R[min2_index : min3_index + 1],
        y1=0,
        y2=workload[min2_index : min3_index + 1],
        color="gold",
        alpha=1,
    )
    ax.fill_between(
        x=R[min3_index:],
        y1=0,
        y2=workload[min3_index:],
        color="red",
        alpha=1,
    )
    ax.set_yticks([])
    ax.set_xticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    ax.xaxis.set_tick_params(labelsize=14)
    ax.set_xlim(0.7, 0.99)

    if max_w >= 3.5 * min_w:
        lim = 3.5 * min_w
    elif max_w >= 3 * min_w:
        lim = 3 * min_w
    elif max_w >= 2.5 * min_w:
        lim = 2.5 * min_w
    elif max_w >= 2 * min_w:
        lim = 2 * min_w
    else:
        lim = 1.1 * max_w

    ax.set_ylim(0, lim)
    ax.set_ylabel("Workload (minutes of study per day)", fontsize=20)
    ax.set_xlabel("Desired Retention", fontsize=20)
    ax.axhline(y=min_w, color="black", alpha=0.75, ls="--")
    ax.text(
        0.701,
        min_w,
        "minimum workload",
        ha="left",
        va="bottom",
        color="black",
        fontsize=12,
    )
    if lim >= 1.8 * min_w:
        ax.axhline(y=1.5 * min_w, color="black", alpha=0.75, ls="--")
        ax.text(
            0.701,
            1.5 * min_w,
            "minimum workload x1.5",
            ha="left",
            va="bottom",
            color="black",
            fontsize=12,
        )
    if lim >= 2.3 * min_w:
        ax.axhline(y=2 * min_w, color="black", alpha=0.75, ls="--")
        ax.text(
            0.701,
            2 * min_w,
            "minimum workload x2",
            ha="left",
            va="bottom",
            color="black",
            fontsize=12,
        )
    if lim >= 2.8 * min_w:
        ax.axhline(y=2.5 * min_w, color="black", alpha=0.75, ls="--")
        ax.text(
            0.701,
            2.5 * min_w,
            "minimum workload x2.5",
            ha="left",
            va="bottom",
            color="black",
            fontsize=12,
        )
    if lim >= 3.3 * min_w:
        ax.axhline(y=3 * min_w, color="black", alpha=0.75, ls="--")
        ax.text(
            0.701,
            3 * min_w,
            "minimum workload x3",
            ha="left",
            va="bottom",
            color="black",
            fontsize=12,
        )
    fig.tight_layout(h_pad=0, w_pad=0)
    return fig


if __name__ == "__main__":
    default_params = {
        "w": [
            0.2172,
            1.1771,
            3.2602,
            16.1507,
            7.0114,
            0.57,
            2.0966,
            0.0069,
            1.5261,
            0.112,
            1.0178,
            1.849,
            0.1133,
            0.3127,
            2.2934,
            0.2191,
            3.0004,
            0.7536,
            0.3332,
            0.1437,
            0.2,
        ],
        "deck_size": 20000,
        "learn_span": 365,
        "max_cost_perday": 1800,
        "learn_limit_perday": math.inf,
        "review_limit_perday": math.inf,
        "max_ivl": 36500,
    }

    schedulers = ["fsrs", "anki"]
    for scheduler_name in schedulers:
        (
            _,
            review_cnt_per_day,
            learn_cnt_per_day,
            memorized_cnt_per_day,
            _,
            revlogs,
        ) = simulate(
            w=default_params["w"],
            max_cost_perday=math.inf,
            learn_limit_perday=10,
            review_limit_perday=50,
            scheduler_name=scheduler_name,
        )

        def moving_average(data, window_size=365 // 20):
            weights = np.ones(window_size) / window_size
            return np.convolve(data, weights, mode="valid")

        plt.figure(1)
        plt.plot(
            moving_average(review_cnt_per_day),
            label=scheduler_name,
        )
        plt.title("Review Count per Day")
        plt.legend()
        plt.grid(True)

        plt.figure(2)
        plt.plot(
            moving_average(learn_cnt_per_day),
            label=scheduler_name,
        )
        plt.title("Learn Count per Day")
        plt.legend()
        plt.grid(True)

        plt.figure(3)
        plt.plot(
            np.cumsum(learn_cnt_per_day),
            label=scheduler_name,
        )
        plt.title("Cumulative Learn Count")
        plt.legend()
        plt.grid(True)

        plt.figure(4)
        plt.plot(
            memorized_cnt_per_day,
            label=scheduler_name,
        )
        plt.title("Memorized Count per Day")
        plt.legend()
        plt.grid(True)

        plt.figure(5)
        plt.plot(
            [
                sum(rating > 1 for rating in day["rating"]) / len(day["rating"])
                for _, day in sorted(revlogs.items(), key=lambda a: a[0])
            ],
            label=scheduler_name,
        )
        plt.ylim(0, 1)
        plt.title("True retention per day")
        plt.legend()
        plt.grid(True)

    plt.show()
    workload_graph(default_params, sampling_size=30).savefig("workload.png")
