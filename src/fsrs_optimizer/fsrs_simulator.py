import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange


DECAY = -0.5
FACTOR = 0.9 ** (1 / DECAY) - 1


def power_forgetting_curve(t, s):
    return (1 + FACTOR * t / s) ** DECAY


def next_interval(s, r):
    ivl = s / FACTOR * (r ** (1 / DECAY) - 1)
    return np.maximum(1, np.round(ivl))


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
]
col = {key: i for i, key in enumerate(columns)}

DEFAULT_LEARN_COSTS = np.array([33.79, 24.3, 13.68, 6.5])
DEFAULT_REVIEW_COSTS = np.array([23.0, 11.68, 7.33, 5.6])
DEFAULT_FIRST_RATING_PROB = np.array([0.24, 0.094, 0.495, 0.171])
DEFAULT_REVIEW_RATING_PROB = np.array([0.224, 0.631, 0.145])
DEFAULT_FIRST_RATING_OFFSETS = np.array([-0.72, -0.15, -0.01, 0.0])
DEFAULT_FIRST_SESSION_LENS = np.array([2.02, 1.28, 0.81, 0.0])
DEFAULT_FORGET_RATING_OFFSET = -0.28
DEFAULT_FORGET_SESSION_LEN = 1.05


def simulate(
    w,
    request_retention=0.9,
    deck_size=10000,
    learn_span=365,
    max_cost_perday=1800,
    learn_limit_perday=math.inf,
    review_limit_perday=math.inf,
    max_ivl=36500,
    learn_costs=DEFAULT_LEARN_COSTS,
    review_costs=DEFAULT_REVIEW_COSTS,
    first_rating_prob=DEFAULT_FIRST_RATING_PROB,
    review_rating_prob=DEFAULT_REVIEW_RATING_PROB,
    first_rating_offset=DEFAULT_FIRST_RATING_OFFSETS,
    first_session_len=DEFAULT_FIRST_SESSION_LENS,
    forget_rating_offset=DEFAULT_FORGET_RATING_OFFSET,
    forget_session_len=DEFAULT_FORGET_SESSION_LEN,
    loss_aversion=2.5,
    seed=42,
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

    review_cnt_per_day = np.zeros(learn_span)
    learn_cnt_per_day = np.zeros(learn_span)
    memorized_cnt_per_day = np.zeros(learn_span)
    cost_per_day = np.zeros(learn_span)

    def stability_after_success(s, r, d, rating):
        hard_penalty = np.where(rating == 2, w[15], 1)
        easy_bonus = np.where(rating == 4, w[16], 1)
        return np.maximum(
            0.01,
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
            0.01,
            np.minimum(
                w[11]
                * np.power(d, -w[12])
                * (np.power(s + 1, w[13]) - 1)
                * np.exp((1 - r) * w[14]),
                s,
            ),
        )

    def stability_short_term(s, init_rating=None):
        if init_rating is not None:
            rating_offset = np.choose(init_rating - 1, first_rating_offset)
            session_len = np.choose(init_rating - 1, first_session_len)
        else:
            rating_offset = forget_rating_offset
            session_len = forget_session_len
        new_s = s * np.exp(w[17] * (rating_offset + session_len * w[18]))
        return new_s

    def init_d(rating):
        return w[4] - np.exp(w[5] * (rating - 1)) + 1

    def init_d_with_short_term(rating):
        rating_offset = np.choose(rating - 1, first_rating_offset)
        new_d = init_d(rating) - w[6] * rating_offset
        return np.clip(new_d, 1, 10)

    def next_d(d, rating):
        new_d = d - w[6] * (rating - 3)
        new_d = mean_reversion(init_d(4), new_d)
        return np.clip(new_d, 1, 10)

    def mean_reversion(init, current):
        return w[7] * init + (1 - w[7]) * current

    for today in trange(learn_span, position=1, leave=False):
        has_learned = card_table[col["stability"]] > 1e-10
        card_table[col["delta_t"]][has_learned] = (
            today - card_table[col["last_date"]][has_learned]
        )
        card_table[col["retrievability"]][has_learned] = power_forgetting_curve(
            card_table[col["delta_t"]][has_learned],
            card_table[col["stability"]][has_learned],
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
            review_costs,
        )
        card_table[col["cost"]][need_review & forget] *= loss_aversion
        true_review = (
            need_review
            & (np.cumsum(card_table[col["cost"]]) <= max_cost_perday)
            & (np.cumsum(need_review) <= review_limit_perday)
        )
        card_table[col["last_date"]][true_review] = today

        card_table[col["lapses"]][true_review & forget] += 1
        card_table[col["reps"]][true_review & ~forget] += 1

        card_table[col["stability"]][true_review & forget] = stability_after_failure(
            card_table[col["stability"]][true_review & forget],
            card_table[col["retrievability"]][true_review & forget],
            card_table[col["difficulty"]][true_review & forget],
        )
        card_table[col["stability"]][true_review & forget] = stability_short_term(
            card_table[col["stability"]][true_review & forget]
        )
        card_table[col["stability"]][true_review & ~forget] = stability_after_success(
            card_table[col["stability"]][true_review & ~forget],
            card_table[col["retrievability"]][true_review & ~forget],
            card_table[col["difficulty"]][true_review & ~forget],
            card_table[col["rating"]][true_review & ~forget],
        )

        card_table[col["difficulty"]][true_review] = next_d(
            card_table[col["difficulty"]][true_review],
            card_table[col["rating"]][true_review],
        )
        card_table[col["difficulty"]][true_review & forget] = np.clip(
            card_table[col["difficulty"]][true_review & forget]
            - (w[6] * forget_rating_offset),
            1,
            10,
        )

        need_learn = card_table[col["due"]] == learn_span
        card_table[col["cost"]][need_learn] = np.choose(
            card_table[col["rating"]][need_learn].astype(int) - 1,
            learn_costs,
        )
        true_learn = (
            need_learn
            & (np.cumsum(card_table[col["cost"]]) <= max_cost_perday)
            & (np.cumsum(need_learn) <= learn_limit_perday)
        )
        card_table[col["last_date"]][true_learn] = today
        card_table[col["stability"]][true_learn] = np.choose(
            card_table[col["rating"]][true_learn].astype(int) - 1, w[:4]
        )
        card_table[col["stability"]][true_learn] = stability_short_term(
            card_table[col["stability"]][true_learn],
            init_rating=card_table[col["rating"]][true_learn].astype(int),
        )
        card_table[col["difficulty"]][true_learn] = init_d_with_short_term(
            card_table[col["rating"]][true_learn].astype(int)
        )

        card_table[col["ivl"]][true_review | true_learn] = np.clip(
            next_interval(
                card_table[col["stability"]][true_review | true_learn],
                request_retention,
            ),
            1,
            max_ivl,
        )
        card_table[col["due"]][true_review | true_learn] = (
            today + card_table[col["ivl"]][true_review | true_learn]
        )

        review_cnt_per_day[today] = np.sum(true_review)
        learn_cnt_per_day[today] = np.sum(true_learn)
        memorized_cnt_per_day[today] = card_table[col["retrievability"]].sum()
        cost_per_day[today] = card_table[col["cost"]][true_review | true_learn].sum()
    return (
        card_table,
        review_cnt_per_day,
        learn_cnt_per_day,
        memorized_cnt_per_day,
        cost_per_day,
    )


def optimal_retention(**kwargs):
    return brent(**kwargs)


def sample(
    r,
    w,
    deck_size=10000,
    learn_span=365,
    max_cost_perday=1800,
    learn_limit_perday=math.inf,
    review_limit_perday=math.inf,
    max_ivl=36500,
    learn_costs=DEFAULT_LEARN_COSTS,
    review_costs=DEFAULT_REVIEW_COSTS,
    first_rating_prob=DEFAULT_FIRST_RATING_PROB,
    review_rating_prob=DEFAULT_REVIEW_RATING_PROB,
    first_rating_offset=DEFAULT_FIRST_RATING_OFFSETS,
    first_session_len=DEFAULT_FIRST_SESSION_LENS,
    forget_rating_offset=DEFAULT_FORGET_RATING_OFFSET,
    forget_session_len=DEFAULT_FORGET_SESSION_LEN,
    loss_aversion=2.5,
):
    memorization = []
    if learn_span < 100:
        SAMPLE_SIZE = 16
    elif learn_span < 365:
        SAMPLE_SIZE = 8
    else:
        SAMPLE_SIZE = 4

    for i in range(SAMPLE_SIZE):
        _, _, _, memorized_cnt_per_day, cost_per_day = simulate(
            w,
            r,
            deck_size,
            learn_span,
            max_cost_perday,
            learn_limit_perday,
            review_limit_perday,
            max_ivl,
            learn_costs,
            review_costs,
            first_rating_prob,
            review_rating_prob,
            first_rating_offset,
            first_session_len,
            forget_rating_offset,
            forget_session_len,
            loss_aversion,
            seed=42 + i,
        )
        memorization.append(cost_per_day.sum() / memorized_cnt_per_day[-1])
    return np.mean(memorization)


def brent(tol=0.01, maxiter=20, **kwargs):
    mintol = 1.0e-11
    cg = 0.3819660

    funccalls = 0
    xb = 0.75
    fb = sample(xb, **kwargs)

    #################################
    # BEGIN
    #################################
    x = w = v = xb
    fw = fv = fx = fb
    a = 0.75
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
        and (0.75 <= xmin <= 0.95)
    )

    if success:
        return xmin
    else:
        raise Exception("The algorithm terminated without finding a valid value.")


def workload_graph(default_params):
    R = [x / 100 for x in range(70, 100)]
    cost_per_memorization = [sample(r=r, **default_params) for r in R]

    # this is for testing
    # cost_per_memorization = [min(x, 2.3 * min(cost_per_memorization)) for x in cost_per_memorization]
    min_w = min(cost_per_memorization)  # minimum workload
    max_w = max(cost_per_memorization)  # maximum workload
    min1_index = R.index(R[cost_per_memorization.index(min_w)])

    min_w2 = 0
    min_w3 = 0
    target2 = 2 * min_w
    target3 = 3 * min_w

    for i in range(len(cost_per_memorization) - 1):
        if (cost_per_memorization[i] <= target2) and (
            cost_per_memorization[i + 1] >= target2
        ):
            if abs(cost_per_memorization[i] - target2) < abs(
                cost_per_memorization[i + 1] - target2
            ):
                min_w2 = cost_per_memorization[i]
            else:
                min_w2 = cost_per_memorization[i + 1]

    for i in range(len(cost_per_memorization) - 1):
        if (cost_per_memorization[i] <= target3) and (
            cost_per_memorization[i + 1] >= target3
        ):
            if abs(cost_per_memorization[i] - target3) < abs(
                cost_per_memorization[i + 1] - target3
            ):
                min_w3 = cost_per_memorization[i]
            else:
                min_w3 = cost_per_memorization[i + 1]

    if min_w2 == 0:
        min2_index = len(R)
    else:
        min2_index = R.index(R[cost_per_memorization.index(min_w2)])

    min1_5_index = int(math.ceil((min2_index + 3 * min1_index) / 4))
    if min_w3 == 0:
        min3_index = len(R)
    else:
        min3_index = R.index(R[cost_per_memorization.index(min_w3)])

    fig = plt.figure(figsize=(16, 8))
    ax = fig.gca()
    if min1_index > 0:
        ax.fill_between(
            x=R[: min1_index + 1],
            y1=0,
            y2=cost_per_memorization[: min1_index + 1],
            color="red",
            alpha=1,
        )
        ax.fill_between(
            x=R[min1_index : min1_5_index + 1],
            y1=0,
            y2=cost_per_memorization[min1_index : min1_5_index + 1],
            color="gold",
            alpha=1,
        )
    else:
        # handle the case when there is no red area to the left
        ax.fill_between(
            x=R[: min1_5_index + 1],
            y1=0,
            y2=cost_per_memorization[: min1_5_index + 1],
            color="gold",
            alpha=1,
        )

    ax.fill_between(
        x=R[min1_5_index : min2_index + 1],
        y1=0,
        y2=cost_per_memorization[min1_5_index : min2_index + 1],
        color="limegreen",
        alpha=1,
    )
    ax.fill_between(
        x=R[min2_index : min3_index + 1],
        y1=0,
        y2=cost_per_memorization[min2_index : min3_index + 1],
        color="gold",
        alpha=1,
    )
    ax.fill_between(
        x=R[min3_index:],
        y1=0,
        y2=cost_per_memorization[min3_index:],
        color="red",
        alpha=1,
    )
    ax.set_yticks([])
    ax.set_xticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    ax.xaxis.set_tick_params(labelsize=14)
    ax.set_xlim(0.7, 0.99)

    if max_w >= 4.5 * min_w:
        lim = 4.5 * min_w
    elif max_w >= 3.5 * min_w:
        lim = 3.5 * min_w
    else:
        lim = max_w

    ax.set_ylim(0, lim)
    ax.set_ylabel("Workload (minutes of study per day)", fontsize=20)
    ax.set_xlabel("Retention", fontsize=20)
    ax.axhline(y=min_w, color="black", alpha=0.75, ls="--")
    ax.text(
        0.701,
        min_w,
        "min. workload",
        ha="left",
        va="bottom",
        color="black",
        fontsize=12,
    )
    if max_w >= 2.3 * min_w:
        ax.axhline(y=2 * min_w, color="black", alpha=0.75, ls="--")
        ax.text(
            0.701,
            2 * min_w,
            "min. workload x2",
            ha="left",
            va="bottom",
            color="black",
            fontsize=12,
        )
    if max_w >= 3.3 * min_w:
        ax.axhline(y=3 * min_w, color="black", alpha=0.75, ls="--")
        ax.text(
            0.701,
            3 * min_w,
            "min. workload x3",
            ha="left",
            va="bottom",
            color="black",
            fontsize=12,
        )
    if max_w >= 4.3 * min_w:
        ax.axhline(y=4 * min_w, color="black", alpha=0.75, ls="--")
        ax.text(
            0.701,
            4 * min_w,
            "min. workload x4",
            ha="left",
            va="bottom",
            color="black",
            fontsize=12,
        )

    return fig


if __name__ == "__main__":
    default_params = {
        "w": [
            0.4197,
            1.1869,
            3.0412,
            15.2441,
            7.1434,
            0.6477,
            1.0007,
            0.0674,
            1.6597,
            0.1712,
            1.1178,
            2.0225,
            0.0904,
            0.3025,
            2.1214,
            0.2498,
            2.9466,
            0.4891,
            0.6468,
        ],
        "deck_size": 20000,
        "learn_span": 365,
        "max_cost_perday": 1800,
        "learn_limit_perday": math.inf,
        "review_limit_perday": math.inf,
        "max_ivl": 36500,
    }
    (_, review_cnt_per_day, learn_cnt_per_day, memorized_cnt_per_day, _) = simulate(
        w=default_params["w"],
        max_cost_perday=math.inf,
        learn_limit_perday=10,
        review_limit_perday=50,
    )

    def moving_average(data, window_size=365 // 20):
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode="valid")

    fig1 = plt.figure()
    ax = fig1.gca()
    ax.plot(
        moving_average(review_cnt_per_day),
    )
    ax.set_title("Review Count per Day")
    ax.grid(True)
    fig2 = plt.figure()
    ax = fig2.gca()
    ax.plot(
        moving_average(learn_cnt_per_day),
    )
    ax.set_title("Learn Count per Day")
    ax.grid(True)
    fig3 = plt.figure()
    ax = fig3.gca()
    ax.plot(np.cumsum(learn_cnt_per_day))
    ax.set_title("Cumulative Learn Count")
    ax.grid(True)
    fig4 = plt.figure()
    ax = fig4.gca()
    ax.plot(memorized_cnt_per_day)
    ax.set_title("Memorized Count per Day")
    ax.grid(True)
    plt.show()
    workload_graph(default_params).savefig("workload.png")
