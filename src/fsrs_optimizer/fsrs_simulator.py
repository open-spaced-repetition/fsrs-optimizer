import numpy as np
from tqdm import trange, tqdm


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

SAMPLE_SIZE = 5


def simulate(
    w,
    request_retention=0.9,
    deck_size=10000,
    learn_span=365,
    max_cost_perday=1800,
    max_ivl=36500,
    recall_costs=np.array([14, 10, 6]),
    forget_cost=50,
    learn_cost=20,
    first_rating_prob=np.array([0.15, 0.2, 0.6, 0.05]),
    review_rating_prob=np.array([0.3, 0.6, 0.1]),
    seed=42,
):
    np.random.seed(seed)
    card_table = np.zeros((len(columns), deck_size))
    card_table[col["due"]] = learn_span
    card_table[col["difficulty"]] = 1e-10
    card_table[col["stability"]] = 1e-10

    review_cnt_per_day = np.zeros(learn_span)
    learn_cnt_per_day = np.zeros(learn_span)
    memorized_cnt_per_day = np.zeros(learn_span)

    def stability_after_success(s, r, d, response):
        hard_penalty = np.where(response == 2, w[15], 1)
        easy_bonus = np.where(response == 4, w[16], 1)
        return s * (
            1
            + np.exp(w[8])
            * (11 - d)
            * np.power(s, -w[9])
            * (np.exp((1 - r) * w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )

    def stability_after_failure(s, r, d):
        return np.maximum(
            0.1,
            np.minimum(
                w[11]
                * np.power(d, -w[12])
                * (np.power(s + 1, w[13]) - 1)
                * np.exp((1 - r) * w[14]),
                s,
            ),
        )

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
        card_table[col["cost"]][need_review & forget] = forget_cost
        card_table[col["rating"]][need_review & ~forget] = np.random.choice(
            [2, 3, 4], np.sum(need_review & ~forget), p=review_rating_prob
        )
        card_table[col["cost"]][need_review & ~forget] = np.choose(
            card_table[col["rating"]][need_review & ~forget].astype(int) - 2,
            recall_costs,
        )
        true_review = need_review & (
            np.cumsum(card_table[col["cost"]]) <= max_cost_perday
        )
        card_table[col["last_date"]][true_review] = today

        card_table[col["lapses"]][true_review & forget] += 1
        card_table[col["reps"]][true_review & ~forget] += 1

        card_table[col["stability"]][true_review & forget] = stability_after_failure(
            card_table[col["stability"]][true_review & forget],
            card_table[col["retrievability"]][true_review & forget],
            card_table[col["difficulty"]][true_review & forget],
        )
        card_table[col["stability"]][true_review & ~forget] = stability_after_success(
            card_table[col["stability"]][true_review & ~forget],
            card_table[col["retrievability"]][true_review & ~forget],
            card_table[col["difficulty"]][true_review & ~forget],
            card_table[col["rating"]][true_review & ~forget],
        )

        card_table[col["difficulty"]][true_review & forget] = np.clip(
            card_table[col["difficulty"]][true_review & forget] + 2 * w[6], 1, 10
        )

        card_table[col["difficulty"]][true_review & ~forget] = np.clip(
            card_table[col["difficulty"]][true_review & ~forget]
            - w[6] * (card_table[col["rating"]][true_review & ~forget] - 3),
            1,
            10,
        )

        need_learn = card_table[col["due"]] == learn_span
        card_table[col["cost"]][need_learn] = learn_cost
        true_learn = need_learn & (
            np.cumsum(card_table[col["cost"]]) <= max_cost_perday
        )
        card_table[col["last_date"]][true_learn] = today
        first_ratings = np.random.choice(
            [1, 2, 3, 4], np.sum(true_learn), p=first_rating_prob
        )
        card_table[col["stability"]][true_learn] = np.choose(first_ratings - 1, w[:4])
        card_table[col["difficulty"]][true_learn] = np.clip(
            w[4] - w[5] * (first_ratings - 3), 1, 10
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
    return card_table, review_cnt_per_day, learn_cnt_per_day, memorized_cnt_per_day


def optimal_retention(**kwargs):
    return brent(**kwargs)


def sample(
    r,
    w,
    deck_size,
    learn_span,
    max_cost_perday,
    max_ivl,
    recall_costs,
    forget_cost,
    learn_cost,
    first_rating_prob,
    review_rating_prob,
):
    memorization = []
    for i in range(SAMPLE_SIZE):
        _, _, _, memorized_cnt_per_day = simulate(
            w,
            request_retention=r,
            deck_size=deck_size,
            learn_span=learn_span,
            max_cost_perday=max_cost_perday,
            max_ivl=max_ivl,
            recall_costs=recall_costs,
            forget_cost=forget_cost,
            learn_cost=learn_cost,
            first_rating_prob=first_rating_prob,
            review_rating_prob=review_rating_prob,
            seed=42 + i,
        )
        memorization.append(memorized_cnt_per_day[-1])
    return np.mean(memorization)


def native(**kwargs):
    low = 0.75
    high = 0.95
    optimal = 0.85
    epsilon = 0.01
    for _ in range(10):
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3
        if sample(r=mid1, **kwargs) > sample(r=mid2, **kwargs):
            high = mid2
        else:
            low = mid1
        optimal = (low + high) / 2
        if high - low < epsilon:
            break
    return optimal


def bracket(xa=0.75, xb=0.95, maxiter=20, **kwargs):
    u_lim = 0.95
    l_lim = 0.75

    grow_limit = 100.0
    gold = 1.6180339
    verysmall_num = 1e-21

    fa = -sample(xa, **kwargs)
    fb = -sample(xb, **kwargs)
    funccalls = 2

    if fa < fb:  # Switch so fa > fb
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = max(min(xb + gold * (xb - xa), u_lim), l_lim)
    fc = -sample(xc, **kwargs)
    funccalls += 1

    iter = 0
    while fc < fb:
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if np.abs(val) < verysmall_num:
            denom = 2.0 * verysmall_num
        else:
            denom = 2.0 * val
        w = max(min((xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom), u_lim), l_lim)
        wlim = max(min(xb + grow_limit * (xc - xb), u_lim), l_lim)

        if iter > maxiter:
            print("Failed to converge")
            break

        iter += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = -sample(w, **kwargs)
            funccalls += 1
            if fw < fc:
                xa = max(min(xb, u_lim), l_lim)
                xb = max(min(w, u_lim), l_lim)
                fa = fb
                fb = fw
                break
            elif fw > fb:
                xc = max(min(w, u_lim), l_lim)
                fc = fw
                break
            w = max(min(xc + gold * (xc - xb), u_lim), l_lim)
            fw = -sample(w, **kwargs)
            funccalls += 1
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = -sample(w, **kwargs)
            funccalls += 1
        elif (w - wlim) * (xc - w) > 0.0:
            fw = -sample(w, **kwargs)
            funccalls += 1
            if fw < fc:
                xb = max(min(xc, u_lim), l_lim)
                xc = max(min(w, u_lim), l_lim)
                w = max(min(xc + gold * (xc - xb), u_lim), l_lim)
                fb = fc
                fc = fw
                fw = -sample(w, **kwargs)
                funccalls += 1
        else:
            w = max(min(xc + gold * (xc - xb), u_lim), l_lim)
            fw = -sample(w, **kwargs)
            funccalls += 1
        xa = max(min(xb, u_lim), l_lim)
        xb = max(min(xc, u_lim), l_lim)
        xc = max(min(w, u_lim), l_lim)
        fa = fb
        fb = fc
        fc = fw

    return xa, xb, xc, fa, fb, fc, funccalls


def brent(tol=0.01, maxiter=20, **kwargs):
    mintol = 1.0e-11
    cg = 0.3819660

    xa, xb, xc, fa, fb, fc, funccalls = bracket(
        xa=0.75, xb=0.95, maxiter=maxiter, **kwargs
    )

    #################################
    # BEGIN
    #################################
    x = w = v = xb
    fw = fv = fx = fb
    if xa < xc:
        a = xa
        b = xc
    else:
        a = xc
        b = xa
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
        fu = -sample(u, **kwargs)  # calculate new output value
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


if __name__ == "__main__":
    default_params = {
        "w": [
            0.5888,
            1.4616,
            3.8226,
            14.1364,
            4.9214,
            1.0325,
            0.8731,
            0.0613,
            1.57,
            0.1395,
            0.988,
            2.212,
            0.0658,
            0.3439,
            1.3098,
            0.2837,
            2.7766,
        ],
        "deck_size": 10000,
        "learn_span": 365,
        "max_cost_perday": 1800,
        "max_ivl": 36500,
        "recall_costs": np.array([14, 10, 6]),
        "forget_cost": 50,
        "learn_cost": 20,
        "first_rating_prob": np.array([0.15, 0.2, 0.6, 0.05]),
        "review_rating_prob": np.array([0.3, 0.6, 0.1]),
    }
    import time

    start = time.time()
    memo1 = native(**default_params)
    print("Native: ", memo1)
    print("Time: ", time.time() - start)
    print("Memorization: ", sample(r=memo1, **default_params))

    start = time.time()
    memo2 = brent(**default_params)
    print("Brent: ", memo2)
    print("Time: ", time.time() - start)
    print("Memorization: ", sample(r=memo2, **default_params))
