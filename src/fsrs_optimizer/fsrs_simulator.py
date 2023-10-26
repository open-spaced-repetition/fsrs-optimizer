import numpy as np
from tqdm import trange, tqdm

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
        hard_penalty = np.where(response == 1, w[15], 1)
        easy_bonus = np.where(response == 3, w[16], 1)
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
        card_table[col["retrievability"]][has_learned] = np.power(
            1
            + card_table[col["delta_t"]][has_learned]
            / (9 * card_table[col["stability"]][has_learned]),
            -1,
        )

        card_table[col["cost"]] = 0
        need_review = card_table[col["due"]] <= today
        card_table[col["rand"]][need_review] = np.random.rand(np.sum(need_review))
        forget = card_table[col["rand"]] > card_table[col["retrievability"]]
        card_table[col["cost"]][need_review & forget] = forget_cost
        card_table[col["rating"]][need_review & ~forget] = np.random.choice(
            [1, 2, 3], np.sum(need_review & ~forget), p=review_rating_prob
        )
        card_table[col["cost"]][need_review & ~forget] = np.choose(
            card_table[col["rating"]][need_review & ~forget].astype(int) - 1,
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
            card_table[col["difficulty"]][true_review & ~forget] - w[6] * (card_table[col["rating"]][true_review & ~forget] - 3) , 1, 10
        )

        need_learn = card_table[col["due"]] == learn_span
        card_table[col["cost"]][need_learn] = learn_cost
        true_learn = need_learn & (
            np.cumsum(card_table[col["cost"]]) <= max_cost_perday
        )
        card_table[col["last_date"]][true_learn] = today
        first_ratings = np.random.choice(4, np.sum(true_learn), p=first_rating_prob)
        card_table[col["stability"]][true_learn] = np.choose(first_ratings, w[:4])
        card_table[col["difficulty"]][true_learn] = w[4] - w[5] * (first_ratings - 3)

        card_table[col["ivl"]][true_review | true_learn] = np.clip(
            np.round(
                9
                * card_table[col["stability"]][true_review | true_learn]
                * (1 / request_retention - 1),
                0,
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


def optimal_retention(
    w,
    deck_size=10000,
    learn_span=365,
    max_cost_perday=1800,
    max_ivl=36500,
    recall_costs=np.array([14, 10, 6]),
    forget_cost=50,
    learn_cost=20,
    first_rating_prob=np.array([0.15, 0.2, 0.6, 0.05]),
    review_rating_prob=np.array([0.3, 0.6, 0.1]),
):
    low = 0.75
    high = 0.95
    optimal = 0.85
    epsilon = 0.01

    pbar = tqdm(desc="optimization", colour="red", total=10)
    for _ in range(10):
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        def sample(
            n,
            mid,
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
            for i in range(n):
                _, _, _, memorized_cnt_per_day = simulate(
                    w,
                    request_retention=mid,
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

        if sample(
            5,
            mid1,
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
        ) > sample(
            5,
            mid2,
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
            high = mid2
        else:
            low = mid1

        optimal = (low + high) / 2
        pbar.update(1)
        if high - low < epsilon:
            break

    return optimal
