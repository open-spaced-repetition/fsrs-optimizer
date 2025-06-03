import zipfile
import sqlite3
import time
import pandas as pd
import numpy as np
import os
import math
from typing import List, Optional, Tuple
from datetime import timedelta, datetime
from collections import defaultdict
import statsmodels.api as sm  # type: ignore
from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import (  # type: ignore
    log_loss,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    roc_auc_score,
)
from scipy.optimize import minimize  # type: ignore
from itertools import accumulate
from tqdm.auto import tqdm  # type: ignore
import warnings

try:
    from .fsrs_simulator import *
except ImportError:
    from fsrs_simulator import *  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)

New = 0
Learning = 1
Review = 2
Relearning = 3

DEFAULT_PARAMETER = [
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    0.1542,
]

DEFAULT_PARAMS_STDDEV_TENSOR = torch.tensor(
    [
        6.43,
        9.66,
        17.58,
        27.85,
        0.57,
        0.28,
        0.6,
        0.12,
        0.39,
        0.18,
        0.33,
        0.3,
        0.09,
        0.16,
        0.57,
        0.25,
        1.03,
        0.31,
        0.32,
        0.14,
        0.27,
    ],
    dtype=torch.float,
)


class FSRS(nn.Module):
    def __init__(self, w: List[float], float_delta_t: bool = False):
        super(FSRS, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.float_delta_t = float_delta_t

    def stability_after_success(
        self, state: Tensor, r: Tensor, rating: Tensor
    ) -> Tensor:
        hard_penalty = torch.where(rating == 2, self.w[15], 1)
        easy_bonus = torch.where(rating == 4, self.w[16], 1)
        new_s = state[:, 0] * (
            1
            + torch.exp(self.w[8])
            * (11 - state[:, 1])
            * torch.pow(state[:, 0], -self.w[9])
            * (torch.exp((1 - r) * self.w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )
        return new_s

    def stability_after_failure(self, state: Tensor, r: Tensor) -> Tensor:
        old_s = state[:, 0]
        new_s = (
            self.w[11]
            * torch.pow(state[:, 1], -self.w[12])
            * (torch.pow(old_s + 1, self.w[13]) - 1)
            * torch.exp((1 - r) * self.w[14])
        )
        new_minimum_s = old_s / torch.exp(self.w[17] * self.w[18])
        return torch.minimum(new_s, new_minimum_s)

    def stability_short_term(self, state: Tensor, rating: Tensor) -> Tensor:
        sinc = torch.exp(self.w[17] * (rating - 3 + self.w[18])) * torch.pow(
            state[:, 0], -self.w[19]
        )
        new_s = state[:, 0] * torch.where(rating >= 3, sinc.clamp(min=1), sinc)
        return new_s

    def init_d(self, rating: Tensor) -> Tensor:
        new_d = self.w[4] - torch.exp(self.w[5] * (rating - 1)) + 1
        return new_d

    def linear_damping(self, delta_d: Tensor, old_d: Tensor) -> Tensor:
        return delta_d * (10 - old_d) / 9

    def next_d(self, state: Tensor, rating: Tensor) -> Tensor:
        delta_d = -self.w[6] * (rating - 3)
        new_d = state[:, 1] + self.linear_damping(delta_d, state[:, 1])
        new_d = self.mean_reversion(self.init_d(4), new_d)
        return new_d

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4])
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0])
            new_s[index[0]] = self.w[index[1]]
            new_d = self.init_d(X[:, 1])
            new_d = new_d.clamp(1, 10)
        else:
            r = power_forgetting_curve(X[:, 0], state[:, 0], -self.w[20])
            short_term = X[:, 0] < 1
            success = X[:, 1] > 1
            new_s = (
                torch.where(
                    short_term,
                    self.stability_short_term(state, X[:, 1]),
                    torch.where(
                        success,
                        self.stability_after_success(state, r, X[:, 1]),
                        self.stability_after_failure(state, r),
                    ),
                )
                if not self.float_delta_t
                else torch.where(
                    success,
                    self.stability_after_success(state, r, X[:, 1]),
                    self.stability_after_failure(state, r),
                )
            )
            new_d = self.next_d(state, X[:, 1])
            new_d = new_d.clamp(1, 10)
        new_s = new_s.clamp(S_MIN, 36500)
        return torch.stack([new_s, new_d], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 2))
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return self.w[7] * init + (1 - self.w[7]) * current


class ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(S_MIN, 100)
            w[1] = w[1].clamp(S_MIN, 100)
            w[2] = w[2].clamp(S_MIN, 100)
            w[3] = w[3].clamp(S_MIN, 100)
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.001, 4)
            w[6] = w[6].clamp(0.001, 4)
            w[7] = w[7].clamp(0.001, 0.75)
            w[8] = w[8].clamp(0, 4.5)
            w[9] = w[9].clamp(0, 0.8)
            w[10] = w[10].clamp(0.001, 3.5)
            w[11] = w[11].clamp(0.001, 5)
            w[12] = w[12].clamp(0.001, 0.25)
            w[13] = w[13].clamp(0.001, 0.9)
            w[14] = w[14].clamp(0, 4)
            w[15] = w[15].clamp(0, 1)
            w[16] = w[16].clamp(1, 6)
            w[17] = w[17].clamp(0, 2)
            w[18] = w[18].clamp(0, 2)
            w[19] = w[19].clamp(0, 0.8)
            w[20] = w[20].clamp(0.1, 0.8)
            module.w.data = w


def lineToTensor(line: str) -> Tensor:
    ivl = line[0].split(",")
    response = line[1].split(",")
    tensor = torch.zeros(len(response), 2)
    for li, response in enumerate(response):
        tensor[li][0] = float(ivl[li])
        tensor[li][1] = int(response)
    return tensor


class BatchDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        batch_size: int = 0,
        sort_by_length: bool = True,
        max_seq_len: int = math.inf,
        device: str = "cpu",
    ):
        if dataframe.empty:
            raise ValueError("Training data is inadequate.")
        dataframe["seq_len"] = dataframe["tensor"].map(len)
        if dataframe["seq_len"].min() > max_seq_len:
            raise ValueError("Training data is inadequate.")
        dataframe = dataframe[dataframe["seq_len"] <= max_seq_len]
        if sort_by_length:
            dataframe = dataframe.sort_values(by=["seq_len"], kind="stable")
        del dataframe["seq_len"]
        self.x_train = pad_sequence(
            dataframe["tensor"].to_list(), batch_first=True, padding_value=0
        )
        self.t_train = torch.tensor(dataframe["delta_t"].values, dtype=torch.float)
        self.y_train = torch.tensor(dataframe["y"].values, dtype=torch.float)
        self.seq_len = torch.tensor(
            dataframe["tensor"].map(len).values, dtype=torch.long
        )
        if "weights" in dataframe.columns:
            self.weights = torch.tensor(dataframe["weights"].values, dtype=torch.float)
        else:
            self.weights = torch.ones(len(dataframe), dtype=torch.float)
        length = len(dataframe)
        batch_num, remainder = divmod(length, max(1, batch_size))
        self.batch_num = batch_num + 1 if remainder > 0 else batch_num
        self.batches = [None] * self.batch_num
        if batch_size > 0:
            for i in range(self.batch_num):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, length)
                sequences = self.x_train[start_index:end_index]
                seq_lens = self.seq_len[start_index:end_index]
                max_seq_len = max(seq_lens)
                sequences_truncated = sequences[:, :max_seq_len]
                self.batches[i] = (
                    sequences_truncated.transpose(0, 1).to(device),
                    self.t_train[start_index:end_index].to(device),
                    self.y_train[start_index:end_index].to(device),
                    seq_lens.to(device),
                    self.weights[start_index:end_index].to(device),
                )

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return self.batch_num


class BatchLoader:
    def __init__(self, dataset: BatchDataset, shuffle: bool = True, seed: int = 2023):
        self.dataset = dataset
        self.batch_nums = len(dataset.batches)
        self.shuffle = shuffle
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            yield from (
                self.dataset[idx]
                for idx in torch.randperm(
                    self.batch_nums, generator=self.generator
                ).tolist()
            )
        else:
            yield from (self.dataset[idx] for idx in range(self.batch_nums))

    def __len__(self):
        return self.batch_nums


class Trainer:
    def __init__(
        self,
        train_set: pd.DataFrame,
        test_set: Optional[pd.DataFrame],
        init_w: List[float],
        n_epoch: int = 5,
        lr: float = 4e-2,
        gamma: float = 1,
        batch_size: int = 512,
        max_seq_len: int = 64,
        float_delta_t: bool = False,
        enable_short_term: bool = True,
    ) -> None:
        if not enable_short_term:
            init_w[17] = 0
            init_w[18] = 0
        self.model = FSRS(init_w, float_delta_t)
        self.init_w_tensor = torch.tensor(init_w, dtype=torch.float)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clipper = ParameterClipper()
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.build_dataset(train_set, test_set)
        self.n_epoch = n_epoch
        self.batch_nums = self.train_data_loader.batch_nums
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.batch_nums * n_epoch
        )
        self.avg_train_losses = []
        self.avg_eval_losses = []
        self.loss_fn = nn.BCELoss(reduction="none")
        self.float_delta_t = float_delta_t
        self.enable_short_term = enable_short_term

    def build_dataset(self, train_set: pd.DataFrame, test_set: Optional[pd.DataFrame]):
        self.train_set = BatchDataset(
            train_set, batch_size=self.batch_size, max_seq_len=self.max_seq_len
        )
        self.train_data_loader = BatchLoader(self.train_set)

        self.test_set = (
            []
            if test_set is None
            else BatchDataset(
                test_set, batch_size=self.batch_size, max_seq_len=self.max_seq_len
            )
        )

    def train(self, verbose: bool = True):
        self.verbose = verbose
        best_loss = np.inf
        epoch_len = len(self.train_set.y_train)
        if verbose:
            pbar = tqdm(desc="train", colour="red", total=epoch_len * self.n_epoch)
        print_len = max(self.batch_nums * self.n_epoch // 10, 1)
        for k in range(self.n_epoch):
            weighted_loss, w = self.eval()
            if weighted_loss < best_loss:
                best_loss = weighted_loss
                best_w = w

            for i, batch in enumerate(self.train_data_loader):
                self.model.train()
                self.optimizer.zero_grad()
                sequences, delta_ts, labels, seq_lens, weights = batch
                real_batch_size = seq_lens.shape[0]
                outputs, _ = self.model(sequences)
                stabilities = outputs[seq_lens - 1, torch.arange(real_batch_size), 0]
                retentions = power_forgetting_curve(
                    delta_ts, stabilities, -self.model.w[20]
                )
                loss = (self.loss_fn(retentions, labels) * weights).sum()
                penalty = (
                    torch.sum(
                        torch.square(self.model.w - self.init_w_tensor)
                        / torch.square(DEFAULT_PARAMS_STDDEV_TENSOR)
                    )
                    * self.gamma
                    * real_batch_size
                    / epoch_len
                )
                loss += penalty
                loss.backward()
                if self.float_delta_t:
                    for param in self.model.parameters():
                        param.grad[:4] = torch.zeros(4)
                if not self.enable_short_term:
                    for param in self.model.parameters():
                        param.grad[17:19] = torch.zeros(2)
                self.optimizer.step()
                self.scheduler.step()
                self.model.apply(self.clipper)
                if verbose:
                    pbar.update(real_batch_size)
                if verbose and (k * self.batch_nums + i + 1) % print_len == 0:
                    tqdm.write(
                        f"iteration: {k * epoch_len + (i + 1) * self.batch_size}"
                    )
                    for name, param in self.model.named_parameters():
                        tqdm.write(
                            f"{name}: {list(map(lambda x: round(float(x), 4),param))}"
                        )
        if verbose:
            pbar.close()

        weighted_loss, w = self.eval()
        if weighted_loss < best_loss:
            best_loss = weighted_loss
            best_w = w

        return best_w

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            losses = []
            for dataset in (self.train_set, self.test_set):
                if len(dataset) == 0:
                    losses.append(0)
                    continue
                sequences, delta_ts, labels, seq_lens, weights = (
                    dataset.x_train,
                    dataset.t_train,
                    dataset.y_train,
                    dataset.seq_len,
                    dataset.weights,
                )
                real_batch_size = seq_lens.shape[0]
                outputs, _ = self.model(sequences.transpose(0, 1))
                stabilities = outputs[seq_lens - 1, torch.arange(real_batch_size), 0]
                retentions = power_forgetting_curve(
                    delta_ts, stabilities, -self.model.w[20]
                )
                loss = (self.loss_fn(retentions, labels) * weights).mean()
                penalty = torch.sum(
                    torch.square(self.model.w - self.init_w_tensor)
                    / torch.square(DEFAULT_PARAMS_STDDEV_TENSOR)
                )
                loss += penalty * self.gamma / len(self.train_set.y_train)
                losses.append(loss)
            self.avg_train_losses.append(losses[0])
            self.avg_eval_losses.append(losses[1])

            w = list(
                map(
                    lambda x: round(float(x), 4),
                    dict(self.model.named_parameters())["w"].data,
                )
            )

            weighted_loss = (
                losses[0] * len(self.train_set) + losses[1] * len(self.test_set)
            ) / (len(self.train_set) + len(self.test_set))

            return weighted_loss, w

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.avg_train_losses, label="train")
        ax.plot(self.avg_eval_losses, label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        return fig


class Collection:
    def __init__(self, w: List[float], float_delta_t: bool = False) -> None:
        self.model = FSRS(w, float_delta_t)
        self.model.eval()

    def predict(self, t_history: str, r_history: str):
        with torch.no_grad():
            line_tensor = lineToTensor(
                list(zip([t_history], [r_history]))[0]
            ).unsqueeze(1)
            output_t = self.model(line_tensor)
            return output_t[-1][0]

    def batch_predict(self, dataset):
        fast_dataset = BatchDataset(dataset, sort_by_length=False)
        with torch.no_grad():
            outputs, _ = self.model(fast_dataset.x_train.transpose(0, 1))
            stabilities, difficulties = outputs[
                fast_dataset.seq_len - 1, torch.arange(len(fast_dataset))
            ].transpose(0, 1)
            return stabilities.tolist(), difficulties.tolist()


def remove_outliers(group: pd.DataFrame) -> pd.DataFrame:
    # threshold = np.mean(group['delta_t']) * 1.5
    # threshold = group['delta_t'].quantile(0.95)
    # Q1 = group['delta_t'].quantile(0.25)
    # Q3 = group['delta_t'].quantile(0.75)
    # IQR = Q3 - Q1
    # threshold = Q3 + 1.5 * IQR
    # group = group[group['delta_t'] <= threshold]
    grouped_group = (
        group.groupby(by=["first_rating", "delta_t"], group_keys=False)
        .agg({"y": ["mean", "count"]})
        .reset_index()
    )
    sort_index = grouped_group.sort_values(
        by=[("y", "count"), "delta_t"], ascending=[True, False]
    ).index

    total = sum(grouped_group[("y", "count")])
    has_been_removed = 0
    for i in sort_index:
        count = grouped_group.loc[i, ("y", "count")]
        delta_t = grouped_group.loc[i, "delta_t"].values[0]
        if has_been_removed + count >= max(total * 0.05, 20):
            if count < 6 or delta_t > (100 if group.name[0] != "4" else 365):
                group.drop(group[group["delta_t"] == delta_t].index, inplace=True)
                has_been_removed += count
        else:
            group.drop(group[group["delta_t"] == delta_t].index, inplace=True)
            has_been_removed += count
    return group


def remove_non_continuous_rows(group):
    discontinuity = group["i"].diff().fillna(1).ne(1)
    if not discontinuity.any():
        return group
    else:
        first_non_continuous_index = discontinuity.idxmax()
        return group.loc[: first_non_continuous_index - 1]


def fit_stability(delta_t, retention, size):
    def loss(stability):
        y_pred = power_forgetting_curve(delta_t, stability)
        loss = sum(
            -(retention * np.log(y_pred) + (1 - retention) * np.log(1 - y_pred)) * size
        )
        return loss

    res = minimize(loss, x0=1, bounds=[(S_MIN, 36500)])
    return res.x[0]


class Optimizer:
    float_delta_t: bool = False
    enable_short_term: bool = True

    def __init__(
        self, float_delta_t: bool = False, enable_short_term: bool = True
    ) -> None:
        tqdm.pandas()
        self.float_delta_t = float_delta_t
        self.enable_short_term = enable_short_term
        global S_MIN
        S_MIN = 1e-6 if float_delta_t else 0.001

    def anki_extract(
        self,
        filename: str,
        filter_out_suspended_cards: bool = False,
        filter_out_flags: List[int] = [],
    ):
        """Step 1"""
        # Extract the collection file or deck file to get the .anki21 database.
        with zipfile.ZipFile(f"{filename}", "r") as zip_ref:
            zip_ref.extractall("./")
            tqdm.write("Deck file extracted successfully!")

        if os.path.isfile("collection.anki21b"):
            os.remove("collection.anki21b")
            raise Exception(
                "Please export the file with `support older Anki versions` if you use the latest version of Anki."
            )
        elif os.path.isfile("collection.anki21"):
            con = sqlite3.connect("collection.anki21")
        elif os.path.isfile("collection.anki2"):
            con = sqlite3.connect("collection.anki2")
        else:
            raise Exception("Collection not exist!")
        cur = con.cursor()

        def flags2str(flags: List[int]) -> str:
            return f"({','.join(map(str, flags))})"

        res = cur.execute(
            f"""
        SELECT *
        FROM revlog
        WHERE cid IN (
            SELECT id
            FROM cards
            WHERE queue != 0
            AND id <= {time.time() * 1000}
            {"AND queue != -1" if filter_out_suspended_cards else ""}
            {"AND flags NOT IN %s" % flags2str(filter_out_flags) if len(filter_out_flags) > 0 else ""}
        )
        AND ease BETWEEN 1 AND 4
        AND (
            type != 3
            OR factor != 0
        )
        AND id <= {time.time() * 1000}
        ORDER BY cid, id
        """
        )
        revlog = res.fetchall()
        if len(revlog) == 0:
            raise Exception("No review log found!")
        df = pd.DataFrame(revlog)
        df.columns = [
            "review_time",
            "card_id",
            "usn",
            "review_rating",
            "ivl",
            "last_ivl",
            "factor",
            "review_duration",
            "review_state",
        ]
        df["i"] = df.groupby("card_id").cumcount() + 1
        df["is_learn_start"] = (df["review_state"] == 0) & (
            (df["review_state"].shift() != 0) | (df["i"] == 1)
        )
        df["sequence_group"] = df["is_learn_start"].cumsum()
        last_learn_start = (
            df[df["is_learn_start"]].groupby("card_id")["sequence_group"].last()
        )
        df["last_learn_start"] = (
            df["card_id"].map(last_learn_start).fillna(0).astype(int)
        )
        df["mask"] = df["last_learn_start"] <= df["sequence_group"]
        df = df[df["mask"] == True].copy()
        df["review_state"] = df["review_state"] + 1
        df.loc[df["is_learn_start"], "review_state"] = New
        df = df.groupby("card_id").filter(
            lambda group: group["review_state"].iloc[0] == New
        )
        df.drop(
            columns=[
                "i",
                "is_learn_start",
                "sequence_group",
                "last_learn_start",
                "mask",
                "usn",
                "ivl",
                "last_ivl",
                "factor",
            ],
            inplace=True,
        )
        df.to_csv("revlog.csv", index=False)
        tqdm.write("revlog.csv saved.")

    def extract_simulation_config(self, df):
        df_tmp = df[
            (df["review_duration"] > 0) & (df["review_duration"] < 1200000)
        ].copy()

        state_rating_costs = (
            df_tmp[df_tmp["review_state"] != 4]
            .groupby(["review_state", "review_rating"])["review_duration"]
            .median()
            .unstack(fill_value=0)
        ) / 1000
        state_rating_counts = (
            df_tmp[df_tmp["review_state"] != 4]
            .groupby(["review_state", "review_rating"])["review_duration"]
            .count()
            .unstack(fill_value=0)
        )

        # Ensure all ratings (1-4) exist in columns
        for rating in range(1, 5):
            if rating not in state_rating_costs.columns:
                state_rating_costs[rating] = 0
            if rating not in state_rating_counts.columns:
                state_rating_counts[rating] = 0

        # Ensure all states exist in index
        for state in [Learning, Review, Relearning]:
            if state not in state_rating_costs.index:
                state_rating_costs.loc[state] = 0
            if state not in state_rating_counts.index:
                state_rating_counts.loc[state] = 0

        self.state_rating_costs = state_rating_costs.values.round(2).tolist()
        for i, (rating_costs, default_rating_cost, rating_counts) in enumerate(
            zip(
                state_rating_costs.values.tolist(),
                DEFAULT_STATE_RATING_COSTS,
                state_rating_counts.values.tolist(),
            )
        ):
            for j, (cost, default_cost, count) in enumerate(
                zip(rating_costs, default_rating_cost, rating_counts)
            ):
                weight = count / (50 + count)
                self.state_rating_costs[i][j] = cost * weight + default_cost * (
                    1 - weight
                )

        df1 = (
            df_tmp.groupby(by=["card_id", "real_days"])
            .agg(
                {
                    "review_state": "first",
                    "review_rating": ["first", list],
                    "review_duration": "sum",
                }
            )
            .reset_index()
        )
        del df1["real_days"]
        df1.columns = [
            "card_id",
            "first_state",
            "first_rating",
            "same_day_ratings",
            "sum_review_duration",
        ]
        model = FirstOrderMarkovChain()
        learning_step_rating_sequences = df1[df1["first_state"] == Learning][
            "same_day_ratings"
        ]
        result = model.fit(learning_step_rating_sequences)
        learning_transition_matrix, learning_transition_counts = (
            result.transition_matrix[:3],
            result.transition_counts[:3],
        )
        self.learning_step_transitions = learning_transition_matrix.round(2).tolist()
        for i, (rating_probs, default_rating_probs, transition_counts) in enumerate(
            zip(
                learning_transition_matrix.tolist(),
                DEFAULT_LEARNING_STEP_TRANSITIONS,
                learning_transition_counts.tolist(),
            )
        ):
            weight = sum(transition_counts) / (50 + sum(transition_counts))
            for j, (prob, default_prob) in enumerate(
                zip(rating_probs, default_rating_probs)
            ):
                self.learning_step_transitions[i][j] = prob * weight + default_prob * (
                    1 - weight
                )

        relearning_step_rating_sequences = df1[
            (df1["first_state"] == Review) & (df1["first_rating"] == 1)
        ]["same_day_ratings"]
        result = model.fit(relearning_step_rating_sequences)
        relearning_transition_matrix, relearning_transition_counts = (
            result.transition_matrix[:3],
            result.transition_counts[:3],
        )
        self.relearning_step_transitions = relearning_transition_matrix.round(
            2
        ).tolist()
        for i, (rating_probs, default_rating_probs, transition_counts) in enumerate(
            zip(
                relearning_transition_matrix.tolist(),
                DEFAULT_RELEARNING_STEP_TRANSITIONS,
                relearning_transition_counts.tolist(),
            )
        ):
            weight = sum(transition_counts) / (50 + sum(transition_counts))
            for j, (prob, default_prob) in enumerate(
                zip(rating_probs, default_rating_probs)
            ):
                self.relearning_step_transitions[i][j] = (
                    prob * weight + default_prob * (1 - weight)
                )

        button_usage_dict = defaultdict(
            int,
            (
                df1.groupby(by=["first_state", "first_rating"])["card_id"]
                .count()
                .to_dict()
            ),
        )
        self.learn_buttons = (
            np.array([button_usage_dict[(1, i)] for i in range(1, 5)]) + 1
        )
        self.review_buttons = (
            np.array([button_usage_dict[(2, i)] for i in range(1, 5)]) + 1
        )
        self.first_rating_prob = self.learn_buttons / self.learn_buttons.sum()
        self.review_rating_prob = (
            self.review_buttons[1:] / self.review_buttons[1:].sum()
        )

        weight = sum(self.learn_buttons) / (50 + sum(self.learn_buttons))
        self.first_rating_prob = (
            self.first_rating_prob * weight + DEFAULT_FIRST_RATING_PROB * (1 - weight)
        )

        weight = sum(self.review_buttons[1:]) / (50 + sum(self.review_buttons[1:]))
        self.review_rating_prob = (
            self.review_rating_prob * weight + DEFAULT_REVIEW_RATING_PROB * (1 - weight)
        )

    def create_time_series(
        self,
        timezone: str,
        revlog_start_date: str,
        next_day_starts_at: int,
        analysis: bool = True,
    ):
        """Step 2"""
        df = pd.read_csv("./revlog.csv")
        df.sort_values(by=["card_id", "review_time"], inplace=True, ignore_index=True)
        df["review_date"] = pd.to_datetime(df["review_time"] // 1000, unit="s")
        df["review_date"] = (
            df["review_date"].dt.tz_localize("UTC").dt.tz_convert(timezone)
        )
        df.drop(df[df["review_date"].dt.year < 2006].index, inplace=True)
        df["real_days"] = df["review_date"] - timedelta(hours=int(next_day_starts_at))
        df["real_days"] = pd.DatetimeIndex(
            df["real_days"].dt.floor(
                "D", ambiguous="infer", nonexistent="shift_forward"
            )
        ).to_julian_date()
        # df.drop_duplicates(["card_id", "real_days"], keep="first", inplace=True)
        if self.float_delta_t:
            df["delta_t"] = df["review_time"].diff().fillna(0) / 1000 / 86400
        else:
            df["delta_t"] = df.real_days.diff()
        df.fillna({"delta_t": 0}, inplace=True)
        df["i"] = df.groupby("card_id").cumcount() + 1
        df.loc[df["i"] == 1, "delta_t"] = -1
        if df.empty:
            raise ValueError("Training data is inadequate.")

        if (
            "review_state" in df.columns
            and "review_duration" in df.columns
            and not (df["review_duration"] == 0).all()
        ):
            df["review_state"] = df["review_state"].map(
                lambda x: x if x != New else Learning
            )
            self.extract_simulation_config(df)
            df.drop(columns=["review_duration", "review_state"], inplace=True)

        def cum_concat(x):
            return list(accumulate(x))

        t_history_list = df.groupby("card_id", group_keys=False)["delta_t"].apply(
            lambda x: cum_concat(
                [[max(0, round(i, 6) if self.float_delta_t else int(i))] for i in x]
            )
        )
        df["t_history"] = [
            ",".join(map(str, item[:-1]))
            for sublist in t_history_list
            for item in sublist
        ]
        r_history_list = df.groupby("card_id", group_keys=False)["review_rating"].apply(
            lambda x: cum_concat([[i] for i in x])
        )
        df["r_history"] = [
            ",".join(map(str, item[:-1]))
            for sublist in r_history_list
            for item in sublist
        ]
        last_rating = []
        for t_sublist, r_sublist in zip(t_history_list, r_history_list):
            for t_history, r_history in zip(t_sublist, r_sublist):
                flag = True
                for t, r in zip(reversed(t_history[:-1]), reversed(r_history[:-1])):
                    if t > 0:
                        last_rating.append(r)
                        flag = False
                        break
                if flag:
                    last_rating.append(r_history[0])
        df["last_rating"] = last_rating

        df = df.groupby("card_id").filter(
            lambda group: group["review_time"].min()
            > time.mktime(datetime.strptime(revlog_start_date, "%Y-%m-%d").timetuple())
            * 1000
        )
        df = df[
            (df["review_rating"] != 0)
            & (df["r_history"].str.contains("0") == 0)
            & (df["delta_t"] != 0)
        ].copy()
        df["i"] = df.groupby("card_id").cumcount() + 1
        df["first_rating"] = df["r_history"].map(lambda x: x[0] if len(x) > 0 else "")
        df["y"] = df["review_rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])

        if not self.float_delta_t:
            df[df["i"] == 2] = (
                df[df["i"] == 2]
                .groupby(by=["first_rating"], as_index=False, group_keys=False)
                .apply(remove_outliers)
            )
            df.dropna(inplace=True)

            df = df.groupby("card_id", as_index=False, group_keys=False).progress_apply(
                remove_non_continuous_rows
            )

        df["review_time"] = df["review_time"].astype(int)
        df["review_rating"] = df["review_rating"].astype(int)
        df["delta_t"] = df["delta_t"].astype(float if self.float_delta_t else int)
        df["i"] = df["i"].astype(int)
        df["t_history"] = df["t_history"].astype(str)
        df["r_history"] = df["r_history"].astype(str)
        df["last_rating"] = df["last_rating"].astype(int)
        df["y"] = df["y"].astype(int)

        df.to_csv("revlog_history.tsv", sep="\t", index=False)
        tqdm.write("Trainset saved.")

        self.S0_dataset_group = (
            df[df["i"] == 2]
            .groupby(by=["first_rating", "delta_t"], group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )
        self.S0_dataset_group.to_csv("stability_for_pretrain.tsv", sep="\t", index=None)
        del df["first_rating"]

        if not analysis:
            return

        df["r_history"] = df.apply(
            lambda row: wrap_short_term_ratings(row["r_history"], row["t_history"]),
            axis=1,
        )

        df["retention"] = df.groupby(by=["r_history", "delta_t"], group_keys=False)[
            "y"
        ].transform("mean")
        df["total_cnt"] = df.groupby(by=["r_history", "delta_t"], group_keys=False)[
            "review_time"
        ].transform("count")
        tqdm.write("Retention calculated.")

        df.drop(
            columns=[
                "review_time",
                "card_id",
                "review_date",
                "real_days",
                "review_rating",
                "t_history",
                "last_rating",
                "y",
            ],
            inplace=True,
        )
        df.drop_duplicates(inplace=True)
        df["retention"] = df["retention"].map(lambda x: max(min(0.99, x), 0.01))

        def cal_stability(group: pd.DataFrame) -> pd.DataFrame:
            group_cnt = sum(group.groupby("delta_t").first()["total_cnt"])
            if group_cnt < 10:
                return pd.DataFrame()
            group["group_cnt"] = group_cnt
            if group["i"].values[0] > 1:
                group["stability"] = round(
                    fit_stability(
                        group["delta_t"], group["retention"], group["total_cnt"]
                    ),
                    1,
                )
            else:
                group["stability"] = 0.0
            group["avg_retention"] = round(
                sum(group["retention"] * pow(group["total_cnt"], 2))
                / sum(pow(group["total_cnt"], 2)),
                3,
            )
            group["avg_interval"] = round(
                sum(group["delta_t"] * pow(group["total_cnt"], 2))
                / sum(pow(group["total_cnt"], 2)),
                1,
            )
            del group["total_cnt"]
            del group["retention"]
            del group["delta_t"]
            return group

        df = df.groupby(by=["r_history"], group_keys=False).progress_apply(
            cal_stability
        )
        if df.empty:
            return "No enough data for stability calculation."
        tqdm.write("Stability calculated.")
        df.reset_index(drop=True, inplace=True)
        df.drop_duplicates(inplace=True)
        df.sort_values(by=["r_history"], inplace=True, ignore_index=True)

        if df.shape[0] > 0:
            for idx in tqdm(df.index, desc="analysis"):
                item = df.loc[idx]
                index = df[
                    (df["i"] == item["i"] + 1)
                    & (df["r_history"].str.startswith(item["r_history"]))
                ].index
                df.loc[index, "last_stability"] = item["stability"]
            df["factor"] = round(df["stability"] / df["last_stability"], 2)
            df = df[(df["i"] >= 2) & (df["group_cnt"] >= 100)].copy()
            df["last_recall"] = df["r_history"].map(lambda x: x[-1])
            df = df[
                df.groupby(["r_history"], group_keys=False)["group_cnt"].transform(
                    "max"
                )
                == df["group_cnt"]
            ]
            df.to_csv("./stability_for_analysis.tsv", sep="\t", index=None)
            tqdm.write("Analysis saved!")
            caption = "1:again, 2:hard, 3:good, 4:easy\n"
            df["first_rating"] = df["r_history"].map(lambda x: x[1])
            analysis = (
                df[df["r_history"].str.contains(r"^\([1-4][^124]*$", regex=True)][
                    [
                        "first_rating",
                        "i",
                        "r_history",
                        "avg_interval",
                        "avg_retention",
                        "stability",
                        "factor",
                        "group_cnt",
                    ]
                ]
                .sort_values(by=["first_rating", "i"])
                .to_string(index=False)
            )
            return caption + analysis

    def define_model(self):
        """Step 3"""
        self.init_w = DEFAULT_PARAMETER.copy()
        """
        For details about the parameters, please see: 
        https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm
        """

    def pretrain(self, dataset=None, verbose=True):
        if dataset is None:
            self.dataset = pd.read_csv(
                "./revlog_history.tsv",
                sep="\t",
                index_col=None,
                dtype={"r_history": str, "t_history": str},
            )
        else:
            self.dataset = dataset
            self.dataset["r_history"] = self.dataset["r_history"].fillna("")
            self.dataset["first_rating"] = self.dataset["r_history"].map(
                lambda x: x[0] if len(x) > 0 else ""
            )
            self.S0_dataset_group = (
                self.dataset[self.dataset["i"] == 2]
                .groupby(by=["first_rating", "delta_t"], group_keys=False)
                .agg({"y": ["mean", "count"]})
                .reset_index()
            )
        self.dataset = self.dataset[
            (self.dataset["i"] > 1) & (self.dataset["delta_t"] > 0)
        ]
        if self.dataset.empty:
            raise ValueError("Training data is inadequate.")
        rating_stability = {}
        rating_count = {}
        average_recall = self.dataset["y"].mean()
        plots = []
        r_s0_default = {str(i): DEFAULT_PARAMETER[i - 1] for i in range(1, 5)}

        for first_rating in ("1", "2", "3", "4"):
            group = self.S0_dataset_group[
                self.S0_dataset_group["first_rating"] == first_rating
            ]
            if group.empty:
                if verbose:
                    tqdm.write(
                        f"Not enough data for first rating {first_rating}. Expected at least 1, got 0."
                    )
                continue
            delta_t = group["delta_t"]
            recall = (
                (group["y"]["mean"] * group["y"]["count"] + average_recall * 1)
                / (group["y"]["count"] + 1)
                if not self.float_delta_t
                else group["y"]["mean"]
            )
            count = group["y"]["count"]

            init_s0 = r_s0_default[first_rating]

            def loss(stability):
                y_pred = power_forgetting_curve(delta_t, stability)
                logloss = sum(
                    -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                    * count
                )
                l1 = np.abs(stability - init_s0) / 16 if not self.float_delta_t else 0
                return logloss + l1

            res = minimize(
                loss,
                x0=init_s0,
                bounds=((S_MIN, 100),),
                options={"maxiter": int(sum(count))},
            )
            params = res.x
            stability = params[0]
            rating_stability[int(first_rating)] = stability
            rating_count[int(first_rating)] = sum(count)
            predict_recall = power_forgetting_curve(delta_t, *params)
            rmse = root_mean_squared_error(recall, predict_recall, sample_weight=count)

            if verbose:
                fig = plt.figure()
                ax = fig.gca()
                ax.plot(delta_t, recall, label="Exact")
                ax.plot(
                    np.linspace(0, 30),
                    power_forgetting_curve(np.linspace(0, 30), *params),
                    label=f"Weighted fit (RMSE: {rmse:.4f})",
                )
                count_percent = np.array([x / sum(count) for x in count])
                ax.scatter(delta_t, recall, s=count_percent * 1000, alpha=0.5)
                ax.legend(loc="upper right", fancybox=True, shadow=False)
                ax.grid(True)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Interval")
                ax.set_ylabel("Recall")
                ax.set_title(
                    f"Forgetting curve for first rating {first_rating} (n={sum(count)}, s={stability:.2f})"
                )
                plots.append(fig)
                tqdm.write(str(rating_stability))

        for small_rating, big_rating in (
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 3),
            (2, 4),
            (1, 4),
        ):
            if small_rating in rating_stability and big_rating in rating_stability:
                # if rating_count[small_rating] > 300 and rating_count[big_rating] > 300:
                #     continue
                if rating_stability[small_rating] > rating_stability[big_rating]:
                    if rating_count[small_rating] > rating_count[big_rating]:
                        rating_stability[big_rating] = rating_stability[small_rating]
                    else:
                        rating_stability[small_rating] = rating_stability[big_rating]

        w1 = 0.41
        w2 = 0.54

        if len(rating_stability) == 0:
            raise Exception("Not enough data for pretraining!")
        elif len(rating_stability) == 1:
            rating = list(rating_stability.keys())[0]
            factor = rating_stability[rating] / r_s0_default[str(rating)]
            init_s0 = list(map(lambda x: x * factor, r_s0_default.values()))
        elif len(rating_stability) == 2:
            if 1 not in rating_stability and 2 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[3], 1 / (1 - w2)
                ) * np.power(rating_stability[4], 1 - 1 / (1 - w2))
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability and 3 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[1], w1 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], 1 - w1 / (w1 + w2 - w1 * w2))
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - w2 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], w2 / (w1 + w2 - w1 * w2))
            elif 2 not in rating_stability and 4 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            elif 3 not in rating_stability and 4 not in rating_stability:
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - 1 / (1 - w1)
                ) * np.power(rating_stability[2], 1 / (1 - w1))
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 3:
            if 1 not in rating_stability:
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
            elif 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
            elif 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 4:
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]

        self.init_w[0:4] = list(map(lambda x: max(min(100, x), S_MIN), init_s0))
        if verbose:
            tqdm.write(f"Pretrain finished!")
        return plots

    def train(
        self,
        lr: float = 4e-2,
        n_epoch: int = 5,
        gamma: float = 1.0,
        batch_size: int = 512,
        verbose: bool = True,
        split_by_time: bool = False,
        recency_weight: bool = False,
    ):
        """Step 4"""
        self.dataset["tensor"] = self.dataset.progress_apply(
            lambda x: lineToTensor(list(zip([x["t_history"]], [x["r_history"]]))[0]),
            axis=1,
        )
        self.dataset["group"] = self.dataset["r_history"] + self.dataset["t_history"]
        if verbose:
            tqdm.write("Tensorized!")

        w = []
        plots = []
        self.dataset.sort_values(by=["review_time"], inplace=True)
        if split_by_time:
            tscv = TimeSeriesSplit(n_splits=5)
            for i, (train_index, test_index) in enumerate(tscv.split(self.dataset)):
                if verbose:
                    tqdm.write(f"TRAIN: {len(train_index)} TEST: {len(test_index)}")
                train_set = self.dataset.iloc[train_index].copy()
                test_set = self.dataset.iloc[test_index].copy()
                trainer = Trainer(
                    train_set,
                    test_set,
                    self.init_w,
                    n_epoch=n_epoch,
                    lr=lr,
                    gamma=gamma,
                    batch_size=batch_size,
                    float_delta_t=self.float_delta_t,
                    enable_short_term=self.enable_short_term,
                )
                w.append(trainer.train(verbose=verbose))
                self.w = w[-1]
                self.evaluate()
                metrics, figures = self.calibration_graph(self.dataset.iloc[test_index])
                for j, f in enumerate(figures):
                    f.savefig(f"graph_{j}_test_{i}.png")
                    plt.close(f)
                if verbose:
                    print(metrics)
                    plots.append(trainer.plot())
        else:
            if recency_weight:
                x = np.linspace(0, 1, len(self.dataset))
                self.dataset["weights"] = 0.25 + 0.75 * np.power(x, 3)
            trainer = Trainer(
                self.dataset,
                None,
                self.init_w,
                n_epoch=n_epoch,
                lr=lr,
                gamma=gamma,
                batch_size=batch_size,
                float_delta_t=self.float_delta_t,
                enable_short_term=self.enable_short_term,
            )
            w.append(trainer.train(verbose=verbose))
            if verbose:
                plots.append(trainer.plot())

        w = np.array(w)
        avg_w = np.round(np.mean(w, axis=0), 4)
        self.w = avg_w.tolist()

        if verbose:
            tqdm.write("\nTraining finished!")
        return plots

    def preview(self, requestRetention: float, verbose=False, n_steps=3):
        my_collection = Collection(self.w, self.float_delta_t)
        preview_text = "1:again, 2:hard, 3:good, 4:easy\n"
        n_learning_steps = n_steps if not self.float_delta_t else 0
        for first_rating in (1, 2, 3, 4):
            preview_text += f"\nfirst rating: {first_rating}\n"
            t_history = "0"
            d_history = "0"
            s_history = "0"
            r_history = f"{first_rating}"  # the first rating of the new card
            if first_rating in (1, 2):
                left = n_learning_steps
            elif first_rating == 3:
                left = n_learning_steps - 1
            else:
                left = 1
            # print("stability, difficulty, lapses")
            for i in range(10):
                states = my_collection.predict(t_history, r_history)
                stability = round(float(states[0]), 1)
                difficulty = round(float(states[1]), 1)
                if verbose:
                    print(
                        "{0:9.2f} {1:11.2f} {2:7.0f}".format(
                            *list(map(lambda x: round(float(x), 4), states))
                        )
                    )
                left -= 1
                next_t = (
                    next_interval(
                        states[0].detach().numpy(), requestRetention, self.float_delta_t
                    )
                    if left <= 0
                    else 0
                )
                t_history += f",{next_t}"
                d_history += f",{difficulty}"
                s_history += f",{stability}"
                r_history += f",3"
            r_history = wrap_short_term_ratings(r_history, t_history)
            preview_text += f"rating history: {r_history}\n"
            preview_text += (
                "interval history: "
                + ",".join(
                    [
                        (
                            f"{ivl:.4f}d"
                            if ivl < 1 and ivl > 0
                            else (
                                f"{ivl:.1f}d"
                                if ivl < 30
                                else (
                                    f"{ivl / 30:.1f}m"
                                    if ivl < 365
                                    else f"{ivl / 365:.1f}y"
                                )
                            )
                        )
                        for ivl in map(
                            int if not self.float_delta_t else float,
                            t_history.split(","),
                        )
                    ]
                )
                + "\n"
            )
            preview_text += (
                "factor history: "
                + ",".join(
                    ["0.0"]
                    + [
                        (
                            f"{float(ivl) / float(pre_ivl):.2f}"
                            if pre_ivl != "0"
                            else "0.0"
                        )
                        for ivl, pre_ivl in zip(
                            t_history.split(",")[1:],
                            t_history.split(",")[:-1],
                        )
                    ]
                )
                + "\n"
            )
            preview_text += f"difficulty history: {d_history}\n"
            preview_text += f"stability history: {s_history}\n"
        return preview_text

    def preview_sequence(self, test_rating_sequence: str, requestRetention: float):
        my_collection = Collection(self.w, self.float_delta_t)

        t_history = "0"
        d_history = "0"
        for i in range(len(test_rating_sequence.split(","))):
            r_history = test_rating_sequence[: 2 * i + 1]
            states = my_collection.predict(t_history, r_history)
            next_t = next_interval(
                states[0].detach().numpy(), requestRetention, self.float_delta_t
            )
            t_history += f",{next_t}"
            difficulty = round(float(states[1]), 1)
            d_history += f",{difficulty}"
        preview_text = f"rating history: {test_rating_sequence}\n"
        preview_text += (
            "interval history: "
            + ",".join(
                [
                    (
                        f"{ivl:.4f}d"
                        if ivl < 1 and ivl > 0
                        else (
                            f"{ivl:.1f}d"
                            if ivl < 30
                            else (
                                f"{ivl / 30:.1f}m" if ivl < 365 else f"{ivl / 365:.1f}y"
                            )
                        )
                    )
                    for ivl in map(
                        int if not self.float_delta_t else float,
                        t_history.split(","),
                    )
                ]
            )
            + "\n"
        )
        preview_text += (
            "factor history: "
            + ",".join(
                ["0.0"]
                + [
                    f"{float(ivl) / float(pre_ivl):.2f}" if pre_ivl != "0" else "0.0"
                    for ivl, pre_ivl in zip(
                        t_history.split(",")[1:],
                        t_history.split(",")[:-1],
                    )
                ]
            )
            + "\n"
        )
        preview_text += f"difficulty history: {d_history}"
        return preview_text

    def predict_memory_states(self):
        my_collection = Collection(self.w, self.float_delta_t)

        stabilities, difficulties = my_collection.batch_predict(self.dataset)
        stabilities = map(lambda x: round(x, 2), stabilities)
        difficulties = map(lambda x: round(x, 2), difficulties)
        self.dataset["stability"] = list(stabilities)
        self.dataset["difficulty"] = list(difficulties)
        prediction = self.dataset.groupby(by=["t_history", "r_history"]).agg(
            {"stability": "mean", "difficulty": "mean", "review_time": "count"}
        )
        prediction.reset_index(inplace=True)
        prediction.sort_values(by=["r_history"], inplace=True)
        prediction.rename(columns={"review_time": "count"}, inplace=True)
        prediction.to_csv("./prediction.tsv", sep="\t", index=None)
        prediction["difficulty"] = prediction["difficulty"].map(lambda x: int(round(x)))
        self.difficulty_distribution = (
            prediction.groupby(by=["difficulty"])["count"].sum()
            / prediction["count"].sum()
        )
        self.difficulty_distribution_padding = np.zeros(10)
        for i in range(10):
            if i + 1 in self.difficulty_distribution.index:
                self.difficulty_distribution_padding[i] = (
                    self.difficulty_distribution.loc[i + 1]
                )
        return self.difficulty_distribution

    def find_optimal_retention(
        self,
        learn_span=365,
        max_ivl=36500,
        verbose=True,
    ):
        """should not be called before predict_memory_states"""
        if verbose:
            print("Learn buttons: ", self.learn_buttons)
            print("Review buttons: ", self.review_buttons)
            print("First rating prob: ", self.first_rating_prob)
            print("Review rating prob: ", self.review_rating_prob)
            print("Learning step transitions: ", self.learning_step_transitions)
            print("Relearning step transitions: ", self.relearning_step_transitions)
            print("State rating costs: ", self.state_rating_costs)

        simulate_config = {
            "w": self.w,
            "deck_size": learn_span * 10,
            "learn_span": learn_span,
            "max_cost_perday": math.inf,
            "learn_limit_perday": 10,
            "review_limit_perday": math.inf,
            "max_ivl": max_ivl,
            "first_rating_prob": self.first_rating_prob,
            "review_rating_prob": self.review_rating_prob,
            "learning_step_transitions": self.learning_step_transitions,
            "relearning_step_transitions": self.relearning_step_transitions,
            "state_rating_costs": self.state_rating_costs,
        }
        self.optimal_retention = optimal_retention(**simulate_config)

        tqdm.write(
            f"\n-----suggested retention (experimental): {self.optimal_retention:.2f}-----"
        )

        if not verbose:
            return ()

        (
            _,
            review_cnt_per_day,
            learn_cnt_per_day,
            memorized_cnt_per_day,
            cost_per_day,
            _,
        ) = simulate(**simulate_config)

        def moving_average(data, window_size=365 // 20):
            weights = np.ones(window_size) / window_size
            return np.convolve(data, weights, mode="valid")

        fig1 = plt.figure()
        ax = fig1.gca()
        ax.plot(
            moving_average(review_cnt_per_day),
            label=f"R={self.optimal_retention*100:.0f}%",
        )
        ax.set_title("Review Count per Day")
        ax.legend()
        ax.grid(True)
        fig2 = plt.figure()
        ax = fig2.gca()
        ax.plot(
            moving_average(learn_cnt_per_day),
            label=f"R={self.optimal_retention*100:.0f}%",
        )
        ax.set_title("Learn Count per Day")
        ax.legend()
        ax.grid(True)
        fig3 = plt.figure()
        ax = fig3.gca()
        ax.plot(
            np.cumsum(learn_cnt_per_day), label=f"R={self.optimal_retention*100:.0f}%"
        )
        ax.set_title("Cumulative Learn Count")
        ax.legend()
        ax.grid(True)
        fig4 = plt.figure()
        ax = fig4.gca()
        ax.plot(memorized_cnt_per_day, label=f"R={self.optimal_retention*100:.0f}%")
        ax.set_title("Memorized Count per Day")
        ax.legend()
        ax.grid(True)

        fig5 = plt.figure()
        ax = fig5.gca()
        ax.plot(cost_per_day, label=f"R={self.optimal_retention*100:.0f}%")
        ax.set_title("Cost per Day")
        ax.legend()
        ax.grid(True)

        fig6 = workload_graph(simulate_config)

        return (fig1, fig2, fig3, fig4, fig5, fig6)

    def evaluate(self, save_to_file=True):
        my_collection = Collection(DEFAULT_PARAMETER, self.float_delta_t)
        if "tensor" not in self.dataset.columns:
            self.dataset["tensor"] = self.dataset.progress_apply(
                lambda x: lineToTensor(
                    list(zip([x["t_history"]], [x["r_history"]]))[0]
                ),
                axis=1,
            )
        stabilities, difficulties = my_collection.batch_predict(self.dataset)
        self.dataset["stability"] = stabilities
        self.dataset["difficulty"] = difficulties
        self.dataset["p"] = power_forgetting_curve(
            self.dataset["delta_t"],
            self.dataset["stability"],
            -my_collection.model.w[20].detach().numpy(),
        )
        self.dataset["log_loss"] = self.dataset.apply(
            lambda row: -np.log(row["p"]) if row["y"] == 1 else -np.log(1 - row["p"]),
            axis=1,
        )
        if "weights" not in self.dataset.columns:
            self.dataset["weights"] = 1
        self.dataset["log_loss"] = (
            self.dataset["log_loss"]
            * self.dataset["weights"]
            / self.dataset["weights"].mean()
        )
        loss_before = self.dataset["log_loss"].mean()

        my_collection = Collection(self.w, self.float_delta_t)
        stabilities, difficulties = my_collection.batch_predict(self.dataset)
        self.dataset["stability"] = stabilities
        self.dataset["difficulty"] = difficulties
        self.dataset["p"] = power_forgetting_curve(
            self.dataset["delta_t"],
            self.dataset["stability"],
            -my_collection.model.w[20].detach().numpy(),
        )
        self.dataset["log_loss"] = self.dataset.apply(
            lambda row: -np.log(row["p"]) if row["y"] == 1 else -np.log(1 - row["p"]),
            axis=1,
        )
        self.dataset["log_loss"] = (
            self.dataset["log_loss"]
            * self.dataset["weights"]
            / self.dataset["weights"].mean()
        )
        loss_after = self.dataset["log_loss"].mean()
        if save_to_file:
            tmp = self.dataset.copy()
            tmp["stability"] = tmp["stability"].map(lambda x: round(x, 2))
            tmp["difficulty"] = tmp["difficulty"].map(lambda x: round(x, 2))
            tmp["p"] = tmp["p"].map(lambda x: round(x, 2))
            tmp["log_loss"] = tmp["log_loss"].map(lambda x: round(x, 2))
            tmp.rename(columns={"p": "retrievability"}, inplace=True)
            tmp[
                [
                    "review_time",
                    "card_id",
                    "review_date",
                    "r_history",
                    "t_history",
                    "delta_t",
                    "review_rating",
                    "stability",
                    "difficulty",
                    "retrievability",
                    "log_loss",
                ]
            ].to_csv("./evaluation.tsv", sep="\t", index=False)
            del tmp
        return loss_before, loss_after

    def calibration_graph(self, dataset=None, verbose=True):
        if dataset is None:
            dataset = self.dataset
        fig1 = plt.figure()
        rmse = rmse_matrix(dataset)
        if verbose:
            tqdm.write(f"RMSE(bins): {rmse:.4f}")
        metrics_all = {}
        metrics = plot_brier(
            dataset["p"], dataset["y"], bins=20, ax=fig1.add_subplot(111)
        )
        metrics["RMSE(bins)"] = rmse
        metrics["AUC"] = (
            roc_auc_score(y_true=dataset["y"], y_score=dataset["p"])
            if len(dataset["y"].unique()) == 2
            else np.nan
        )
        metrics["LogLoss"] = log_loss(y_true=dataset["y"], y_pred=dataset["p"])
        metrics_all["all"] = metrics
        fig2 = plt.figure(figsize=(16, 12))
        for last_rating in (1, 2, 3, 4):
            calibration_data = dataset[dataset["last_rating"] == last_rating]
            if calibration_data.empty:
                continue
            rmse = rmse_matrix(calibration_data)
            if verbose:
                tqdm.write(f"\nLast rating: {last_rating}")
                tqdm.write(f"RMSE(bins): {rmse:.4f}")
            metrics = plot_brier(
                calibration_data["p"],
                calibration_data["y"],
                bins=20,
                ax=fig2.add_subplot(2, 2, int(last_rating)),
                title=f"Last rating: {last_rating}",
            )
            metrics["RMSE(bins)"] = rmse
            metrics["AUC"] = (
                roc_auc_score(
                    y_true=calibration_data["y"],
                    y_score=calibration_data["p"],
                )
                if len(calibration_data["y"].unique()) == 2
                else np.nan
            )
            metrics["LogLoss"] = log_loss(
                y_true=calibration_data["y"], y_pred=calibration_data["p"]
            )
            metrics_all[last_rating] = metrics

        fig3 = plt.figure()
        self.calibration_helper(
            dataset[["stability", "p", "y"]].copy(),
            "stability",
            lambda x: math.pow(1.2, math.floor(math.log(x, 1.2))),
            True,
            fig3.add_subplot(111),
        )

        fig4 = plt.figure(figsize=(16, 12))
        for last_rating in (1, 2, 3, 4):
            calibration_data = dataset[dataset["last_rating"] == last_rating]
            if calibration_data.empty:
                continue
            self.calibration_helper(
                calibration_data[["stability", "p", "y"]].copy(),
                "stability",
                lambda x: math.pow(1.2, math.floor(math.log(x, 1.2))),
                True,
                fig4.add_subplot(2, 2, int(last_rating)),
            )
        fig5 = plt.figure()
        self.calibration_helper(
            dataset[["difficulty", "p", "y"]].copy(),
            "difficulty",
            lambda x: round(x),
            False,
            fig5.add_subplot(111),
        )
        return metrics_all, (fig1, fig2, fig3, fig4, fig5)

    def calibration_helper(self, calibration_data, key, bin_func, semilogx, ax1):
        ax2 = ax1.twinx()
        lns = []

        def to_percent(temp, position):
            return "%1.0f" % (100 * temp) + "%"

        calibration_data["bin"] = calibration_data[key].map(bin_func)
        calibration_group = calibration_data.groupby("bin").count()

        lns1 = ax1.bar(
            x=calibration_group.index,
            height=calibration_group["y"],
            width=calibration_group.index / 5.5 if key == "stability" else 0.8,
            ec="k",
            lw=0.2,
            label="Number of predictions",
            alpha=0.5,
        )
        ax1.set_ylabel("Number of predictions")
        ax1.set_xlabel(key.title())
        if semilogx:
            ax1.semilogx()
        lns.append(lns1)

        calibration_group = calibration_data.groupby(by="bin").agg("mean")
        lns2 = ax2.plot(calibration_group["y"], label="Actual retention")
        lns3 = ax2.plot(calibration_group["p"], label="Predicted retention")
        ax2.set_ylabel("Retention")
        ax2.set_ylim(0, 1)
        lns.append(lns2[0])
        lns.append(lns3[0])

        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc="lower right")
        ax2.grid(linestyle="--")
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
        return ax1

    def formula_analysis(self):
        analysis_df = self.dataset[self.dataset["i"] > 2].copy()
        analysis_df["tensor"] = analysis_df["tensor"].map(lambda x: x[:-1])
        my_collection = Collection(self.w, self.float_delta_t)
        stabilities, difficulties = my_collection.batch_predict(analysis_df)
        analysis_df["last_s"] = stabilities
        analysis_df["last_d"] = difficulties
        analysis_df["last_delta_t"] = analysis_df["t_history"].map(
            lambda x: (
                int(x.split(",")[-1])
                if not self.float_delta_t
                else float(x.split(",")[-1])
            )
        )
        analysis_df["last_r"] = power_forgetting_curve(
            analysis_df["delta_t"], analysis_df["last_s"]
        )
        analysis_df["last_s_bin"] = analysis_df["last_s"].map(
            lambda x: math.pow(1.2, math.floor(math.log(x, 1.2)))
        )
        analysis_df["last_d_bin"] = analysis_df["last_d"].map(lambda x: round(x))
        bins = 20
        analysis_df["last_r_bin"] = analysis_df["last_r"].map(
            lambda x: (
                np.log(
                    np.minimum(np.floor(np.exp(np.log(bins + 1) * x) - 1), bins - 1) + 1
                )
                / np.log(bins)
            ).round(3)
        )
        figs = []
        for group_key in ("last_s_bin", "last_d_bin", "last_r_bin"):
            for last_rating in (1, 3):
                analysis_group = (
                    analysis_df[analysis_df["last_rating"] == last_rating]
                    .groupby(
                        by=["last_s_bin", "last_d_bin", "last_r_bin", "delta_t"],
                        group_keys=True,
                        as_index=False,
                    )
                    .agg(
                        {
                            "y": ["mean", "count"],
                            "p": "mean",
                            "stability": "mean",
                            "last_d": "mean",
                        }
                    )
                )
                analysis_group.columns = [
                    "_".join(col_name).rstrip("_")
                    for col_name in analysis_group.columns
                ]

                def cal_stability(tmp):
                    delta_t = tmp["delta_t"]
                    recall = tmp["y_mean"]
                    count = tmp["y_count"]
                    total_count = sum(count)

                    tmp["true_s"] = fit_stability(delta_t, recall, count)
                    tmp["predicted_s"] = np.average(
                        tmp["stability_mean"], weights=count
                    )
                    tmp["total_count"] = total_count
                    return tmp

                analysis_group = analysis_group.groupby(
                    by=[group_key], group_keys=False
                ).apply(cal_stability)
                analysis_group.dropna(inplace=True)
                analysis_group.drop_duplicates(subset=[group_key], inplace=True)
                analysis_group.sort_values(by=[group_key], inplace=True)
                mape = mean_absolute_percentage_error(
                    analysis_group["true_s"],
                    analysis_group["predicted_s"],
                    sample_weight=analysis_group["total_count"],
                )
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.set_title(f"MAPE={mape:.2f}, last rating={last_rating}")
                ax1.scatter(
                    analysis_group[group_key],
                    analysis_group["true_s"],
                    s=np.sqrt(analysis_group["total_count"]),
                    label="True stability",
                    alpha=0.5,
                )
                ax1.plot(
                    analysis_group[group_key],
                    analysis_group["predicted_s"],
                    label="Predicted stability",
                    color="orange",
                )
                ax1.set_ylim(0, analysis_group["predicted_s"].max() * 1.1)
                ax1.legend(loc="upper left")
                ax1.set_xlabel(group_key)
                if group_key == "last_s_bin":
                    ax1.set_ylim(
                        max(analysis_group["predicted_s"].min(), S_MIN),
                        analysis_group["predicted_s"].max() * 1.1,
                    )
                    ax1.set_xscale("log")
                    ax1.set_yscale("log")
                ax1.set_ylabel("Next Stability (days)")
                ax1.grid()
                ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                figs.append(fig)
        return figs

    def bw_matrix(self):
        B_W_Metric_raw = self.dataset[["difficulty", "stability", "p", "y"]].copy()
        B_W_Metric_raw["s_bin"] = B_W_Metric_raw["stability"].map(
            lambda x: round(math.pow(1.4, math.floor(math.log(x, 1.4))), 2)
        )
        B_W_Metric_raw["d_bin"] = B_W_Metric_raw["difficulty"].map(
            lambda x: int(round(x))
        )
        B_W_Metric = (
            B_W_Metric_raw.groupby(by=["s_bin", "d_bin"]).agg("mean").reset_index()
        )
        B_W_Metric_count = (
            B_W_Metric_raw.groupby(by=["s_bin", "d_bin"]).agg("count").reset_index()
        )
        B_W_Metric["B-W"] = B_W_Metric["p"] - B_W_Metric["y"]
        n = len(self.dataset)
        bins = len(B_W_Metric)
        B_W_Metric_pivot = B_W_Metric[
            B_W_Metric_count["p"] > max(50, n / (3 * bins))
        ].pivot(index="s_bin", columns="d_bin", values="B-W")
        return (
            B_W_Metric_pivot.apply(pd.to_numeric)
            .style.background_gradient(cmap="seismic", axis=None, vmin=-0.2, vmax=0.2)
            .format("{:.2%}", na_rep="")
        )

    def compare_with_sm2(self):
        self.dataset["sm2_ivl"] = self.dataset["tensor"].map(sm2)
        self.dataset["sm2_p"] = np.exp(
            np.log(0.9) * self.dataset["delta_t"] / self.dataset["sm2_ivl"]
        )
        self.dataset["log_loss"] = self.dataset.apply(
            lambda row: (
                -np.log(row["sm2_p"]) if row["y"] == 1 else -np.log(1 - row["sm2_p"])
            ),
            axis=1,
        )
        tqdm.write(f"Loss of SM-2: {self.dataset['log_loss'].mean():.4f}")
        dataset = self.dataset[["sm2_p", "p", "y"]].copy()
        dataset.rename(columns={"sm2_p": "R (SM2)", "p": "R (FSRS)"}, inplace=True)
        fig1 = plt.figure()
        plot_brier(
            dataset["R (SM2)"],
            dataset["y"],
            bins=20,
            ax=fig1.add_subplot(111),
        )
        universal_metrics, fig2 = cross_comparison(dataset, "SM2", "FSRS")

        tqdm.write(f"Universal Metric of FSRS: {universal_metrics[0]:.4f}")
        tqdm.write(f"Universal Metric of SM2: {universal_metrics[1]:.4f}")

        return fig1, fig2


# code from https://github.com/papousek/duolingo-halflife-regression/blob/master/evaluation.py
def load_brier(predictions, real, bins=20):
    # https://www.scirp.org/pdf/ojs_2021101415023495.pdf
    # Note that my implementation isn't exactly the same as in the paper, but it still has good coverage, better than Clopper-Pearson
    # I also made it possible to deal with k=0 and k=n, which was an issue with how this method is described in the paper
    def likelihood_interval(k, n, alpha=0.05):
        def log_likelihood(p: np.ndarray, k, n):
            assert k <= n
            p_hat = k / n

            def log_likelihood_f(k, n, p):
                one_minus_p = np.ones_like(p) - p
                if k == 0:
                    return n * np.log(one_minus_p)
                elif k == n:
                    return k * np.log(p)
                else:
                    return k * np.log(p) + (n - k) * np.log(one_minus_p)

            return log_likelihood_f(k, n, p) - log_likelihood_f(k, n, p_hat)

        def calc(x: np.ndarray, y: np.ndarray, target_p: float):
            def loss(guess_y: float, target_p: float) -> float:
                # Find segments where the horizontal line intersects the curve
                # This creates a boolean array where True indicates a potential intersection
                intersect_segments = ((y[:-1] <= guess_y) & (y[1:] >= guess_y)) | (
                    (y[:-1] >= guess_y) & (y[1:] <= guess_y)
                )

                # Get indices of segments where intersections occur
                intersection_indices = np.where(intersect_segments)[0]

                # If we don't have intersections, return a large error
                if len(intersection_indices) < 2:
                    return 1e100

                # Find the first two intersection points (we only need two for a connected curve)
                intersection_points = []

                for idx in intersection_indices[
                    :2
                ]:  # Take at most first two intersections
                    # Linear interpolation to find the x value at the intersection
                    x1, x2 = x[idx], x[idx + 1]
                    y1, y2 = y[idx], y[idx + 1]

                    # If points are exactly the same, just take the x
                    if y1 == y2:
                        intersection_points.append(x1)
                    else:
                        # Linear interpolation
                        t = (guess_y - y1) / (y2 - y1)
                        intersection_x = x1 + t * (x2 - x1)
                        intersection_points.append(intersection_x)

                # Get the range bounds
                x_low, x_high = min(intersection_points), max(intersection_points)

                # Find indices of x values that fall within our range
                in_range = (x >= x_low) & (x <= x_high)

                # Calculate the sum of probabilities in the range
                probability_sum = np.sum(y[in_range])

                # Return the absolute difference from target probability
                return abs(probability_sum - target_p)

            def bracket(xa, xb, maxiter, target_p):
                u_lim = xa
                l_lim = xb

                grow_limit = 100.0
                gold = 1.6180339
                verysmall_num = 1e-21

                fa = loss(xa, target_p)
                fb = loss(xb, target_p)
                funccalls = 2

                if fa < fb:  # Switch so fa > fb
                    xa, xb = xb, xa
                    fa, fb = fb, fa
                xc = max(min(xb + gold * (xb - xa), u_lim), l_lim)
                fc = loss(xc, target_p)
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
                    w = max(
                        min(
                            (xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom), u_lim
                        ),
                        l_lim,
                    )
                    wlim = max(min(xb + grow_limit * (xc - xb), u_lim), l_lim)

                    if iter > maxiter:
                        print("Failed to converge")
                        break

                    iter += 1
                    if (w - xc) * (xb - w) > 0.0:
                        fw = loss(w, target_p)
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
                        fw = loss(w, target_p)
                        funccalls += 1
                    elif (w - wlim) * (wlim - xc) >= 0.0:
                        w = wlim
                        fw = loss(w, target_p)
                        funccalls += 1
                    elif (w - wlim) * (xc - w) > 0.0:
                        fw = loss(w, target_p)
                        funccalls += 1
                        if fw < fc:
                            xb = max(min(xc, u_lim), l_lim)
                            xc = max(min(w, u_lim), l_lim)
                            w = max(min(xc + gold * (xc - xb), u_lim), l_lim)
                            fb = fc
                            fc = fw
                            fw = loss(w, target_p)
                            funccalls += 1
                    else:
                        w = max(min(xc + gold * (xc - xb), u_lim), l_lim)
                        fw = loss(w, target_p)
                        funccalls += 1
                    xa = max(min(xb, u_lim), l_lim)
                    xb = max(min(xc, u_lim), l_lim)
                    xc = max(min(w, u_lim), l_lim)
                    fa = fb
                    fb = fc
                    fc = fw

                return xa, xb, xc, fa, fb, fc, funccalls

            def brent_minimization(tol, maxiter):
                mintol = 1.0e-11
                cg = 0.3819660

                xa, xb, xc, fa, fb, fc, funccalls = bracket(
                    xa=min(y), xb=max(y), maxiter=maxiter, target_p=target_p
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
                    fu = loss(u, target_p)  # calculate new output value
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
                    # print(f'Iteration={iter}')
                    # print(f'x={x:.3f}')
                #################################
                # END
                #################################

                xmin = x
                fval = fx

                success = not (np.isnan(fval) or np.isnan(xmin)) and (0 <= xmin <= 1)

                if success:
                    # print(f'Loss function called {funccalls} times')
                    return xmin
                else:
                    raise Exception(
                        "The algorithm terminated without finding a valid value."
                    )

            best_guess_y = brent_minimization(1e-5, 50)

            intersect_segments = (
                (y[:-1] <= best_guess_y) & (y[1:] >= best_guess_y)
            ) | ((y[:-1] >= best_guess_y) & (y[1:] <= best_guess_y))
            intersection_indices = np.where(intersect_segments)[0]
            intersection_points = []

            for idx in intersection_indices[:2]:
                x1, x2 = x[idx], x[idx + 1]
                y1, y2 = y[idx], y[idx + 1]
                if y1 == y2:
                    intersection_points.append(x1)
                else:
                    t = (best_guess_y - y1) / (y2 - y1)
                    intersection_x = x1 + t * (x2 - x1)
                    intersection_points.append(intersection_x)

            x_low, x_high = min(intersection_points), max(intersection_points)
            in_range = (x >= x_low) & (x <= x_high)
            probability_sum = np.sum(y[in_range])
            return x_low, x_high, probability_sum

        p_hat = k / n
        # continuity correction
        if k == 0 or k == n:
            k, n = k + 0.5, n + 1

        probs = np.arange(1e-5, 1, 1e-5)

        likelihoods = np.exp(log_likelihood(probs, k, n))
        likelihoods = np.asarray(likelihoods)
        y = likelihoods / np.sum(likelihoods)

        x_low_cred, x_high_cred, probsum = calc(probs, y, 1 - alpha)
        assert 0 <= probsum <= 1

        if p_hat == 1.0:
            x_high_cred = 1.0
        elif p_hat == 0.0:
            x_low_cred = 0.0

        assert not np.isnan(x_low_cred)
        assert not np.isnan(x_high_cred)
        assert (
            x_low_cred <= p_hat <= x_high_cred
        ), f"{x_low_cred}, {p_hat}, {k / n}, {x_high_cred}"
        return x_low_cred, x_high_cred

    counts = np.zeros(bins)
    correct = np.zeros(bins)
    prediction = np.zeros(bins)

    two_d_list = [[] for _ in range(bins)]

    def get_bin(x, bins=bins):
        return np.floor(np.exp(np.log(bins + 1) * x)) - 1

    for p, r in zip(predictions, real):
        bin = int(min(get_bin(p), bins - 1))
        counts[bin] += 1
        correct[bin] += r
        prediction[bin] += p
        two_d_list[bin].append(r)  # for confidence interval calculations

    np.seterr(invalid="ignore")
    prediction_means = prediction / counts
    real_means = correct / counts
    size = len(predictions)
    answer_mean = sum(correct) / size

    real_means_upper = []
    real_means_lower = []
    for n in range(len(two_d_list)):
        if len(two_d_list[n]) > 0:
            lower_bound, upper_bound = likelihood_interval(
                sum(two_d_list[n]), len(two_d_list[n])
            )
        else:
            lower_bound, upper_bound = float("NaN"), float("NaN")
        real_means_upper.append(upper_bound)
        real_means_lower.append(lower_bound)

    assert len(real_means_lower) == len(prediction_means) == len(real_means_upper)
    # sanity check
    for n in range(len(real_means)):
        # check that the mean is within the bounds, unless they are NaNs
        if not np.isnan(real_means_lower[n]):
            assert (
                real_means_lower[n] <= real_means[n] <= real_means_upper[n]
            ), f"{real_means_lower[n]:4f}, {real_means[n]:4f}, {real_means_upper[n]:4f}"

    return {
        "reliability": sum(counts * (real_means - prediction_means) ** 2) / size,
        "resolution": sum(counts * (real_means - answer_mean) ** 2) / size,
        "uncertainty": answer_mean * (1 - answer_mean),
        "detail": {
            "bin_count": bins,
            "bin_counts": counts,
            "bin_prediction_means": prediction_means,
            "bin_real_means_upper_bounds": real_means_upper,
            "bin_real_means_lower_bounds": real_means_lower,
            "bin_real_means": real_means,
        },
    }


def plot_brier(predictions, real, bins=20, ax=None, title=None):
    y, p = zip(*sorted(zip(real, predictions), key=lambda x: x[1]))
    observation = lowess(
        y, p, it=0, delta=0.01 * (max(p) - min(p)), is_sorted=True, return_sorted=False
    )
    ici = np.mean(np.abs(observation - p))
    e_50 = np.median(np.abs(observation - p))
    e_90 = np.quantile(np.abs(observation - p), 0.9)
    e_max = np.max(np.abs(observation - p))
    brier = load_brier(predictions, real, bins=bins)
    bin_prediction_means = brier["detail"]["bin_prediction_means"]

    bin_real_means = brier["detail"]["bin_real_means"]
    bin_real_means_upper_bounds = brier["detail"]["bin_real_means_upper_bounds"]
    bin_real_means_lower_bounds = brier["detail"]["bin_real_means_lower_bounds"]
    bin_real_means_errors_upper = bin_real_means_upper_bounds - bin_real_means
    bin_real_means_errors_lower = bin_real_means - bin_real_means_lower_bounds

    bin_counts = brier["detail"]["bin_counts"]
    mask = bin_counts > 0
    r2 = r2_score(
        bin_real_means[mask],
        bin_prediction_means[mask],
        sample_weight=bin_counts[mask],
    )
    mae = mean_absolute_error(
        bin_real_means[mask],
        bin_prediction_means[mask],
        sample_weight=bin_counts[mask],
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    try:
        fit_wls = sm.WLS(
            bin_real_means[mask],
            sm.add_constant(bin_prediction_means[mask]),
            weights=bin_counts[mask],
        ).fit()
        params = fit_wls.params
        y_regression = [params[0] + params[1] * x for x in [0, 1]]
        ax.plot(
            [0, 1],
            y_regression,
            label=f"y = {params[0]:.3f} + {params[1]:.3f}x",
            color="green",
        )
    except:
        pass
    # ax.plot(
    #     bin_prediction_means[mask],
    #     bin_correct_means[mask],
    #     label="Actual Calibration",
    #     color="#1f77b4",
    #     marker="*",
    # )
    assert not any(np.isnan(bin_real_means_errors_upper[mask]))
    assert not any(np.isnan(bin_real_means_errors_lower[mask]))
    ax.errorbar(
        bin_prediction_means[mask],
        bin_real_means[mask],
        yerr=[bin_real_means_errors_lower[mask], bin_real_means_errors_upper[mask]],
        label="Actual Calibration",
        color="#1f77b4",
        ecolor="black",
        elinewidth=1.0,
        capsize=3.5,
        capthick=1.0,
        marker="",
    )
    # ax.plot(p, observation, label="Lowess Smoothing", color="red")
    ax.plot((0, 1), (0, 1), label="Perfect Calibration", color="#ff7f0e")
    bin_count = brier["detail"]["bin_count"]
    counts = np.array(bin_counts)
    bins = np.log((np.arange(bin_count)) + 1) / np.log(bin_count + 1)
    widths = np.diff(bins)
    widths = np.append(widths, 1 - bins[-1])
    ax.legend(loc="upper center")
    ax.set_xlabel("Predicted R")
    ax.set_ylabel("Actual R")
    ax2 = ax.twinx()
    ax2.set_ylabel("Number of reviews")
    ax2.bar(
        bins,
        counts,
        width=widths,
        ec="k",
        linewidth=0,
        alpha=0.5,
        label="Number of reviews",
        align="edge",
    )
    ax2.legend(loc="lower center")
    if title:
        ax.set_title(title)
    metrics = {
        "R-squared": r2,
        "MAE": mae,
        "ICI": ici,
        "E50": e_50,
        "E90": e_90,
        "EMax": e_max,
    }
    return metrics


def sm2(history):
    ivl = 0
    ef = 2.5
    reps = 0
    for delta_t, rating in history:
        delta_t = delta_t.item()
        rating = rating.item() + 1
        if rating > 2:
            if reps == 0:
                ivl = 1
                reps = 1
            elif reps == 1:
                ivl = 6
                reps = 2
            else:
                ivl = ivl * ef
                reps += 1
        else:
            ivl = 1
            reps = 0
        ef = max(1.3, ef + (0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02)))
        ivl = max(1, round(ivl + 0.01))
    return ivl


def cross_comparison(dataset, algoA, algoB):
    if algoA != algoB:
        cross_comparison_record = dataset[[f"R ({algoA})", f"R ({algoB})", "y"]].copy()
        bin_algo = (
            algoA,
            algoB,
        )
        pair_algo = [(algoA, algoB), (algoB, algoA)]
    else:
        cross_comparison_record = dataset[[f"R ({algoA})", "y"]].copy()
        bin_algo = (algoA,)
        pair_algo = [(algoA, algoA)]

    def get_bin(x, bins=20):
        return (
            np.log(np.minimum(np.floor(np.exp(np.log(bins + 1) * x) - 1), bins - 1) + 1)
            / np.log(bins)
        ).round(3)

    for algo in bin_algo:
        cross_comparison_record[f"{algo}_B-W"] = (
            cross_comparison_record[f"R ({algo})"] - cross_comparison_record["y"]
        )
        cross_comparison_record[f"{algo}_bin"] = cross_comparison_record[
            f"R ({algo})"
        ].map(get_bin)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.axhline(y=0.0, color="black", linestyle="-")

    universal_metric_list = []

    for algoA, algoB in pair_algo:
        cross_comparison_group = cross_comparison_record.groupby(by=f"{algoA}_bin").agg(
            {"y": ["mean"], f"{algoB}_B-W": ["mean"], f"R ({algoB})": ["mean", "count"]}
        )
        universal_metric = root_mean_squared_error(
            y_true=cross_comparison_group["y", "mean"],
            y_pred=cross_comparison_group[f"R ({algoB})", "mean"],
            sample_weight=cross_comparison_group[f"R ({algoB})", "count"],
        )
        cross_comparison_group[f"R ({algoB})", "percent"] = (
            cross_comparison_group[f"R ({algoB})", "count"]
            / cross_comparison_group[f"R ({algoB})", "count"].sum()
        )
        ax.scatter(
            cross_comparison_group.index,
            cross_comparison_group[f"{algoB}_B-W", "mean"],
            s=cross_comparison_group[f"R ({algoB})", "percent"] * 1024,
            alpha=0.5,
        )
        ax.plot(
            cross_comparison_group[f"{algoB}_B-W", "mean"],
            label=f"{algoB} by {algoA}, UM={universal_metric:.4f}",
        )
        universal_metric_list.append(universal_metric)

    ax.legend(loc="lower center")
    ax.grid(linestyle="--")
    ax.set_title(f"{algoA} vs {algoB}")
    ax.set_xlabel("Predicted R")
    ax.set_ylabel("B-W Metric")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    return universal_metric_list, fig


def rmse_matrix(df):
    tmp = df.copy()

    def count_lapse(r_history, t_history):
        lapse = 0
        for r, t in zip(r_history.split(","), t_history.split(",")):
            if t != "0" and r == "1":
                lapse += 1
        return lapse

    tmp["lapse"] = tmp.apply(
        lambda x: count_lapse(x["r_history"], x["t_history"]), axis=1
    )
    tmp["delta_t"] = tmp["delta_t"].map(
        lambda x: round(2.48 * np.power(3.62, np.floor(np.log(x) / np.log(3.62))), 2)
    )
    tmp["i"] = tmp["i"].map(
        lambda x: round(1.99 * np.power(1.89, np.floor(np.log(x) / np.log(1.89))), 0)
    )
    tmp["lapse"] = tmp["lapse"].map(
        lambda x: (
            round(1.65 * np.power(1.73, np.floor(np.log(x) / np.log(1.73))), 0)
            if x != 0
            else 0
        )
    )
    if "weights" not in tmp.columns:
        tmp["weights"] = 1
    tmp = (
        tmp.groupby(["delta_t", "i", "lapse"])
        .agg({"y": "mean", "p": "mean", "weights": "sum"})
        .reset_index()
    )
    return root_mean_squared_error(tmp["y"], tmp["p"], sample_weight=tmp["weights"])


def wrap_short_term_ratings(r_history, t_history):
    result = []
    in_zero_sequence = False

    for t, r in zip(t_history.split(","), r_history.split(",")):
        if t in ("-1", "0"):
            if not in_zero_sequence:
                result.append("(")
                in_zero_sequence = True
            result.append(r)
            result.append(",")
        else:
            if in_zero_sequence:
                result[-1] = "),"
                in_zero_sequence = False
            result.append(r)
            result.append(",")

    if in_zero_sequence:
        result[-1] = ")"
    else:
        result.pop()
    return "".join(result)


class FirstOrderMarkovChain:
    def __init__(self, n_states=4):
        """
        Initialize a first-order Markov chain model

        Parameters:
        n_states: Number of states, default is 4 (corresponding to states 1,2,3,4)
        """
        self.n_states = n_states
        self.transition_matrix = None
        self.initial_distribution = None
        self.transition_counts = None
        self.initial_counts = None

    def fit(self, sequences, smoothing=1.0):
        """
        Fit the Markov chain model based on given sequences

        Parameters:
        sequences: List of sequences, each sequence is a list containing 1,2,3,4
        smoothing: Laplace smoothing parameter to avoid zero probability issues
        """
        # Initialize transition count matrix and initial state counts
        self.transition_counts = np.zeros((self.n_states, self.n_states))
        self.initial_counts = np.zeros(self.n_states)

        # Count transition frequencies and initial state frequencies
        for sequence in sequences:
            if len(sequence) == 0:
                continue

            # Record initial state
            self.initial_counts[sequence[0] - 1] += 1

            # Record transitions
            for i in range(len(sequence) - 1):
                current_state = sequence[i] - 1  # Convert to 0-indexed
                next_state = sequence[i + 1] - 1  # Convert to 0-indexed
                self.transition_counts[current_state, next_state] += 1

        # Apply Laplace smoothing and calculate probabilities
        self.transition_counts += smoothing
        self.initial_counts += smoothing

        # Calculate transition probability matrix
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            row_sum = np.sum(self.transition_counts[i])
            if row_sum > 0:
                self.transition_matrix[i] = self.transition_counts[i] / row_sum
            else:
                # If a state never appears, assume uniform distribution
                self.transition_matrix[i] = np.ones(self.n_states) / self.n_states

        # Calculate initial state distribution
        self.initial_distribution = self.initial_counts / np.sum(self.initial_counts)

        return self

    def generate_sequence(self, length):
        """
        Generate a new sequence

        Parameters:
        length: Length of the sequence to generate

        Returns:
        Generated sequence (elements are 1,2,3,4)
        """
        if self.transition_matrix is None or self.initial_distribution is None:
            raise ValueError("Model not yet fitted, please call the fit method first")

        sequence = []

        # Generate initial state
        current_state = np.random.choice(self.n_states, p=self.initial_distribution)
        sequence.append(current_state + 1)  # Convert to 1-indexed

        # Generate subsequent states
        for _ in range(length - 1):
            current_state = np.random.choice(
                self.n_states, p=self.transition_matrix[current_state]
            )
            sequence.append(current_state + 1)  # Convert to 1-indexed

        return sequence

    def log_likelihood(self, sequences):
        """
        Calculate the log-likelihood of sequences

        Parameters:
        sequences: List of sequences

        Returns:
        Log-likelihood value
        """
        if self.transition_matrix is None or self.initial_distribution is None:
            raise ValueError("Model not yet fitted, please call the fit method first")

        log_likelihood = 0.0

        for sequence in sequences:
            if len(sequence) == 0:
                continue

            # Log probability of initial state
            log_likelihood += np.log(self.initial_distribution[sequence[0] - 1])

            # Log probability of transitions
            for i in range(len(sequence) - 1):
                current_state = sequence[i] - 1
                next_state = sequence[i + 1] - 1
                log_likelihood += np.log(
                    self.transition_matrix[current_state, next_state]
                )

        return log_likelihood

    def print_model(self):
        """Print model parameters"""
        print("Initial state distribution:")
        for i in range(self.n_states):
            print(f"State {i+1}: {self.initial_distribution[i]:.4f}")

        print("\nTransition probability matrix:")
        print("    | " + " ".join([f"  {i+1}  " for i in range(self.n_states)]))
        print("----+" + "------" * self.n_states)
        for i in range(self.n_states):
            print(
                f" {i+1}  | "
                + " ".join(
                    [f"{self.transition_matrix[i,j]:.4f}" for j in range(self.n_states)]
                )
            )

        print("Initial counts:")
        print(self.initial_counts.astype(int))
        print("Transition counts:")
        print(self.transition_counts.astype(int))
