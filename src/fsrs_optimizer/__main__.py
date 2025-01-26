import fsrs_optimizer
import argparse
import shutil
import json
import pytz
import os
import functools
from pathlib import Path

import matplotlib.pyplot as plt


def prompt(msg: str, fallback):
    default = ""
    if fallback:
        default = f"(default: {fallback})"

    response = input(f"{msg} {default}: ")
    if response == "":
        if fallback is not None:
            return fallback
        else:  # If there is no fallback
            raise Exception("You failed to enter a required parameter")
    return response


def process(filepath, filter_out_flags: list[int]):
    suffix = filepath.split("/")[-1].replace(".", "_").replace("@", "_")
    proj_dir = Path(f"{suffix}")
    proj_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(proj_dir)

    try:  # Try and remember the last values inputted.
        with open(config_save, "r") as f:
            remembered_fallbacks = json.load(f)
    except FileNotFoundError:
        remembered_fallbacks = {  # Defaults to this if not there
            "timezone": None,  # Timezone starts with no default
            "next_day": 4,
            "revlog_start_date": "2006-10-05",
            "preview": "y",
            "filter_out_suspended_cards": "n",
            "enable_short_term": "y",
        }

    # Prompts the user with the key and then falls back on the last answer given.
    def remembered_fallback_prompt(key: str, pretty: str = None):
        if pretty is None:
            pretty = key
        remembered_fallbacks[key] = prompt(
            f"input {pretty}", remembered_fallbacks.get(key, None)
        )

    print("The defaults will switch to whatever you entered last.\n")

    if not args.yes:
        print(
            "Timezone list: https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568"
        )
        remembered_fallback_prompt("timezone", "used timezone")
        if remembered_fallbacks["timezone"] not in pytz.all_timezones:
            raise Exception("Not a valid timezone, Check the list for more information")

        remembered_fallback_prompt("next_day", "used next day start hour")
        remembered_fallback_prompt(
            "revlog_start_date",
            "the date at which before reviews will be ignored | YYYY-MM-DD",
        )
        remembered_fallback_prompt(
            "filter_out_suspended_cards", "filter out suspended cards? (y/n)"
        )
        remembered_fallback_prompt(
            "enable_short_term", "enable short-term component in FSRS model? (y/n)"
        )

        graphs_input = prompt("Save graphs? (y/n)", remembered_fallbacks["preview"])
    else:
        graphs_input = remembered_fallbacks["preview"]

    if graphs_input.lower() != "y":
        remembered_fallbacks["preview"] = "n"
    else:
        remembered_fallbacks["preview"] = "y"

    with open(
        config_save, "w+"
    ) as f:  # Save the settings to load next time the program is run
        json.dump(remembered_fallbacks, f)

    save_graphs = graphs_input != "n"
    enable_short_term = remembered_fallbacks["enable_short_term"] == "y"

    optimizer = fsrs_optimizer.Optimizer(enable_short_term=enable_short_term)
    if filepath.endswith(".apkg") or filepath.endswith(".colpkg"):
        optimizer.anki_extract(
            f"{filepath}",
            remembered_fallbacks["filter_out_suspended_cards"] == "y",
            filter_out_flags,
        )
    else:
        # copy the file to the current directory and rename it as revlog.csv
        shutil.copyfile(f"{filepath}", "revlog.csv")
    analysis = optimizer.create_time_series(
        remembered_fallbacks["timezone"],
        remembered_fallbacks["revlog_start_date"],
        remembered_fallbacks["next_day"],
        save_graphs,
    )
    print(analysis)

    filename = os.path.splitext(os.path.basename(filepath))[0]

    optimizer.define_model()
    figures = optimizer.pretrain(verbose=save_graphs)
    for i, f in enumerate(figures):
        f.savefig(f"pretrain_{i}.png")
        plt.close(f)
    figures = optimizer.train(verbose=save_graphs, recency_weight=True)
    for i, f in enumerate(figures):
        f.savefig(f"train_{i}.png")
        plt.close(f)

    optimizer.predict_memory_states()
    try:
        figures = optimizer.find_optimal_retention(verbose=save_graphs)
        for i, f in enumerate(figures):
            f.savefig(f"find_optimal_retention_{i}.png")
            plt.close(f)
    except Exception as e:
        print(e)
        print("Failed to find optimal retention")
        optimizer.optimal_retention = 0.9

    print(optimizer.preview(optimizer.optimal_retention))

    profile = f"""{{
    // Generated, Optimized anki deck settings
    "deckName": "{filename}",// PLEASE CHANGE THIS TO THE DECKS PROPER NAME
    "w": {optimizer.w},
    "requestRetention": {optimizer.optimal_retention},
    "maximumInterval": 36500,
}},
"""

    print("Paste this into your scheduling code")
    print(profile)

    if args.out:
        with open(args.out, "a+") as f:
            f.write(profile)

    loss_before, loss_after = optimizer.evaluate()
    print(f"Loss before training: {loss_before:.4f}")
    print(f"Loss after training: {loss_after:.4f}")
    metrics, figures = optimizer.calibration_graph(verbose=False)
    for partition in metrics:
        print(f"Last rating = {partition}")
        for metric in metrics[partition]:
            print(f"{metric}: {metrics[partition][metric]:.4f}")
        print()

    metrics["Log loss"] = loss_after
    if save_graphs:
        for i, f in enumerate(figures):
            f.savefig(f"calibration_{i}.png")
            plt.close(f)
    figures = optimizer.formula_analysis()
    if save_graphs:
        for i, f in enumerate(figures):
            f.savefig(f"formula_analysis_{i}.png")
            plt.close(f)
    figures = optimizer.compare_with_sm2()
    if save_graphs:
        for i, f in enumerate(figures):
            f.savefig(f"compare_with_sm2_{i}.png")
            plt.close(f)

    evaluation = {
        "filename": filename,
        "size": optimizer.dataset.shape[0],
        "parameters": optimizer.w,
        "metrics": metrics,
    }

    with open("evaluation.json", "w+") as f:
        json.dump(evaluation, f)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("filenames", nargs="+")
    parser.add_argument(
        "-y",
        "--yes",
        action=argparse.BooleanOptionalAction,
        help="If set automatically defaults on all stdin settings.",
    )
    parser.add_argument(
        "--flags",
        help="Remove any cards with the given flags from the training set.",
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "-o", "--out", help="File to APPEND the automatically generated profile to."
    )

    return parser


if __name__ == "__main__":
    config_save = os.path.expanduser(".fsrs_optimizer")

    parser = create_arg_parser()
    args = parser.parse_args()

    def lift(file_or_dir):
        return os.listdir(file_or_dir) if os.path.isdir(file_or_dir) else [file_or_dir]

    def flatten(fl):
        return sum(fl, [])

    def mapC(f):
        return lambda x: map(f, x)

    def filterC(f):
        return lambda x: filter(f, x)

    def pipe(functions, value):
        return functools.reduce(lambda out, f: f(out), functions, value)

    curdir = os.getcwd()

    files = pipe(
        [
            mapC(lift),  # map file to [ file ], dir to [ file1, file2, ... ]
            flatten,  # flatten into [ file1, file2, ... ]
            mapC(os.path.abspath),  # map to absolute path
            filterC(lambda f: not os.path.isdir(f)),  # file filter
            filterC(
                lambda f: f.lower().endswith(".apkg")
                or f.lower().endswith(".colpkg")
                or f.lower().endswith(".csv")
            ),  # extension filter
        ],
        args.filenames,
    )

    for filename in files:
        try:
            print(f"Processing {filename}")
            process(filename, args.flags)
        except Exception as e:
            print(e)
            print(f"Failed to process {filename}")
        finally:
            plt.close("all")
            os.chdir(curdir)
            continue
