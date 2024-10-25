# FSRS Optimizer

[![PyPi](https://img.shields.io/pypi/v/FSRS-Optimizer)](https://pypi.org/project/FSRS-Optimizer/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The FSRS Optimizer is a Python library capable of utilizing personal spaced repetition review logs to refine the FSRS algorithm. Designed with the intent of delivering a standardized, universal optimizer to various FSRS implementations across numerous programming languages, this tool is set to establish a ubiquitous standard for spaced repetition review logs. By facilitating the uniformity of learning data among different spaced repetition softwares, it guarantees learners consistent review schedules across a multitude of platforms.

Delve into the underlying principles of the FSRS Optimizer's training process at: https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-mechanism-of-optimization

Explore the mathematical formula of the FSRS model at: https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm

# Review Logs Schema

The `review_logs` table captures the review activities performed by users. Each log records the details of a single review instance. The schema for this table is as follows:

| Column Name | Data Type | Description | Constraints |
|-------------|-----------|-------------|-------------|
| card_id | integer or string | The unique identifier of the flashcard being reviewed | Not null |
| review_time | timestamp  in *miliseconds* | The exact moment when the review took place | Not null |
| review_rating | integer | The user's rating for the review. This rating is subjective and depends on how well the user believes they remembered the information on the card | Not null, Values: {1 (Again), 2 (Hard), 3 (Good), 4 (Easy)} |
| review_state | integer | The state of the card at the time of review. This describes the learning phase of the card | Optional, Values: {0 (New), 1 (Learning), 2 (Review), 3 (Relearning)} |
| review_duration | integer | The time spent on reviewing the card, typically in miliseconds | Optional, Non-negative |

Extra Info:
- `timezone`: The time zone of the user when they performed the review, which is used to identify the start of a new day.
- `day_start`: The hour (0-23) at which the user starts a new day, which is used to separate reviews that are divided by sleep into different days.

Notes:
- All timestamp fields are expected to be in UTC.
- The `card_id` should correspond to a valid card in the corresponding flashcards dataset.
- `review_rating` should be a reflection of the user's memory of the card at the time of the review.
- `review_state` helps to understand the learning progress of the card.
- `review_duration` measures the cost of the review.
- `timezone` should be a string from the IANA Time Zone Database (e.g., "America/New_York"). For more information, refer to this [list of IANA time zones](https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568).
- `day_start` determines the start of the learner's day and is used to correctly assign reviews to days, especially when reviews are divided by sleep.

Please ensure your data conforms to this schema for optimal compatibility with the optimization process.

# Optimize FSRS with your review logs

**Installation**

Install the package with the command:

```
python -m pip install fsrs-optimizer
```

You should upgrade regularly to make sure you have the most recent version of FSRS-Optimizer:

```
python -m pip install fsrs-optimizer --upgrade
```

**Opimization**

If you have a file named `revlog.csv` with the above schema, you can run:

```
python -m fsrs_optimizer "revlog.csv"
```

**Expected Functionality**

![image](https://github.com/open-spaced-repetition/fsrs-optimizer/assets/32575846/fad7154a-9667-4eea-b868-d94c94a50912)

![image](https://github.com/open-spaced-repetition/fsrs-optimizer/assets/32575846/f868aac4-2e9e-4101-b8ad-eccc1d9b1bd5)

---

## Alternative

Are you getting tired of installing torch? Try [fsrs-rs-python](https://github.com/open-spaced-repetition/fsrs-rs-python)!
