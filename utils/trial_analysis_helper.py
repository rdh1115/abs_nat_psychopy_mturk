import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from matplotlib import pyplot as plt

from utils.helper import _subpath_after, TaskConfig


def mturk_localize(resource_dir, p: str):
    segment = "HvM_with_discfade"
    relative = _subpath_after(Path(p), segment)
    relative = relative.with_name(relative.name.lstrip("_"))
    return resource_dir / relative


def parse_psychopy_date(date_str):
    """
    Convert PsychoPy MTurk date like '2025-11-07_16h15.44.244'
    into a proper pandas Timestamp.
    """
    if pd.isna(date_str):
        return pd.NaT

    # Example: 2025-11-07_16h15.44.244
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})\.(\d{2})\.(\d{3})", str(date_str))
    if not m:
        return pd.NaT

    yyyy_mm_dd, hh, mm, ss, ms = m.groups()

    fmt = f"{yyyy_mm_dd} {hh}:{mm}:{ss}.{ms}"

    try:
        return pd.to_datetime(fmt)
    except Exception:
        return pd.NaT


def make_unique_columns(cols: Iterable) -> List[str]:
    """
    Make column names unique by appending suffixes: 'resp.rt', 'resp.rt__1', ...
    Also converts NaNs to empty strings.
    """
    seen = {}
    out = []
    for c in cols:
        if pd.isna(c):
            c = ""
        c = str(c)
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out


def load_blocks_from_csv(responses_csv, header_token="Welcome.started"):
    """
    Read the raw PsychoPy MTurk CSV that has repeated header rows
    and split it into blocks. Each block gets its *own* header row
    (the row whose first cell == header_token).

    Returns a list of per-block DataFrames.
    """
    raw = responses_csv

    # Find header rows
    header_mask = raw.iloc[:, 0] == header_token
    header_idx = list(raw.index[header_mask])
    header_idx.append(len(raw))

    blocks = []
    for start, end in zip(header_idx[:-1], header_idx[1:]):
        block = raw.iloc[start:end].reset_index(drop=True)
        if block.shape[0] <= 1:
            continue  # header only, no data

        header_row = block.iloc[0].tolist()
        data = block.iloc[1:].copy()
        data.columns = header_row

        last_real_col = 0
        for i, col in enumerate(header_row):
            if not pd.isna(col) and str(col).strip() != "":
                last_real_col = i

        data = data.iloc[:, : last_real_col + 1]
        header_row = header_row[:last_real_col + 1]
        data = data.iloc[:, :-3]  # last 2 columns are worker ID and time
        header_row = header_row[:-3]
        data.columns = make_unique_columns(header_row)
        data = data.dropna(how="all")

        if not data.empty:
            blocks.append(data)

    return blocks


def split_by_worker_and_exp(
        csv_path,
        out_dir,
        worker_list=None,
        task_list=None,
        header_token="Welcome.started",
):
    """
    Read a CSV with columns including 'workerId' and 'expName',
    and save one CSV per (workerId, expName) group.
    """
    csv_path = Path(csv_path)
    responses_csv = pd.read_csv(csv_path, header=None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = load_blocks_from_csv(responses_csv, header_token=header_token)
    common_cols = blocks[0].columns
    # print("Columns to be used across all experiments:")
    # print(common_cols)
    aligned_blocks = [b.reindex(columns=common_cols) for b in blocks]

    # Concatenate all blocks into one big DataFrame
    df = pd.concat(aligned_blocks, ignore_index=True)
    header = list(df.columns)

    embedded_header_rows = df.apply(
        lambda r: any(str(val).strip() in header for val in r),
        axis=1
    )
    df = df[~embedded_header_rows]

    # Ensure required columns exist
    required_cols = {"workerId", "expName", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {required_cols}. Missing: {missing}")

    clean_df = df[
        df["workerId"].notna()
        & (df["workerId"].astype(str).str.strip() != "")
        & (df["workerId"] != "workerId")
        & df["expName"].notna()
        & df["date"].notna()
        ].copy()

    if worker_list:
        clean_df = clean_df[clean_df["workerId"].isin(worker_list)]

    if task_list:
        clean_df = clean_df[clean_df["expName"].isin(task_list)]

    if clean_df.empty:
        print("No valid rows after filtering; nothing to split.")
        return

    clean_df["date_parsed"] = clean_df["date"].apply(parse_psychopy_date)

    all_dates_for_worker_exp = defaultdict(set)
    indices_by_submission = defaultdict(list)
    for idx, row in clean_df.iterrows():
        worker = str(row["workerId"]).strip()
        exp = str(row["expName"]).strip()
        date = row["date_parsed"]
        we = (worker, exp)
        key = (worker, exp, date)
        all_dates_for_worker_exp[we].add(date)
        indices_by_submission[key].append(idx)

    worker_exp_final = dict()
    for (worker, exp), date_set in all_dates_for_worker_exp.items():
        if len(date_set) > 1:
            print(f"Multiple submissions detected for worker={worker}, exp={exp}:")
            print(f"  Dates: {date_set}")
            earliest_date = sorted(list(date_set))[0]
            print(f"  Keeping first-seen date: {earliest_date}")

            earliest_indices = indices_by_submission[(worker, exp, earliest_date)]
        else:
            earliest_indices = indices_by_submission[(worker, exp, date_set.pop())]
        worker_exp_final[(worker, exp)] = earliest_indices

    for (worker, exp), idx_list in worker_exp_final.items():
        sub_df = clean_df.loc[idx_list].reset_index(drop=True)
        fname = f"{worker}_{exp}.csv"
        sub_df.to_csv(out_dir / fname, index=False)
        print(f"Saved {len(sub_df)} rows to {out_dir / fname}")
    return


def analyze_accuracy(
        behaviour_df: pd.DataFrame,
        trial_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        task_config: TaskConfig,
):
    n_frames = len(task_config.stim_cols)
    if task_config.trial_start_col not in behaviour_df.columns:
        raise ValueError(f"Column '{task_config.trial_start_col}' not found in behaviour_df")

    meta_lookup = meta_df.set_index('filename').to_dict(orient="index")
    trials = []
    details = []
    trial_start_indices = behaviour_df.index[
        behaviour_df[task_config.trial_start_col].notna()
    ].tolist()
    n_trials = min(len(trial_start_indices), len(trial_df))

    if len(trial_start_indices) != len(trial_df):
        print(
            f"Warning: #trial starts in log ({len(trial_start_indices)}) != "
            f"#rows in condition file ({len(trial_df)}). Using the minimum of the two."
        )

    for i in range(n_trials):
        start_idx = trial_start_indices[i]
        resp_indices = [r_i + start_idx - 1 for r_i in task_config.act_indices]
        act_idx = start_idx + n_frames  # row containing correct answers

        if act_idx >= len(behaviour_df):
            break

        responses = behaviour_df.loc[resp_indices, task_config.resp_col].tolist()
        if task_config.rt_col in behaviour_df.columns:
            rts = behaviour_df.loc[resp_indices, task_config.rt_col].tolist()
        else:
            rts = [None] * len(resp_indices)

        # correct answers from the behavior log
        correct_answers = behaviour_df.loc[act_idx, task_config.act_cols].tolist()
        cond_row = trial_df.iloc[i]
        assert correct_answers == trial_df.loc[
            i, task_config.act_cols].tolist(), f'{correct_answers} vs. {trial_df.loc[i, task_config.act_cols].tolist()}'

        # session info from condition file (if exists)
        session = cond_row["session"] if "session" in trial_df.columns else None

        # compute correctness & missing
        trial_correct = []
        trial_missing = 0
        for r, c in zip(responses, correct_answers):
            r_stripped = None if pd.isna(r) else str(r).strip()
            c_stripped = None if pd.isna(c) else str(c).strip()

            if r_stripped in (None, ""):
                trial_missing += 1

            if r_stripped is None or c_stripped is None or r_stripped == "":
                trial_correct.append(False)
            else:
                trial_correct.append(r_stripped.lower() == c_stripped.lower())

        n_correct = sum(trial_correct)
        n_total = len(trial_correct)
        accuracy = n_correct / n_total if n_total > 0 else float("nan")

        trials.append({
            "trial_id": i + 1,
            "session": session,
            "n_correct": n_correct,
            "n_total": n_total,
            "accuracy": accuracy,
            "n_missing": trial_missing,
        })

        # detailed per-response info
        for j, (idx, r, rt, c, corr) in enumerate(
                zip(resp_indices, responses, rts, correct_answers, trial_correct), start=1
        ):
            r_stripped = None if pd.isna(r) else str(r).strip()
            is_missing = (r_stripped in (None, ""))

            # current stimulus is frame j+1 ->
            stim_col = f"stim{j + 1}"
            stim_path = Path(cond_row.get(stim_col, None))
            stim_path = stim_path.with_name(stim_path.name.lstrip("_"))
            meta_row = meta_lookup[stim_path]
            feature_values = dict()
            for feat in task_config.features:
                feature_values[feat] = meta_row.get(feat, None)

            details_row = {
                "trial_id": i + 1,
                "session": session,
                "resp_index": j,
                "csv_row": idx,
                "response": r,
                "rt": rt,
                "correct_answer": c,
                "is_correct": corr,
                "is_missing": is_missing,
                "stim_col": stim_col,
                "stim_path": stim_path,
            }
            details_row.update(feature_values)
            details.append(details_row)

    summary_df = pd.DataFrame(trials)
    detail_df = pd.DataFrame(details)

    # Overall accuracies
    overall_trial_accuracy = summary_df["accuracy"].mean()
    overall_frame_accuracy = detail_df["is_correct"].mean()

    print(f"\nAnalyzed {len(trial_df)} trials")
    print(f"Overall (mean per-trial) accuracy: {overall_trial_accuracy:.2%}")
    print(f"Overall (per-frame) accuracy:      {overall_frame_accuracy:.2%}\n")
    return summary_df, detail_df


def task_analysis(
        summary_df: pd.DataFrame,
        detail_df: pd.DataFrame,
        task_config: TaskConfig,
        output_prefix: Path,
):
    # -------- Session-level summaries --------
    if "session" in summary_df.columns and summary_df["session"].notna().any():
        # missing responses per session
        missing_per_session = detail_df.groupby("session")["is_missing"].sum().rename("n_missing").reset_index()

        # frame-wise accuracy per session
        frame_acc_per_session = detail_df.groupby("session")["is_correct"].mean().rename("frame_accuracy")

        # trial-wise accuracy per session
        trial_acc_per_session = summary_df.groupby("session")["accuracy"].mean().rename("trial_accuracy")

        session_summary = (
            summary_df.groupby("session")["trial_id"]
            .nunique()
            .rename("n_trials")
            .to_frame()
            .join(frame_acc_per_session)
            .join(trial_acc_per_session)
            .join(missing_per_session.set_index("session"))
            .reset_index()
        )

        print("Session-level summary:")
        print(session_summary.to_string(index=False))
        print()
    else:
        session_summary = pd.DataFrame()
        print("No 'session' information available for session-level summary.\n")

    feature_session_summaries = {}

    for feat in task_config.features:
        feat_df = detail_df[detail_df[feat].notna()].copy()
        feat_session_summary = (
            feat_df.groupby(["session", feat])
            .agg(
                n_samples=("is_correct", "size"),
                accuracy=("is_correct", "mean"),
                n_missing=("is_missing", "sum"),
            )
            .reset_index()
        )
        feature_session_summaries[feat] = feat_session_summary

        print(f"Per-{feat} accuracy per session:")
        print(feat_session_summary.to_string(index=False))
        print()

    # -------- RT distribution across all sessions --------
    if "rt" in detail_df.columns and detail_df["rt"].notna().any():
        rt_series = pd.to_numeric(detail_df["rt"], errors="coerce").dropna()
        if not rt_series.empty:
            rt_fig_path = output_prefix / "rt_distribution.png"

            plt.figure(figsize=(6, 4))
            plt.hist(rt_series, bins=30)
            plt.xlabel("Response time (s)")
            plt.ylabel("Count")
            plt.title(f"{task_config.task_name}: RT distribution (all sessions)")
            plt.tight_layout()
            plt.savefig(rt_fig_path, dpi=150)
            plt.close()

            print(f"Saved RT distribution plot to: {rt_fig_path}")
        elif rt_series.empty:
            print("No valid RT values found; RT distribution plot not generated.")
    else:
        print("Column for RT not found or empty; RT distribution plot not generated.")

    # -------- Save outputs --------
    if output_prefix is not None:
        trial_path = output_prefix / "trial_accuracy.csv"
        detail_path = output_prefix / "detailed_accuracy.csv"
        session_path = output_prefix / "session_summary.csv"

        summary_df.to_csv(trial_path, index=False)
        detail_df.to_csv(detail_path, index=False)
        session_summary.to_csv(session_path, index=False)

        print(f"Saved per-trial accuracy to:    {trial_path}")
        print(f"Saved detailed responses to:    {detail_path}")
        print(f"Saved per-session summary to:   {session_path}")

        # Save per-feature summaries
        for feat, feat_df in feature_session_summaries.items():
            feat_path = output_prefix / f"session_{feat}_accuracy.csv"
            feat_df.to_csv(feat_path, index=False)
            print(f"Saved per-session/{feat} stats to: {feat_path}")
    return {
        "trial_summary": summary_df,
        "detail": detail_df,
        "session_summary": session_summary,
        "feature_session_summaries": feature_session_summaries,
    }
