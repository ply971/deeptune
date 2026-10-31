"""
The rows shared by a DeepTune embeddings Parquet file and a CSV file, usually the 40% holdout df-analyze set aside.

    python -m handlers.get_intersection --df_parquet_path embeddings.parquet --df_csv_path holdout.csv --out results

Works for every modality, and imports no torch. A CSV cannot hold a column's type, so numbers are compared as float32 (as
written by DeepTune) and column names as text, since a frame of embeddings may name its columns with the integers 0, 1, ...
The output keeps the Parquet file's values and row order, with column names as text, plus any columns only the CSV has. A
row of the Parquet file that matches several identical CSV rows appears once. At least one shared column must be a feature:
sharing only the target or an id would match every row with that value. Names cannot show that for a custom target, so the
shared columns must also tell at least half of the Parquet rows apart.
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


NON_FEATURE_COLUMNS = {'label', 'labels', 'target', 'split', 'group', 'video_id', 'row_id'}


def _comparable(column: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(column):
        return column.astype(np.float32)
    if pd.api.types.is_numeric_dtype(column):
        return column.astype(np.float32)
    return column.astype(str)


def intersect(df_parquet: pd.DataFrame, df_csv: pd.DataFrame) -> pd.DataFrame:
    """Rows of `df_parquet` whose values on all shared columns appear in `df_csv`."""
    left = df_parquet.rename(columns=str).reset_index(drop=True)
    right = df_csv.rename(columns=str).reset_index(drop=True)
    for name, frame in (('Parquet', left), ('CSV', right)):
        if frame.columns.duplicated().any():
            raise ValueError(f'The {name} file has repeated column names, so its rows cannot be compared.')
    common = [c for c in left.columns if c in right.columns]
    if not common:
        raise ValueError(f'The two files share no column. Parquet: {list(left.columns)[:8]}...; CSV: {list(right.columns)[:8]}...')
    if not set(common) - NON_FEATURE_COLUMNS:
        raise ValueError(f'The two files share only {common}, which are no features, so every row with the same label would match. '
                         'Do the two files have the same feature columns? Compare them after the same export.')
    keys_left = pd.DataFrame({c: _comparable(left[c]) for c in common})
    distinct = len(keys_left.drop_duplicates())
    if distinct * 2 < len(left):  # any name: a target called 'diagnosis' is no feature either
        raise ValueError(f'The shared columns {common[:8]} tell only {distinct} of {len(left)} Parquet rows apart, so they look like a target or an id, '
                         'not features. Compare the two files after the same export.')
    keys_right = pd.DataFrame({c: _comparable(right[c]) for c in common})
    only_csv = [c for c in right.columns if c not in common]
    keys_right = pd.concat([keys_right, right[only_csv]], axis=1).drop_duplicates(subset=common)
    keys_left['__row__'] = np.arange(len(left))
    merged = keys_left.merge(keys_right, on=common, how='inner').sort_values('__row__', kind='stable')
    result = left.iloc[merged['__row__'].to_numpy()].reset_index(drop=True)
    for column in only_csv:
        result[column] = merged[column].to_numpy()
    return result


def main(argv=None) -> Path:
    args = make_parser().parse_args(argv)
    df_parquet = pd.read_parquet(args.df_parquet_path)
    df_csv = pd.read_csv(args.df_csv_path)
    result = intersect(df_parquet, df_csv)
    print(f'There are {len(result)} records in intersection between both dataframes '
          f'({len(df_parquet)} in the Parquet file, {len(df_csv)} in the CSV file).')
    if not len(result):
        print('Nothing matched. Was the CSV changed after export, for example normalised by df-analyze?')
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"intersection_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet"
    result.to_parquet(path)
    print(f'Saved to {path}')
    return path


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Obtain the intersection between a parquet file and csv file, mainly implemented to extract the 40 percent holdout dataset used by df-analyze from Deeptune embeddings representations fed.')
    parser.add_argument('--df_parquet_path', type=Path, required=True,
                        help="Path to the Parquet file, usually DeepTune's embeddings.")
    parser.add_argument('--df_csv_path', type=Path, required=True,
                        help='Path to the CSV file, usually the 40 percent subset obtained from df-analyze.')
    parser.add_argument('--out', type=Path, required=True,
                        help='Folder for the result, intersection_<yyyymmdd_hhmm>.parquet. Created if missing.')
    return parser


if __name__ == "__main__":
    main()
