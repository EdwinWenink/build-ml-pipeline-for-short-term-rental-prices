#!/usr/bin/env python
"""
Download the raw dataset from W&B and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
from typing import List

import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def remove_outliers(df: pd.DataFrame, min_price: float, max_price: float, price_col='price') -> pd.DataFrame:
    """
    Remove rows with rents that do not fall in the inclusive range [min_price, max_price]

    Args:
        df: input dataframe
        min_price: The minimum rent price that will be included.
        max_price: The maximum rent price that will be included.
        price_col: The column to filter on. Default is 'price'.

    Returns:
        A new filtered dataframe
    """
    logger.info("Only keeping prices between %s and %s dollars.", min_price, max_price)
    assert pd.api.types.is_numeric_dtype(df[price_col])
    assert isinstance(min_price, float), print(type(min_price))
    assert isinstance(max_price, float), print(type(max_price))
    idx = df[price_col].between(min_price, max_price)
    return df[idx].copy()


def col_to_datetime(df: pd.DataFrame, col_name: str):
    """
    Convert columns with string dates to dtype datetime.

    Args:
        df: input dataframe
        cols: list of column names with string dates to convert

    Returns:
        The dataframe with converted columns
    """
    logger.info("Converting column %s to datetime", col_name)
    logger.info("Example of date format:\n %s", df[col_name].head(1))
    df[col_name] = pd.to_datetime(df[col_name], infer_datetime_format=True)
    return df


def filter_geolocation(df: pd.DataFrame, lon_range: tuple, lat_range:tuple):
    """
    Only keep entries with longitude and latitude in a given range.
    NOTE: assumes the presence of columns called "longitude" and "latitude".

    Args:
        df: input dataframe.
        lon_range: a tuple indicating an inclusive longitude range.
        lat_range: a tuple indicating an inclusive latitude range.

    Returns:
        dataframe filtered on geolocation.
    """
    idx = df['longitude'].between(*lon_range) & df['latitude'].between(*lat_range)
    df = df[idx].copy()
    return df


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Download data artifact (csv).")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Read in input dataframe and check typing
    df = pd.read_csv(artifact_local_path)
    logger.info("Dataframe data types: %s", df.dtypes)

    # Only keep rows with rent in the given price range
    df = remove_outliers(df, args.min_price, args.max_price, price_col=args.price_col)

    # Filter out geolocations outside New York
    lon_range = (-74.25, -73.50)
    lat_range = (40.5, 41.2)
    df = filter_geolocation(df, lon_range, lat_range)

    # Convert string dates to datetime
    df = col_to_datetime(df, col_name='last_review')

    # Write cleaned data to disk
    df.to_csv("clean_sample.csv", index=False)

    # Upload cleaned artifact
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning step")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact with raw data that is to be cleaned.",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact with cleaned data.",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type of the output artifact.",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact.",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum rent price that will be included.",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum rent price that will be included.",
        required=True
    )

    parser.add_argument(
        "--price_col",
        type=str,
        help="The name of the column containing rent prices.",
        required=True
    )

    args = parser.parse_args()

    go(args)
