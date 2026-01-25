import os
import functools
import argparse
import sys

import pandas as pd
import zipfile
import tqdm
from collections import defaultdict
import warnings

from multiprocessing import Pool
from tqdm import tqdm


def get_closest_maturity_per_root(ssf_instruments):
    ssf_instruments['ROOT'] = ssf_instruments['INSTRUMENT_SERIES'].str[:-4]
    ssf_instruments['MATURITY'] = pd.to_numeric(ssf_instruments['INSTRUMENT_SERIES'].str[-4:], errors='coerce')
    ssf_instruments = ssf_instruments.dropna(subset=['MATURITY'])
    mat_map = dict()
    for k in ssf_instruments['MATURITY'].unique():
        mat_map[k] = f"{k:04}"
    ssf_instruments['MATURITY'] = ssf_instruments['MATURITY'].map(mat_map)
    return ssf_instruments.loc[ssf_instruments.groupby('ROOT')['MATURITY'].idxmin()]


def get_highest_num_traders_per_root(ssf_instruments):
    ssf_instruments['ROOT'] = ssf_instruments['INSTRUMENT_SERIES'].str[:-4]
    ssf_instruments['MATURITY'] = ssf_instruments['INSTRUMENT_SERIES'].str[-4:]
    #pd.to_numeric(ssf_instruments['INSTRUMENT_SERIES'].str[-4:], errors='coerce')
    # ssf_instruments = ssf_instruments.dropna(subset=['MATURITY'])
    value_counts = ssf_instruments.groupby(['ROOT'])['MATURITY'].value_counts()
    max_value_counts = value_counts.groupby(level=(0,)).idxmax()
    return ssf_instruments.set_index(['ROOT', 'MATURITY']).loc[max_value_counts].reset_index()


def get_buy_sell_factor():
    buy_sell_map = defaultdict(lambda: 1.0)
    buy_sell_map['S'] = -1.0
    return buy_sell_map


def process_zip_file(zip_file_path,
                     num_stock_futures,
                     num_index_futures,
                     to_parquet
                     ):
    if zip_file_path is None:
        return None, None

    output_filename = zip_file_path.replace(".zip",
                                            f"_output_maxVol_F{str(num_stock_futures)}_F{str(num_index_futures)}" +
                                            f".{'parquet' if to_parquet else 'csv'}")

    # Step 2: Open the zip file and read the CSV inside
    with (zipfile.ZipFile(zip_file_path, 'r') as zip_file):
        # Assuming there's only one CSV file inside the zip
        csv_file_name = zip_file.namelist()[0]

        # Read the CSV file, skip the first row, and use the second row as headers
        df = pd.read_csv(zip_file.open(csv_file_name), delimiter=';', skiprows=1, low_memory=False)

        # Normalize column names
        df.columns = [c.upper().replace(' ', '_') for c in df.columns]

        # Check if 'DATE' column is present
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y %H:%M:%S')
            df['EPOCH'] = (df['DATE'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            df['DATE_ONLY'] = df['DATE'].dt.date  # Renaming to avoid confusion
            df['TIME_ONLY'] = df['DATE'].dt.time  # Renaming to avoid confusion
        else:
            print("ERROR: 'DATE' column not found")

        # df['MARKET_SEGMENT'] you can look SSF: single stock futures,
        # SSO: Sİngle stock options,
        # INF: Index Futures, starts with PM precious metals,
        # df['MARKET_SEGMENT'].value_counts()/df.shape[0]
        df = df[df['MARKET_SEGMENT'].isin(['INF', 'SSF'])]

        # Filter the DataFrame based on instrument series
        ssf_instruments = df.query('MARKET_SEGMENT=="SSF"')['INSTRUMENT_SERIES'].value_counts().nlargest(
            num_stock_futures).reset_index()
        inf_instruments = df.query('MARKET_SEGMENT=="INF"')['INSTRUMENT_SERIES'].value_counts().nlargest(
            num_index_futures).reset_index()

        ssf_instruments = get_highest_num_traders_per_root(ssf_instruments)
        inf_instruments = get_highest_num_traders_per_root(inf_instruments)

        # ['F_AKBNK0824', 'F_YKBNK0824', 'F_GARAN0824']
        df = df[df['INSTRUMENT_SERIES'].isin(
            ssf_instruments['INSTRUMENT_SERIES'].tolist() + inf_instruments['INSTRUMENT_SERIES'].tolist())]
        df = df.dropna()
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            df.loc['QUANTITY'] = df['BUY_SELL'].map(get_buy_sell_factor()) * df['QUANTITY']
            if 'DATE' in df.columns:
                df.loc['datetime_second'] = df['DATE'].dt.floor('s')

        # Group by the instrument and datetime_second column and then aggregate
        # df_max = df.groupby(['INSTRUMENT_SERIES', 'datetime_second']).agg({
        #   'PRICE': 'max',
        #   'EXECUTION': lambda x: max(x.abs())  # Applying absolute value before finding the max
        # }).reset_index()

        if not to_parquet:
            df.to_csv(output_filename, index=False)
        else:
            df.to_parquet(output_filename, index=False)

    return output_filename, os.path.isfile(output_filename)


def process_zip_file_list(full_path_file_list,
                          to_parquet,
                          num_stock_futures=30,
                          num_index_futures=4,
                          num_process=4):
    with Pool(processes=num_process) as pool:
        process_function = functools.partial(process_zip_file,
                                             num_stock_futures=num_stock_futures,
                                             num_index_futures=num_index_futures,
                                             to_parquet=to_parquet)
        return pool.starmap(process_function, [(x,) for x in full_path_file_list])


def main(read_write_dir: str,
         serial=False,
         to_parquet=False,
         num_stock_futures=30,
         num_index_futures=4,
         num_process=4
         ):

    zips_to_read = [os.path.abspath(f"{read_write_dir}/{x}") for x in os.listdir(read_write_dir) if x.endswith('.zip')]
    if serial:
        for zip_file_path in (pb := tqdm(zips_to_read)):
            pb.set_description(f"Processing {zip_file_path}")
            process_zip_file(zip_file_path,
                             num_stock_futures=num_stock_futures,
                             num_index_futures=num_index_futures,
                             to_parquet=to_parquet)
    else:
        list_of_results = process_zip_file_list(zips_to_read,
                                                to_parquet,
                                                num_stock_futures=num_stock_futures,
                                                num_index_futures=num_index_futures,
                                                num_process=num_process
                                                )
        for output_filename, was_success in list_of_results:
            if was_success:
                print(f"Saved {output_filename}")
            else:
                print(f"FAIL {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script taking in a directory with zip files that contain market "
                                                 "data")

    # Add arguments
    parser.add_argument("dir", type=str, help="the directory with zip files in")
    parser.add_argument("--num_stock_futures", type=int, default=30, nargs="?",
                        help="The number of single stock futures (defaults to 30)")
    parser.add_argument("--num_index_futures", type=int, default=4, nargs="?",
                        help="The number of index futures (defaults to 4)")
    parser.add_argument("--num_process", type=int, default=4, nargs="?",
                        help="The number of processes to use when running in parallel (defaults to 4)")
    parser.add_argument("--parallel", action="store_true",
                        help="whether the files should be processes in parallel (defaults to False)")
    parser.add_argument("--to_parquet", action="store_true",
                        help="whether the files should be written to parquet format (defaults to False)")

    args = parser.parse_args()

    READ_WRITE_DIR = args.dir  # './20240815_2024-08-16_1608'
    main(READ_WRITE_DIR,
         serial=not args.parallel,
         to_parquet=args.to_parquet,
         num_stock_futures=args.num_stock_futures,
         num_index_futures=args.num_index_futures,
         num_process=args.num_process)
