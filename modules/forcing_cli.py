from data_sources.source_validation import validate_all
from ngiab_data_cli.custom_logging import setup_logging
from data_processing.forcings import compute_zonal_stats
from data_processing.zarr_utils import get_forcing_data
from data_processing.file_paths import file_paths
import argparse
import logging
import time
import xarray as xr
import geopandas as gpd
from datetime import datetime
from pathlib import Path
import shutil
from data_processing.gpkg_utils import head_gdf_selection, tail_gdf_selection
import json

# Constants
DATE_FORMAT = "%Y-%m-%d"  # used for datetime parsing
DATE_FORMAT_HINT = "YYYY-MM-DD"  # printed in help message

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Subsetting hydrofabrics, forcing generation, and realization creation"
    )

    parser.add_argument(
        "--hf",
        "--hydrofabric",
        type=Path,
        help="path to hydrofabric gpkg",
    )


    # parser.add_argument(
    #     "-u",
    #     "--upstream_input_id",
    #     type=int,
    #     help="upstream input reach id",
    #     required=True,
    # )

    parser.add_argument(
        "-p",
        "--pairs_file", 
        type=Path,
        help="path to txt file of pairs in dict form",
        required=True
    )

    # parser.add_argument(
    #     "-d",
    #     "--downstream_input_id",
    #     type=int,
    #     help="downstream input reach id",
    #     required=True,
    # )

    # parser.add_argument(
    #     "-o",
    #     "--output_file",
    #     type=Path,
    #     help="path to the forcing output file, e.g. /path/to/forcings.nc",
    #     required=True,
    # )
    parser.add_argument(
        "--start_date",
        "--start",
        type=lambda s: datetime.strptime(s, DATE_FORMAT),
        help=f"Start date for forcings/realization (format {DATE_FORMAT_HINT})",
        required=True,
    )
    parser.add_argument(
        "--end_date",
        "--end",
        type=lambda s: datetime.strptime(s, DATE_FORMAT),
        help=f"End date for forcings/realization (format {DATE_FORMAT_HINT})",
        required=True,
    )
    parser.add_argument(
        "-D",
        "--debug",
        action="store_true",
        help="enable debug logging",
    )

    return parser.parse_args()

def process_catchment(id, gdf, args, pair_dir):
    catchment_dir = Path(f"{pair_dir}/{id}/")
    if not catchment_dir.exists():
        catchment_dir.mkdir()

    input_file = Path(f"{pair_dir}/{id}/{id}.gpkg")
    output_file = Path(f"{pair_dir}/{id}/{id}-aggregated.nc")

    start_time = args.start_date.strftime("%Y-%m-%d %H:%M")
    end_time = args.end_date.strftime("%Y-%m-%d %H:%M")

    cached_nc_path = output_file.parent / (input_file.stem + "-raw-gridded-data.nc")
    logging.debug(f"cached nc path: {cached_nc_path}")
    merged_data = get_forcing_data(cached_nc_path, start_time, end_time, gdf)
    logging.debug(merged_data)
    forcing_working_dir = output_file.parent / (input_file.stem + "-working-dir")
    if not forcing_working_dir.exists():
        forcing_working_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = forcing_working_dir / "temp"
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
    gdf = gdf.to_crs(merged_data.crs.esri_pe_string)
    compute_zonal_stats(gdf, merged_data, forcing_working_dir)

    shutil.copy(forcing_working_dir / "forcings.nc", output_file)
    logging.info(f"Created forcings file: {output_file}")
    # remove the working directory
    shutil.rmtree(forcing_working_dir)


def process_pair(k, v, cat_gdb, args):
    k_gdf = head_gdf_selection(k, cat_gdb)
    v_gdf = tail_gdf_selection(k, v, cat_gdb)
    
    k_gdf.set_geometry('geometry', inplace=True)
    logging.debug(f"upstream gdf  bounds: {k_gdf.total_bounds}")
    v_gdf.set_geometry('geometry', inplace=True)
    logging.debug(f"downstream gdf bounds:{v_gdf.bounds}")

    pair_dir = Path(f"../output/{k}-{v}/")
    if not pair_dir.exists():
        pair_dir.mkdir()

    process_catchment(k, k_gdf, args, pair_dir)
    process_catchment(v, v_gdf, args, pair_dir)

    ds_u = xr.open_dataset(Path(f"{pair_dir}/{k}/{k}-aggregated.nc"))
    ds_d = xr.open_dataset(Path(f"{pair_dir}/{v}/{v}-aggregated.nc"))

    name_dict_d = {varname: varname+'_d' for varname in list(ds_d.keys())}
    ds_d = ds_d.rename_vars(name_dict_d)

    ds = xr.merge([ds_u, ds_d], join="inner")
    ds.to_netcdf(Path(f"{pair_dir}/{k}-{v}.nc"))

    
def main() -> None:
    time.sleep(0.01)
    args = parse_arguments()

    setup_logging(args.debug)
    #validate_all()

    logging.debug("debug works")
    if not Path("../hfv3_conuscats.parquet").exists():
        cat_gdb = gpd.read_file(args.hydrofabric, layer="nwm_catchments_conus")
        cat_gdb.to_parquet("../hfv3_conuscats.parquet", index=False)
    else:
        cat_gdb = gpd.read_parquet("../hfv3_conuscats.parquet")

    with open(args.pairs_file, "r") as f:
        pairs = json.load(f)
    
    # gdf = gpd.read_file(args.input_file, layer="divides")
    
    for k, v in list(pairs.items()):
        if v < 0: # in case we forgot to remove a terminal basin
            continue
        if k not in cat_gdb.index:
            continue
        if v not in cat_gdb.index:
            continue
        process_pair(k, v, cat_gdb, args)

if __name__ == "__main__":
    main()
