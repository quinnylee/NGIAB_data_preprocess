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
from data_processing.gpkg_utils import head_geom_selection, tail_geom_selection
import json
import os

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

    parser.add_argument(
        "-p",
        "--pairs_dir", 
        type=Path,
        help="path to directory of txt file of pairs in dict form",
        required=True
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="path to directory of raw gridded netcdf files"
    )

    # parser.add_argument(
    #     "--start_date",
    #     "--start",
    #     type=lambda s: datetime.strptime(s, DATE_FORMAT),
    #     help=f"Start date for forcings/realization (format {DATE_FORMAT_HINT})",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--end_date",
    #     "--end",
    #     type=lambda s: datetime.strptime(s, DATE_FORMAT),
    #     help=f"End date for forcings/realization (format {DATE_FORMAT_HINT})",
    #     required=True,
    # )

    parser.add_argument(
        "-D",
        "--debug",
        action="store_true",
        help="enable debug logging",
    )

    return parser.parse_args()

def process_catchment(id, geometries_dict, pair_dir, merged_data):
    catchment_dir = Path(f"{pair_dir}/{id}/")
    if not catchment_dir.exists():
        catchment_dir.mkdir()

    input_file = Path(f"{pair_dir}/{id}/{id}.gpkg")
    output_file = Path(f"{pair_dir}/{id}/{id}-aggregated.nc")

    forcing_working_dir = output_file.parent / (input_file.stem + "-working-dir")
    if not forcing_working_dir.exists():
        forcing_working_dir.mkdir(parents=True, exist_ok=True)

    geom_id = geometries_dict[id]
    gdf_dict = {'ID': [id], 'geometry': [geom_id]}
    gdf_id = gpd.GeoDataFrame(gdf_dict, crs="EPSG:4326")

    temp_dir = forcing_working_dir / "temp"
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
    gdf_id = gdf_id.to_crs(merged_data.crs.esri_pe_string)
    compute_zonal_stats(gdf_id, merged_data, forcing_working_dir)

    shutil.copy(forcing_working_dir / "forcings.nc", output_file)
    logging.info(f"Created forcings file: {output_file}")
    # remove the working directory
    shutil.rmtree(forcing_working_dir)


def process_pair(k, v, geometries_k, geometries_v, merged_data):

    pair_dir = Path(f"../output/{k}-{v}/")
    if not pair_dir.exists():
        pair_dir.mkdir()

    process_catchment(k, geometries_k, pair_dir, merged_data)
    process_catchment(v, geometries_v, pair_dir, merged_data)

    ds_u = xr.open_dataset(Path(f"{pair_dir}/{k}/{k}-aggregated.nc"))
    ds_d = xr.open_dataset(Path(f"{pair_dir}/{v}/{v}-aggregated.nc"))

    name_dict_d = {varname: varname+'_d' for varname in list(ds_d.keys())}
    ds_d = ds_d.rename_vars(name_dict_d)

    ds = xr.merge([ds_u, ds_d], join="inner")
    ds.to_netcdf(Path(f"{pair_dir}/{k}-{v}.nc"))

    shutil.rmtree(Path(f"{pair_dir}/{k}/"))
    shutil.rmtree(Path(f"{pair_dir}/{v}/"))
    
def main() -> None:
    time.sleep(0.01)
    args = parse_arguments()

    setup_logging(args.debug)
    #validate_all()
    pairs_files = os.listdir(args.pairs_dir)
    pairs_list = {pair[:-4]: pair for pair in pairs_files}
    gridded_files = os.listdir(args.output_dir)
    # print(pairs)

    logging.debug("debug works")
    if not Path("../hfv3_conuscats.parquet").exists():
        cat_gdb = gpd.read_file(args.hydrofabric, layer="nwm_catchments_conus")
        cat_gdb.to_parquet("../hfv3_conuscats.parquet", index=False)
    else:
        cat_gdb = gpd.read_parquet("../hfv3_conuscats.parquet")

   
    # start_time = args.start_date.strftime("%Y-%m-%d %H:%M")
    # end_time = args.end_date.strftime("%Y-%m-%d %H:%M")

    for grid_file in gridded_files:
        region_name = grid_file[8:-20]
        pair_file = pairs_list[str(region_name)]

        with open(Path(args.pairs_dir / pair_file), 'r') as f:
            pairs = json.load(f)
        
        geometries_k = {}
        geometries_v = {}
        corrected_list = []

        for k, v in list(pairs.items()):
            if v < 0: # in case we forgot to remove a terminal basin
                continue
            if int(k) not in cat_gdb['ID'].values:
                continue
            if v not in cat_gdb['ID'].values:
                continue
            corrected_list.append([int(k),v])
    
            k_geom = head_geom_selection(k, cat_gdb)
            # logging.debug(k_geom)
            v_geom = tail_geom_selection(k, v, cat_gdb)
            # logging.debug(v_geom)

            geometries_k[int(k)] = k_geom
            geometries_v[v] = v_geom

        if len(geometries_v) == 0:
            os.remove(grid_file)
            continue 

        merged_data = xr.open_dataset(Path(args.output_dir / grid_file))
    # logging.debug(merged_data)
        for [k,v] in corrected_list:
            process_pair(k, v, geometries_k, geometries_v, merged_data)

        os.remove(Path(f"{args.output_dir}/{grid_file}"))

if __name__ == "__main__":
    main()
