import logging
import os
from pathlib import Path
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from dask.distributed import Client,  progress
import datetime

logger = logging.getLogger(__name__)

# known ngen variable names
# https://github.com/CIROH-UA/ngen/blob/4fb5bb68dc397298bca470dfec94db2c1dcb42fe/include/forcing/AorcForcing.hpp#L77

def validate_dataset_format(dataset: xr.Dataset) -> None:
    """
    Validate the format of the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to be validated.

    Raises
    ------
    ValueError
        If the dataset is not in the correct format.
    """
    if "time" not in dataset.coords:
        raise ValueError("Dataset must have a 'time' coordinate")
    if not np.issubdtype(dataset.time.dtype, np.datetime64):
        raise ValueError("Time coordinate must be a numpy datetime64 type")
    if "x" not in dataset.coords:
        raise ValueError("Dataset must have an 'x' coordinate")
    if "y" not in dataset.coords:
        raise ValueError("Dataset must have a 'y' coordinate")
    if "crs" not in dataset.attrs:
        raise ValueError("Dataset must have a 'crs' attribute")
    if "name" not in dataset.attrs:
        raise ValueError("Dataset must have a name attribute to identify it")

def validate_time_range(dataset: xr.Dataset, start_time: str, end_time: str) -> Tuple[str, str]:
    '''
    Ensure that all selected times are in the passed dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with a time coordinate.
    start_time : str
        Desired start time in YYYY/MM/DD HH:MM:SS format.
    end_time : str
        Desired end time in YYYY/MM/DD HH:MM:SS format.

    Returns
    -------
    str
        start_time, or if not available, earliest available timestep in dataset.
    str
        end_time, or if not available, latest available timestep in dataset.
    '''
    end_time_in_dataset = dataset.time.isel(time=-1).values
    start_time_in_dataset = dataset.time.isel(time=0).values
    if np.datetime64(start_time) < start_time_in_dataset:
        logger.warning(
            f"provided start {start_time} is before the start of the dataset {start_time_in_dataset}, selecting from {start_time_in_dataset}"
        )
        start_time = start_time_in_dataset
    if np.datetime64(end_time) > end_time_in_dataset:
        logger.warning(
            f"provided end {end_time} is after the end of the dataset {end_time_in_dataset}, selecting until {end_time_in_dataset}"
        )
        end_time = end_time_in_dataset
    return start_time, end_time


def clip_dataset_to_bounds(
    dataset: xr.Dataset, bounds: Tuple[float, float, float, float], start_time: str, end_time: str
) -> xr.Dataset:
    """
    Clip the dataset to specified geographical bounds.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to be clipped.
    bounds : tuple[float, float, float, float]
        Corners of bounding box. bounds[0] is x_min, bounds[1] is y_min, 
        bounds[2] is x_max, bounds[3] is y_max.
    start_time : str
        Desired start time in YYYY/MM/DD HH:MM:SS format.
    end_time : str
        Desired end time in YYYY/MM/DD HH:MM:SS format.
    
    Returns
    -------
    xr.Dataset
        Clipped dataset.
    """
    # check time range here in case just this function is imported and not the whole module
    start_time, end_time = validate_time_range(dataset, start_time, end_time)
    dataset = dataset.sel(
        x=slice(bounds[0], bounds[2]),
        y=slice(bounds[1], bounds[3]),
        time=slice(start_time, end_time),
    )
    logger.info("Selected time range and clipped to bounds")
    return dataset


def save_to_cache(stores: xr.Dataset, cached_nc_path: Path) -> xr.Dataset:
    """Compute the store and save it to a cached netCDF file. This is not required but will save time and bandwidth."""
    logger.info("Downloading and caching forcing data, this may take a while")

    if not cached_nc_path.parent.exists():
        cached_nc_path.parent.mkdir(parents=True)

    # sort of terrible work around for half downloaded files
    temp_path = cached_nc_path.with_suffix(".downloading.nc")
    if os.path.exists(temp_path):
        os.remove(temp_path)

    ## Cast every single variable to float32 to save space to save a lot of memory issues later
    ## easier to do it now in this slow download step than later in the steps without dask
    for var in stores.data_vars: 
        stores[var] = stores[var].astype("float32")

    client = Client.current()
    future = client.compute(stores.to_netcdf(temp_path, compute=False))
    # Display progress bar
    progress(future)
    future.result()

    os.rename(temp_path, cached_nc_path)

    data = xr.open_mfdataset(cached_nc_path, parallel=True, engine="h5netcdf")
    return data


def check_local_cache(
    cached_nc_path: Path,
    start_time: str,
    end_time: str,
    gdf: gpd.GeoDataFrame,
    remote_dataset: xr.Dataset
) -> Union[xr.Dataset, None]:

    merged_data = None

    if not os.path.exists(cached_nc_path):
        logger.info("No cache found")
        return

    logger.info("Found cached nc file")
    # open the cached file and check that the time range is correct
    cached_data = xr.open_mfdataset(
        cached_nc_path, parallel=True, engine="h5netcdf"
    )

    if "name" not in cached_data.attrs or "name" not in remote_dataset.attrs:
        logger.warning("No name attribute found to compare datasets")
        return
    if cached_data.name != remote_dataset.name:
        logger.warning("Cached data from different source, .name attr doesn't match")
        return

    range_in_cache = cached_data.time[0].values <= np.datetime64(
        start_time
    ) and cached_data.time[-1].values >= np.datetime64(end_time)

    if not range_in_cache:
        # the cache does not contain the desired time range
        logger.warning("Requested time range not in cache")
        return

    cached_vars = cached_data.data_vars.keys()
    forcing_vars = remote_dataset.data_vars.keys()
    # replace rainrate with precip
    missing_vars = set(forcing_vars) - set(cached_vars)
    if len(missing_vars) > 0:
        logger.warning(f"Missing forcing vars in cache: {missing_vars}")
        return

    if range_in_cache:
        logger.info("Time range is within cached data")
        logger.debug(f"Opened cached nc file: [{cached_nc_path}]")
        merged_data = clip_dataset_to_bounds(
            cached_data, gdf.total_bounds, start_time, end_time
        )
        logger.debug("Clipped stores")        

    return merged_data


def save_and_clip_dataset(
    dataset: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    cache_location: Path,
) -> xr.Dataset:
    """convenience function clip the remote dataset, and either load from cache or save to cache if it's not present"""
    gdf = gdf.to_crs(dataset.crs)

    cached_data = check_local_cache(cache_location, start_time, end_time, gdf, dataset)

    if not cached_data:
        clipped_data = clip_dataset_to_bounds(dataset, gdf.total_bounds, start_time, end_time)
        cached_data = save_to_cache(clipped_data, cache_location)
    return cached_data