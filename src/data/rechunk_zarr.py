# takes the path to the dataset and new chunking partitions and overrides the original dataset with a new version containing the 
#   new chunking pattern

# Example usage:
#   python rechunk_zarr path/to/.zarr time latitude longitude
# If any of the variables are set to -1, the full size of the dimention will be used

import sys
import shutil
import os
import xarray as xr

def chunk_zarr(data_path, time_chunk, lat_chunk, long_chunk):
    temp_path = data_path.rstrip("/") + "_temp.zarr"

    z_file = xr.open_dataset(data_path, engine="zarr", chunks={})
    z_file = z_file.chunk({
        "time": time_chunk,
        "latitude": lat_chunk,
        "longitude": long_chunk
    })
    z_file.unify_chunks()
    z_file.to_zarr(
        temp_path,
        mode="w",
        align_chunks=True
    )

    # Only replace the original once the write has succeeded
    shutil.rmtree(data_path)
    os.rename(temp_path, data_path)


if __name__ == "__main__":
    data_path, time_chunk, lat_chunk, long_chunk = sys.argv[1:]
    chunk_zarr(data_path, int(time_chunk), int(lat_chunk), int(long_chunk))
