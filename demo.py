import acquire
import dotenv
import logging
import os
import s3fs
import zarr

logging.getLogger().setLevel(logging.DEBUG)

dotenv.load_dotenv()

def list_devices():
    runtime = acquire.Runtime()
    dm = runtime.device_manager()
    for device in dm.devices():
        print(device)

def get_device_capabilities():
    runtime = acquire.Runtime()
    dm = runtime.device_manager()

    props = runtime.get_configuration()
    props.video[0].camera.identifier = dm.select(
        acquire.DeviceKind.Camera, ".*Blackfly.*"
    )
    props.video[0].camera.settings.shape = (1, 1)

    props = runtime.set_configuration(props)
    caps = runtime.get_capabilities()

    camera = caps.video[0].camera
    print(camera.dict())


def acquire_to_s3():
    dataset_root = "hello-hackathon.zarr"

    runtime = acquire.Runtime()

    required_env_vars = [
        "ZARR_S3_ENDPOINT",
        "ZARR_S3_BUCKET_NAME",
        "ZARR_S3_ACCESS_KEY_ID",
        "ZARR_S3_SECRET_ACCESS_KEY",
    ]

    for var in required_env_vars:
        if var not in os.environ:
            raise ValueError(f"Environment variable {var} is required")

    zarr_s3_endpoint = os.environ["ZARR_S3_ENDPOINT"]
    zarr_s3_bucket_name = os.environ["ZARR_S3_BUCKET_NAME"]
    zarr_s3_access_key_id = os.environ["ZARR_S3_ACCESS_KEY_ID"]
    zarr_s3_secret_access_key = os.environ["ZARR_S3_SECRET_ACCESS_KEY"]

    dm = runtime.device_manager()
    props = runtime.get_configuration()
    video = props.video[0]

    video.camera.identifier = dm.select(
        acquire.DeviceKind.Camera, ".*Blackfly.*"
    )
    video.camera.settings.shape = (1920, 1200)
    video.camera.settings.pixel_type = acquire.SampleType.U8

    video.storage.identifier = dm.select(
        acquire.DeviceKind.Storage,
        "Zarr",
    )
    video.storage.settings.uri = (
        f"{zarr_s3_endpoint}/{zarr_s3_bucket_name}/{dataset_root}"
    )
    video.storage.settings.s3_access_key_id = zarr_s3_access_key_id
    video.storage.settings.s3_secret_access_key = zarr_s3_secret_access_key

    video.max_frame_count = 64

    # configure storage dimensions
    dimension_x = acquire.StorageDimension(
        name="x", kind="Space", array_size_px=1920, chunk_size_px=960, shard_size_chunks = 2,
    )

    dimension_y = acquire.StorageDimension(
        name="y", kind="Space", array_size_px=1200, chunk_size_px=600, shard_size_chunks = 2,
    )

    dimension_t = acquire.StorageDimension(
        name="t", kind="Time", array_size_px=0, chunk_size_px=64, shard_size_chunks = 1
    )

    video.storage.settings.acquisition_dimensions = [
        dimension_t,
        dimension_y,
        dimension_x,
    ]

    runtime.set_configuration(props)

    runtime.start()
    runtime.stop()

    s3 = s3fs.S3FileSystem(
        key=zarr_s3_access_key_id,
        secret=zarr_s3_secret_access_key,
        client_kwargs={"endpoint_url": zarr_s3_endpoint},
    )
    store = s3fs.S3Map(
        root=f"{zarr_s3_bucket_name}/{dataset_root}", s3=s3
    )
    cache = zarr.LRUStoreCache(store, max_size=2**28)
    group = zarr.group(store=cache)

    data = group["0"]

    assert data.chunks == (64, 600, 960)
    assert data.shape == (
        64,
        video.camera.settings.shape[1],
        video.camera.settings.shape[0],
    )
    assert data.nchunks == 4

    # cleanup
    s3.rm(f"{zarr_s3_bucket_name}/{dataset_root}", recursive=True)

def main():
    acquire_to_s3()

if __name__ == "__main__":
    main()