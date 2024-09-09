import acquire
import dotenv
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import s3fs
import zarr

logging.getLogger().setLevel(logging.DEBUG)

zarr_s3_endpoint = None
zarr_s3_bucket_name = None
zarr_s3_access_key_id = None
zarr_s3_secret_access_key = None

dataset_root = "hello-hackathon.zarr"


def load_env_vars():
    dotenv.load_dotenv()

    global zarr_s3_endpoint
    global zarr_s3_bucket_name
    global zarr_s3_access_key_id
    global zarr_s3_secret_access_key

    zarr_s3_endpoint = os.environ["ZARR_S3_ENDPOINT"]
    zarr_s3_bucket_name = os.environ["ZARR_S3_BUCKET_NAME"]
    zarr_s3_access_key_id = os.environ["ZARR_S3_ACCESS_KEY_ID"]
    zarr_s3_secret_access_key = os.environ["ZARR_S3_SECRET_ACCESS_KEY"]

    if not zarr_s3_endpoint:
        raise ValueError("ZARR_S3_ENDPOINT is required")
    if not zarr_s3_bucket_name:
        raise ValueError("ZARR_S3_BUCKET_NAME is required")
    if not zarr_s3_access_key_id:
        raise ValueError("ZARR_S3_ACCESS_KEY_ID is required")
    if not zarr_s3_secret_access_key:
        raise ValueError("ZARR_S3_SECRET_ACCESS_KEY is required")


def list_devices():
    runtime = acquire.Runtime()
    dm = runtime.device_manager()
    for device in dm.devices():
        print(device)


def configure_camera(runtime):
    dm = runtime.device_manager()
    props = runtime.get_configuration()
    video = props.video[0]

    video.camera.identifier = dm.select(
        acquire.DeviceKind.Camera, ".*Blackfly.*"
    )

    video.camera.settings.shape = (1920, 1200)
    video.camera.settings.pixel_type = acquire.SampleType.U8
    video.camera.settings.exposure_time_us = 2e7

    runtime.set_configuration(props)


def configure_storage(runtime):
    load_env_vars()

    dm = runtime.device_manager()
    props = runtime.get_configuration()
    video = props.video[0]

    video.storage.identifier = dm.select(
        acquire.DeviceKind.Storage,
        "Zarr",
    )

    # configure storage dimensions
    dimension_x = acquire.StorageDimension(
        name="x", kind="Space", array_size_px=1920, chunk_size_px=960, shard_size_chunks=2,
    )

    dimension_y = acquire.StorageDimension(
        name="y", kind="Space", array_size_px=1200, chunk_size_px=600, shard_size_chunks=2,
    )

    dimension_t = acquire.StorageDimension(
        name="t", kind="Time", array_size_px=0, chunk_size_px=64, shard_size_chunks=1
    )

    video.storage.settings.acquisition_dimensions = [
        dimension_t,
        dimension_y,
        dimension_x,
    ]

    global zarr_s3_endpoint
    global zarr_s3_bucket_name
    global zarr_s3_access_key_id
    global zarr_s3_secret_access_key
    global dataset_root

    video.storage.settings.uri = (
        f"{zarr_s3_endpoint}/{zarr_s3_bucket_name}/{dataset_root}"
    )
    video.storage.settings.s3_access_key_id = zarr_s3_access_key_id
    video.storage.settings.s3_secret_access_key = zarr_s3_secret_access_key

    runtime.set_configuration(props)

def configure_stream(runtime):
    configure_camera(runtime)
    configure_storage(runtime)

    props = runtime.get_configuration()
    video = props.video[0]
    video.max_frame_count = 64


def acquire_to_s3():
    runtime = acquire.Runtime()

    configure_stream(runtime)

    runtime.start()
    runtime.stop()


def load_from_s3_and_display():
    load_env_vars()

    global zarr_s3_endpoint
    global zarr_s3_bucket_name
    global zarr_s3_access_key_id
    global zarr_s3_secret_access_key
    global dataset_root

    s3 = s3fs.S3FileSystem(
        key=zarr_s3_access_key_id,
        secret=zarr_s3_secret_access_key,
        client_kwargs={"endpoint_url": zarr_s3_endpoint},
    )

    s3_path = f"{zarr_s3_bucket_name}/{dataset_root}"
    store = s3fs.S3Map(
        s3_path, s3=s3
    )

    cache = zarr.LRUStoreCache(store, max_size=2 ** 28)
    group = zarr.group(store=cache)

    data = group["0"]

    fig, ax = plt.subplots()

    # Initialize with the first frame
    im = ax.imshow(data[0], animated=True, cmap="gray")

    # Remove axis ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Function to update the image for each frame
    def update(frame):
        im.set_array(data[frame])
        return [im]

    # Create the animation
    anim = FuncAnimation(fig, update, frames=64, interval=50, blit=True)

    # To display the animation (not interactively)
    plt.show(block=True)

    # cleanup
    s3.rm(f"{zarr_s3_bucket_name}/{dataset_root}", recursive=True)


def main():
    acquire_to_s3()
    print("Acquisition complete")
    load_from_s3_and_display()


if __name__ == "__main__":
    main()
