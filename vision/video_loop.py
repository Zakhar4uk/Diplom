import imageio.v3 as iio

# read a single frame
frame = iio.imread(
    "imageio:cockatoo.mp4",
    index=42,
    plugin="pyav",
)

# bulk read all frames
# Warning: large videos will consume a lot of memory (RAM)
frames = iio.imread("imageio:cockatoo.mp4", plugin="pyav")

# iterate over large videos
for frame in iio.imiter("imageio:cockatoo.mp4", plugin="pyav"):
    print(frame.shape, frame.dtype)