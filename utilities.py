import cv2
import datetime
import ffmpeg
import json
import matplotlib.pyplot as plt
import numpy as np
import psutil
import subprocess
import sys
import time


from scipy.signal.windows import gaussian
from typing import Generic, TypeVar, Union


TIME_MACHINES = "https://tiles.cmucreatelab.org/ecam/timemachines"

CAMERAS = {
    "Clairton Coke Works": "clairton4",
    "Shell Plastics West": "vanport3",
    "Edgar Thomson South": "westmifflin2",
    "Metalico": "accan2",
    "Revolution ETC/Harmon Creek Gas Processing Plants": "cryocam",
    "Riverside Concrete": "cementcam",
    "Shell Plastics East": "center1",
    "Irvin": "irvin1",
    "North Shore": "heinz",
    "Mon. Valley": "walnuttowers1",
    "Downtown": "trimont1",
    "Oakland": "oakland"
}


def decode_video_frames(video_url, start_frame=None, n_frames=None, start_time=None, end_time=None, get_metadata: bool = False):
    """Downloads a video using ffmpeg
    
    Parameters
    """
    # Input validation
    if (start_frame is not None) ^ ((n_frames is not None) or (end_time is not None)):
        raise ValueError("Both start_frame and n_frames must be provided together")

    if start_frame is not None and start_time is not None:
        raise ValueError("Cannot specify both frame numbers and timestamps")

    # Get video information using ffprobe
    probe_cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        '-select_streams', 'v:0',
        video_url
    ]

    try:
        probe_output, probe_error = subprocess.Popen(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        ).communicate()
        metadata = json.loads(probe_output)

        if not metadata.get('streams'):
            raise ValueError("No streams found in video file")

        # Get the first video stream
        video_stream = metadata['streams'][0]

        # Extract video properties
        try:
            width = int(video_stream['width'])
            height = int(video_stream['height'])

            # Parse frame rate which might be in different formats
            if 'r_frame_rate' in video_stream:
                num, den = map(int, video_stream['r_frame_rate'].split('/'))
                fps = num / den
            elif 'avg_frame_rate' in video_stream:
                num, den = map(int, video_stream['avg_frame_rate'].split('/'))
                fps = num / den
            else:
                raise KeyError("Could not find frame rate information")

        except KeyError as e:
            raise KeyError(f"Missing required video property: {str(e)}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe error: {e.stderr.decode()}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse FFprobe output: {str(e)}")

    # Calculate duration based on input parameters
    if start_frame is not None and n_frames is not None:
        start_time = start_frame / fps
        duration = n_frames / fps
        expected_frames = n_frames
    elif start_time is not None and end_time is not None:
        duration = end_time - start_time
        expected_frames = int(duration * fps)
    else:
        raise ValueError("Either frame numbers or timestamps must be provided")

    # Build ffmpeg command
    cmd = ['ffmpeg', '-ss', str(start_time), '-t', str(duration)]

    # Add video url
    cmd.extend(['-i', video_url])

    # Add output format settings
    cmd.extend([
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-'
    ])

    # Run ffmpeg process with communicate()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 ** 8  # Use large buffer size for video data
        )
        raw_data, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")

    # Verify the output size
    expected_bytes = width * height * 3 * expected_frames
    actual_bytes = len(raw_data)

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"FFmpeg output size mismatch: expected {expected_bytes} bytes "
            f"({expected_frames} frames) but got {actual_bytes} bytes "
            f"({actual_bytes // (width * height * 3)} frames)"
        )

    # Reshape into frames
    frames = np.frombuffer(raw_data, dtype=np.uint8)
    frames = frames.reshape((expected_frames, height, width, 3))

    return frames, metadata if get_metadata else frames


def backtrack(tm: datetime.time, seconds: int) -> datetime.time:
    """Steps a time instance back a number of seconds
    
    Parameters
    ----------
    tm - the time
    seconds - the number of seconds to step back
    """
    if tm.second < seconds:
        second = 60 + tm.second - seconds

        if tm.minute == 0:
            if tm.hour == 0:
                raise Exception(f"Invalid start frame: {tm.strftime('%H:%M:%S')}")
            else:
                tm = tm.replace(hour=tm.hour - 1, minute=59, second=second)
        else:
            tm = tm.replace(minute=tm.minute - 1, second=second)
    else:
        tm = tm.replace(second=tm.second - seconds)

    return tm


def bgr2mixed(ibgr: np.ndarray, sr: float = 0.2126, sg: float = 0.7152, sb: float = 0.0722) -> np.ndarray:
    """Mixes the channels of a BGR image into a single value.

    The default values correspond to BT.709 recommendations for converting from red/green/blue to
    grayscale.

    Parameters
    ----------
    ibgr - an image in BGR format
    sr - red scale
    sg - green scale
    sb - blue scale

    Returns
    -------
    A 2D array where each element is the combined value of its BGR channels in the original
    """
    return ibgr[:, :, 2] * sr + ibgr[:, :, 1] * sg + ibgr[:, :, 0] * sb


def coords2D_in_radius(radius: float, point: tuple[int, int], matrix: np.ndarray):
    """Draws a circle around a point in a matrix and returns all indices within it.

    Parameters
    ----------
    * radius - radius of the circle
    * point - row and column of center point
    * matrix - 2D array

    Returns
    -------
    A list of indices within the radius of the circle surrounding the given point.
    This includes any points with a distance less than or equal to the radius. When
    the radius is a whole number, the indices correspond to all pixels that
    are `radius` steps up, down, left, and right from the pixel.
    """

    r, c = point
    rows, columns = np.indices(matrix.shape).reshape(2, -1)
    distance = np.sqrt((columns - c) ** 2 + (rows - r) ** 2)

    return np.array(np.where((distance <= radius).reshape(matrix.shape))).T


def coords3D_in_radius(radius: int, depth: int, point: tuple[int, int, int], matrix: np.ndarray):
    """Draws a circle around a point in a matrix and returns all indices within it.

    Parameters
    ----------
    * radius - radius of the circle
    * point - frame, row, and column of center point
    * matrix - 2D array

    Returns
    -------
    A list of indices within the radius of the circle surrounding the given point.
    This includes any points with a distance less than or equal to the radius. When
    the radius is a whole number, the indices correspond to all pixels that
    are `radius` steps up, down, left, and right from the pixel.
    """

    f, r, c = point
    frames, rows, columns = np.indices(matrix.shape[:3]).reshape(3, -1)
    distance = (columns - c) ** 2 + (rows - r) ** 2 + (frames - f) ** 2
    nbrs = np.array(np.where((distance <= (radius * radius)).reshape(matrix.shape[:3]))).T
    nbrs = nbrs[nbrs[:, 0] == f]
    extras = np.array([(z, y, x) for z in range(max(0, f - depth), f) for _, y, x in nbrs])

    return np.concatenate([nbrs, extras], axis=0) if extras.size > 0 else nbrs


def export_mp4(path: str, video: np.ndarray):
    h, w = video.shape[1], video.shape[2]

    process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{w}x{h}")
        .filter("pad", width="ceil(iw/2)*2", height="ceil(ih/2)*2", color="black")
        .output(path, pix_fmt="yuv420p", vcodec="libx264", r=12, loglevel="quiet")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in video:
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()


def gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    window = gaussian(kernel_size, std=sigma).reshape(kernel_size, 1)

    return np.einsum("ai,aj,ak->ijk", window, window, window)


def get_camera_id(name: str):
    if name in CAMERAS:
        return CAMERAS[name]

    for cam in CAMERAS.values():
        if cam == name:
            return cam

    return None


def get_frame_time(t: datetime.time, fps: int) -> datetime.time:
    extra = t.second % fps

    return t if extra == 0 else backtrack(t, extra)


def get_previous_frame_time(t: datetime.time, fps: int, nframes: int = 1) -> datetime.time:
    extra = t.second % fps

    return backtrack(t, fps * nframes if extra == 0 else (fps - 1) * nframes + extra)


def rgb2mixed(irgb: np.ndarray, sr: float = 0.2126, sg: float = 0.7152, sb: float = 0.0722) -> np.ndarray:
    """Mixes the channels of an RGB image into a single value.

    The default values correspond to BT.709 recommendations for converting from red/green/blue to
    grayscale.

    Parameters
    ----------
    irgb - an image in RGB format
    sr - red scale
    sg - green scale
    sb - blue scale

    Returns
    -------
    A 2D array where each element is the combined value of its BGR channels in the original
    """
    return irgb[:, :, 0] * sr + irgb[:, :, 1] * sg + irgb[:, :, 2] * sb


def rgb2rgba(img: np.ndarray, alpha):
    """Adds an alpha channel to an RGB image

    Parameters
    ----------
    * img - an array with shape (height x width x 3)
    * alpha - the value of the alpha channel

    Returns
    -------
    An array with shape (height x width x 4)
    """
    return np.concatenate(
        [img, np.array([[[alpha]] * img.shape[1]] * img.shape[0])], 
        axis=2
    )


T = TypeVar('T')

class DisjointSet(Generic[T]):
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, value: T):
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0
        elif self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])

    def find(self, value: T):
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

            return value

        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])

        return self.parent[value]

    def union(self, x: T, y: T):
        px, py = self.find(x), self.find(y)

        if px == py:
            return

        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
        elif self.rank[px] < self.rank[py]:
            self.parent[px] = py
        else:
            self.parent[py] = px
            self.rank[px] += 1


class BrownsDoubleExponentialSmoother:
    """Implements double exponential smoothing.
    
    Smoothing works by updating two components, `level` and `trend`,
    using smoothing factors `alpha` and `gamma`.

    `alpha` controls smoothing of the `level` component, the higher the value
    the more weight that is given to recent data. `gamma` controls smoothing
    of the `trend` component.
    """

    def __init__(self, level: np.ndarray, trend: np.ndarray, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.delta_a = 1 - alpha
        self.gamma = self.alpha / self.delta_a
        self._level = level
        self._trend = trend
        self._slevel = level
        self._strend = trend

    @property
    def level(self) -> np.ndarray:
        return self._level
    
    @property
    def trend(self) -> np.ndarray:
        return self._trend

    def update(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert data.shape == self._level.shape, "shape of data does not match shape of component"
        assert data.shape == self._trend.shape, "shape of data does not match shape of component"

        next_level = self.alpha * data + self.delta_a * self._slevel
        next_trend = self.alpha * next_level + self.delta_a * self._strend

        self._slevel = next_level
        self._strend = next_trend
        self._level = 2 * self._slevel - self._strend
        self._trend = self.gamma * (self._slevel - self._strend)

        return self._level, self._trend


class DoubleExponentialSmoother:
    """Implements double exponential smoothing.
    
    Smoothing works by updating two components, `level` and `trend`,
    using smoothing factors `alpha` and `gamma`.

    `alpha` controls smoothing of the `level` component, the higher the value
    the more weight that is given to recent data. `gamma` controls smoothing
    of the `trend` component.
    """

    def __init__(self, level: np.ndarray, trend: np.ndarray, alpha: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.delta_a = 1 - alpha
        self.delta_g = 1 - gamma
        self._level = level
        self._trend = trend

    @property
    def level(self) -> np.ndarray:
        return self._level
    
    @property
    def trend(self) -> np.ndarray:
        return self._trend

    def update(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert data.shape == self._level.shape, "shape of data does not match shape of component"
        assert data.shape == self._trend.shape, "shape of data does not match shape of component"

        next_level = self.alpha * data + self.delta_a * (self._level - self._trend)
        next_trend = self.gamma * (next_level - self._level) + self.delta_g * self._trend

        self._level = next_level
        self._trend = next_trend

        return next_level, next_trend


class ExponentialSmoother:
    """Implements simple exponential smoothing.
    
    Smoothing works by a `level` component using a smoothing factor, `alpha`.

    `alpha` controls smoothing of the `level` component, the higher the value
    the more weight that is given to recent data.
    """
    def __init__(self, level: np.ndarray, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.delta = 1 - alpha
        self._level = level

    @property
    def level(self) -> np.ndarray:
        return self._level

    def update(self, data: np.ndarray) -> np.ndarray:
        assert data.shape == self._level.shape, "shape of data does not match shape of component"
        
        self._level = self.alpha * data + self.delta * self._level

        return self._level




class Stopwatch:
    def __init__(self, name, print_stats=True):
        self.name = name
        self.stats_msg = None
        self.print_stats = print_stats

    def __enter__(self):
        self.start_wall_time = time.time()
        self.start_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
        self.start_cpu_count = psutil.cpu_count()
        return self

    def set_stats_msg(self, stats_msg):
        self.stats_msg = stats_msg
    
    def start(self):
        self.start_wall_time = time.time()
        self.start_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
        self.start_cpu_count = psutil.cpu_count()
        
    def wall_elapsed(self):
        return time.time() - self.start_wall_time

    def cpu_elapsed(self):
        end_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
        return end_cpu_time - self.start_cpu_time

    def __exit__(self, type, value, traceback):
        msg =  self.stats_msg = f'{self.name}: {self.wall_elapsed():.1f} seconds, {self.cpu_elapsed():.1f} seconds CPU' 
        if self.stats_msg is not None:
            msg += f', {self.stats_msg}'

        if self.print_stats:
            sys.stdout.write('%s: %s\n' % (self.name, self.stats_msg))
            sys.stdout.flush()


class View:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def intersection(self, other) -> Union["View", None]:
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        return View(left, top, right, bottom) if left < right and top < bottom else None

    def translate(self, dx, dy) -> "View":
        return View(self.left + dx, self.top + dy, self.right + dx, self.bottom + dy)

    @staticmethod
    def full():
        return View(0, 0, 7930, 2808)

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @staticmethod
    def parse(pts: str) -> "View":
        # Example: "4654,2127,4915,2322,pts"
        tokens = pts.rstrip(",pts").split(",")

        assert len(tokens) == 4, "expected string with format 'left,top,right,bottom[,pts]"

        return View(*map(float, tokens[:4]))

    def bounding_box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom
    
    def center(self):
        return (self.right - self.left) // 2, (self.bottom - self.top) // 2

    def corners(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.left, self.top), (self.right, self.bottom)

    def subsample(self, nlevels) -> "View":
        assert (nlevels & (nlevels - 1)) == 0, "Expected level to be power of 2."

        return View(
            round(self.left / nlevels),
            round(self.top / nlevels),
            round(self.right / nlevels),
            round(self.bottom / nlevels)
        )

    def to_pts(self):
        return f"{','.join(map(str, [int(num) if num.is_integer() else num for num in self.corners()]))},pts"

    def upsample(self, nlevels) -> "View":
        assert (nlevels & (nlevels - 1)) == 0, "Expected level to be power of 2."

        return View(
            round(self.left * nlevels),
            round(self.top * nlevels),
            round(self.right * nlevels),
            round(self.bottom * nlevels)
        )

    def __repr__(self):
        return f"{View.__name__}(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"