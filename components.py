import datetime
import math
import multiprocessing as mp
from itertools import product
import numpy as np
import os
import re
import requests

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union


from utilities import TIME_MACHINES, CAMERAS, View, decode_video_frames, export_mp4, get_camera_id, get_frame_time


DateFormatter = Callable[[str], datetime.date]
FramePadding = Union[int, tuple[int, int]]
PixelPadding = Union[int, tuple[int, int], tuple[int, int, int, int]]
HuMoments = tuple[float, float, float, float, float, float, float]


class BreatheCam:
    def __init__(self, root_url: str):
        day = re.search(r"\d\d\d\d-\d\d-\d\d", root_url)

        if day is None:
            raise Exception(f"Invalid root url: `{root_url}`.")

        self.day = day[0]
        self.root_url = root_url
        self.tm_url = f"{root_url}/tm.json"
        self.tm = requests.get(self.tm_url).json()

        datasets = self.tm["datasets"]

        assert len(datasets) == 1

        dataset = datasets[0]
        did = dataset["id"]

        self.tile_root_url = f"{root_url}/{did}"
        self.r_url = f"{self.tile_root_url}/r.json"
        self.r = requests.get(self.r_url).json()
        self.levels = self.r["nlevels"]

    @staticmethod
    def init_from(loc: str, day: Union[datetime.date, str], *, formatter: Union[DateFormatter, None] = None):
        if (loc_id := get_camera_id(loc)) is None:
            raise Exception(f"Invalid camera: {loc}.")

        if isinstance(day, str):
            day = (formatter or datetime.date.fromisoformat)(day)

        return BreatheCam(f"{TIME_MACHINES}/{loc_id}/{day.strftime('%Y-%m-%d')}.timemachine")

    @staticmethod
    def download(loc: str,
                 day: datetime.date,
                 time: datetime.time,
                 view: Union[View, None] = None,
                 frames: int = 1,
                 nlevels: int = 1) -> np.ndarray:
        day_str = day.strftime("%Y-%m-%d")
        start_time = f"{day_str} {get_frame_time(time, 3).strftime('%H:%M:%S')}"
        url = f"{TIME_MACHINES}/{CAMERAS[get_camera_id(loc)]}/{day_str}.timemachine"
        cam = BreatheCam(url)
        start_frame = cam.capture_time_to_frame(start_time)

        if start_frame < 0:
            raise Exception("First frame invalid.")

        remaining_frames = len(cam.capture_times) - start_frame

        if remaining_frames < frames:
            frames = remaining_frames

        return cam.download_video(start_frame, frames, view or View.full(), nlevels)

    @property
    def capture_times(self):
        return self.tm["capture-times"]

    @property
    def fps(self) -> int:
        return self.r["fps"]

    @property
    def level_info(self):
        return self.r["level_info"]

    @property
    def tile_height(self) -> int:
        return self.r["video_height"]

    @property
    def tile_width(self) -> int:
        return self.r["video_width"]

    # Coordinates:  The View (rectangle) is in full-resolution coords
    # Internal to this function, the view is modified to match the subsample as the internal
    # coords are divided by subsample
    def download_video(self,
                       start_frame_no: Union[int, datetime.time],
                       nframes: int,
                       view: Union[View, None] = None,
                       nlevels: int = 1) -> np.ndarray:

        if isinstance(start_frame_no, datetime.time):
            start_time = f"{self.day} {get_frame_time(start_frame_no, 3).strftime('%H:%M:%S')}"
            start_frame_no = self.capture_time_to_frame(start_time)

        if start_frame_no < 0 or start_frame_no >= len(self.capture_times):
            raise Exception("First frame invalid.")

        nframes = min(nframes, len(self.capture_times) - start_frame_no)
        view = (view or View.full()).subsample(nlevels)
        level = self.level_from_subsample(nlevels)
        result = np.zeros((nframes, view.height, view.width, 3), dtype=np.uint8)
        th, tw = self.tile_height, self.tile_width
        min_tile_y = view.top // th
        max_tile_y = 1 + (view.bottom - 1) // th
        min_tile_x = view.left // tw
        max_tile_x = 1 + (view.right - 1) // tw

        for tile_y, tile_x in product(range(min_tile_y, max_tile_y), range(min_tile_x, max_tile_x)):
            tile_url = self.tile_url(level, tile_x, tile_y)
            response = requests.head(tile_url)

            if response.status_code == 404:
                print(f"Warning: tile {tile_x},{tile_y} does not exist, skipping...")
                continue

            tile_view = View(tile_x * tw, tile_y * th, (tile_x + 1) * tw, (tile_y + 1) * th)

            intersection = view.intersection(tile_view)

            assert intersection, f"Tile ({tile_x}, {tile_y}) does not intersect view {view}"

            src_view = intersection.translate(-tile_view.left, -tile_view.top)
            dest_view = intersection.translate(-view.left, -view.top)

            try:
                # Download the tile video
                frames = decode_video_frames(tile_url, start_frame_no, nframes)

                # Copy the intersection region to the result array
                result[:, dest_view.top:dest_view.bottom, dest_view.left:dest_view.right, :] = (
                    frames[:, src_view.top:src_view.bottom, src_view.left:src_view.right, :])

            except Exception as e:
                print(f"Error processing tile {tile_url}: {str(e)}")
                continue

        return result

    def download_video_in_parallel(self,
                                   start_frame_no: Union[int, datetime.time],
                                   nframes: int,
                                   view: Union[View, None] = None,
                                   nlevels: int = 1,
                                   cpus: int = 2,
                                   chunk_size: Union[int, None] = None) -> np.ndarray:
        if cpus < 0:
            cpus = os.cpu_count()

        if cpus < 2 or (chunk_size is not None and (chunk_size < 0 or chunk_size >= nframes)):
            return self.download_video(start_frame_no, nframes, view, nlevels)
        
        if isinstance(start_frame_no, datetime.time):
            start_time = f"{self.day} {get_frame_time(start_frame_no, 3).strftime('%H:%M:%S')}"
            start_frame_no = self.capture_time_to_frame(start_time)

        if start_frame_no < 0 or start_frame_no >= len(self.capture_times):
            raise Exception("First frame invalid.")
        
        chunk_size = chunk_size or (nframes // cpus)
        chunks = np.linspace(start_frame_no, start_frame_no + nframes, (nframes // chunk_size) + 1, dtype=int)
        
        with closing(mp.Pool(processes=cpus)) as pool:
            parameters = []

            for i in range(len(chunks) - 1):
                chunk = chunks[i]
                cframes = chunks[i + 1] - chunk
                
                parameters.append((chunk, cframes, view, nlevels))

            video_chunks = pool.starmap_async(self.download_video, parameters)

        pool.join()

        return np.vstack([chunk for chunk in video_chunks.get()])
        
    def capture_time_to_frame(self, date: str) -> int:
        return self.tm["capture-times"].index(date)

    def height(self, nlevels: int = 1) -> int:
        return int(math.ceil(self.r["height"] / nlevels))

    def level_from_subsample(self, nlevels: int) -> int:
        assert ((nlevels & (nlevels - 1)) == 0)

        level = self.levels - nlevels.bit_length()

        assert level >= 0, f"Subsample {nlevels} is too high for timemachine with {self.levels} levels."

        return level

    def subsample_from_level(self, level: int) -> int:
        assert (level > 0) and ((self.levels - level) > 0)

        return 2 ** (self.levels - level - 1)

    def tile_url(self, level: int, tile_x: int, tile_y: int) -> str:
        return f"{self.tile_root_url}/{level}/{tile_y * 4}/{tile_x * 4}.mp4"

    def width(self, nlevels: int = 1) -> int:
        return int(math.ceil(self.r["width"] / nlevels))

    
def _hu(points: np.ndarray) -> HuMoments:
    m00 = len(points)
    m002 = m00 * m00
    m10, m01 = np.sum(points[:, 0]), np.sum(points[:, 1])
    cy, cx = m10 / m00, m01 / m00
    dpy, dpx = points[:, 0] - cy, points[:, 1] - cx
    dpy2, dpx2 = dpy * dpy,  dpx * dpx
    n02 = np.sum(dpx2) / m002
    n03 = np.sum(dpx * dpx2) / m002
    n11 = np.dot(dpy, dpx) / m002
    n12 = np.dot(dpy, dpx2) / m002
    n20 = np.sum(dpy2) / m002
    n21 = np.dot(dpy2, dpx) / m002
    n30 = np.sum(dpy * dpy2) / m002
    t1, t2, t3 = n20 - n02, 4 * n11, n30 - 3 * n12
    t4, t5, t6 = 3 * n21 - n03, n30 + n12, n21 + n03
    t7, t8 = t5 * t5, t6 * t6
    t9, t10 = t7 - 3 * t8, 3 * t7 - t8
    t11, t12 = t5 * t9, t6 * t10
    m1 = n20 + n02
    m2 = t1 * t1 + t2 * t2
    m3 = t3 * t3 + t4 * t4
    m4 = t7 + t8
    m5 = t3 * t11 + t4 * t12
    m6 = t1 * (t5 - t6) * (t5 + t6) + t2 * t5 * t6
    m7 = t4 * t11 - t3 * t12

    return np.array([m1, m2, m3, m4, m5, m6, m7])


class TemporalContour:
    def __init__(self, points: list[tuple[int, int, int]]):
        """Creates a new contour from a list of points

        Parameters
        ----------
        * points - a collection of (frame, row, column) triples
        """
        self.points = np.array(points)
        self.frames = np.sort(np.unique(self.points[:, 0]))
        self.number_of_frames = len(self.frames)
        self.number_of_points = len(points)
        max_y, max_x, min_y, min_x = 0, 0, 2147483647, 2147483647

        for frame in self.frames:
            mask_candidates = self.points[:, 0] == frame
            points = self.points[mask_candidates, 1:]
            y_values, x_values = points[:, 0], points[:, 1]

            max_y = max(max_y, np.max(y_values))
            max_x = max(max_x, np.max(x_values))
            min_y = min(min_y, np.min(y_values))
            min_x = min(min_x, np.min(x_values))

        self.region = View(min_x, min_y, max_x, max_y)
        self.width = self.region.width
        self.height = self.region.height

    def centroids(self, digits=None) -> list[tuple[int, float, float]]:
        centroids = []

        for frame in self.frames:
            mask_candidates = self.points[:, 0] == frame
            points = self.points[mask_candidates, 1:]

            m00 = len(points)
            m10 = np.sum(points[:, 0])
            m01 = np.sum(points[:, 1])

            centroids.append((frame, round(m10 / m00, digits), round(m01 / m00, digits)))

        return centroids
    
    def corners(self):
        """The top left and bottom right corners of the contour"""
        return (
            (self.region.top, self.region.left),
            (self.region.bottom, self.region.right)
        )

    def crop(
            self,
            src: np.ndarray,
            src_nlevels: int,
            contour_nlevels: int,
            pad_frames: FramePadding = 0,
            pad_region: PixelPadding = 0) -> np.ndarray:
        """Crops the part of a video corresponding to the contour

        Parameters
        ----------
            * src - source video
            * src_nlevels - resolution of the source video
            * contour_nlevels - resolution of the video used to generate the contour
            * pad_frames - number of extra frames to add to start and/or end
            * pad_region - number of pixels to add to bounding rectangle dimensions

        Returns
        -------
            An array of frames from `video` trimmed to the region containing the pixels in `contour`.
        """
        assert contour_nlevels >= src_nlevels, "downsampling is not supported"

        nlevels = contour_nlevels / src_nlevels
        region = self.region.upsample(nlevels)
        l, t, r, b = region.left, region.top, region.right, region.bottom
        n, h, w = src.shape[:3]
        frame_start_padding, frame_end_padding = (pad_frames, pad_frames) if isinstance(pad_frames, int) else pad_frames

        if isinstance(pad_region, int):
            pixels_l, pixels_t, pixels_r, pixels_b = pad_region, pad_region, pad_region, pad_region
        elif len(pad_region) == 2:
            pixels_l, pixels_t, pixels_r, pixels_b = pad_region[0], pad_region[1], pad_region[0], pad_region[1]
        else:
            pixels_l, pixels_t, pixels_r, pixels_b = pad_region

        l = max(l - pixels_l, 0)
        t = max(t - pixels_t, 0)
        r = min(r + pixels_r, w - 1)
        b = min(b + pixels_b, h - 1)
        start_frame = max(self.frames[0] - frame_start_padding, 0)
        end_frame = min(self.frames[-1] + frame_end_padding, n - 1)
        frames = [*range(start_frame, self.frames[0]), *self.frames, *range(n, end_frame + 1)]

        return np.array([src[f][t:(b + 1), l:(r + 1), :] for f in frames])

    def density(self, digits: int = 3) -> float:
        """The ratio of points in the contour to total number of points in the region

        Parameters
        ----------
            * digits - number of digits to round to

        Returns
        -------
            * The result of dividing the number of points in the contour by the number of points in the region
              containing the contour
        """
        return round(self.number_of_points / (len(self.frames) * self.region.width * self.region.height), digits)

    def geometric_moment(self, i, j) -> list[tuple[int, float]]:
        moments = []

        for frame in self.frames:
            mask_candidates = self.points[:, 0] == frame
            points = self.points[mask_candidates, 1:]
            moments.append((frame, np.dot(points[:, 0] ** i, points[:, 1] ** j)))

        return moments
    
    def hu_moments(self, scale: bool = True):
        moments = []
        ones = np.ones((7,), dtype=float)
        
        for frame in self.frames:
            points = self.points[:, 0] == frame
            hu = _hu(points)
            
            if scale:
                moments.append([frame, *(-np.copysign(ones, hu) * np.log10(np.fabs(hu)))])
            else:
                moments.append([frame, *hu])
            
        return np.array(moments, dtype=float)

    def mask(self, video: np.ndarray, nlevels: int = 1) -> np.ndarray:
        masks = np.zeros_like(video, dtype=np.uint8)

        for f, r, c in self.points:
            masks[f, r * nlevels, c * nlevels] = 255

        return masks

    def mask_from(self, video: np.ndarray, nlevels: int = 1):
        masks = np.zeros_like(video, dtype=np.uint8)

        for f, r, c in self.points:
            masks[f, r * nlevels, c * nlevels] = video[f, r * nlevels, c * nlevels]

        return masks

    def metadata(self):
        l, t, r, b = self.region.bounding_box()

        TemporalContourMetadata(
            start_frame=self.frames[0],
            nframes=self.number_of_frames,
            npoints=self.number_of_points,
            bounding_box={
                "left": l,
                "top": t,
                "right": r,
                "bottom": b,
                "top_left": [t, l],
                "bottom_right": [b, r],
                "width": self.region.width,
                "height": self.region.height
            },
            points=list(map(tuple, list(self.points)))
        )

    def pixels_by_frame(self) -> dict[int, np.ndarray]:
        return { f: self.points[self.points[:, 0] == f, 1:] for f in self.frames}

    def save(
            self,
            path: str,
            src: np.ndarray,
            src_nlevels: int,
            contour_nlevels: int,
            pad_frames: FramePadding = 0,
            pad_region: PixelPadding = 0
    ) -> np.ndarray:
        path = Path(path)

        if Path.is_dir(path):
            raise Exception("expected file name")
        
        Path(os.dirname(path)).mkdir(parents=True, exist_ok=True)

        cropped = self.crop(src, src_nlevels, contour_nlevels, pad_frames, pad_region)
        
        export_mp4(path, cropped)
        
    def __len__(self):
        """Total number of points in the contour"""
        return len(self.points)


@dataclass
class TemporalContourMetadata:
    bounding_box: dict[str]
    start_frame: int
    nframes: int
    npoints: int
    points: list[tuple[int, int, int]]
