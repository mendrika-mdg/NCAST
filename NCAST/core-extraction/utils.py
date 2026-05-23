import os
import numpy as np
import pandas as pd
from scipy.ndimage import label
from datetime import datetime, timedelta
from scipy.spatial import cKDTree


class GeoGrid:

    # would be nice to test this on the core data / originally defined via manual section
    def __init__(
        self,
        lats,
        lons,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        fill_value=-999.999
    ):

        # class initialisation
        self.fill_value = fill_value              

        self.lats, self.lons, self.area, self.y0, self.y1, self.x0, self.x1 = \
            self._crop_with_area(
                lats, lons,
                lat_min, lat_max,
                lon_min, lon_max
            )

        # lats and lons should have the same shape
        self.Ny, self.Nx = self.lats.shape

        # pairs of all lat, lon from the georef
        points = np.column_stack([
            self.lats.ravel(),
            self.lons.ravel()
        ])

        # creating the KDTree for easier indexing so we don't have to define it manually, this will return the nearest yx corresponding to latlon
        self.tree = cKDTree(points)


    def _crop_with_area(
        self,
        lats,
        lons,
        lat_min,
        lat_max,
        lon_min,
        lon_max
    ):

        # crop the domain and compute the size of each grid cell before reprojection

        # replace all filled values with nans
        lats_nan = np.where(lats == self.fill_value, np.nan, lats)
        lons_nan = np.where(lons == self.fill_value, np.nan, lons)

        # 
        mask = (
            (lats_nan >= lat_min) &
            (lats_nan <= lat_max) &
            (lons_nan >= lon_min) &
            (lons_nan <= lon_max)
        )

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        y0, y1 = rows[0], rows[-1]
        x0, x1 = cols[0], cols[-1]

        lats_crop = lats[y0:y1+1, x0:x1+1]
        lons_crop = lons[y0:y1+1, x0:x1+1]

        R = 6371.0

        lat_rad = np.deg2rad(lats_crop)
        lon_rad = np.deg2rad(lons_crop)

        dlat = np.gradient(lat_rad, axis=0)
        dlon = np.gradient(lon_rad, axis=1)

        dy = R * dlat
        dx = R * np.cos(lat_rad) * dlon

        area = np.abs(dx * dy)

        return lats_crop, lons_crop, area, y0, y1, x0, x1


    def crop(self, field):
        return field[self.y0:self.y1+1, self.x0:self.x1+1]

    def query(self, lat, lon):
        _, idx = self.tree.query([lat, lon])
        return np.unravel_index(idx, (self.Ny, self.Nx))

    def query_many(self, lat, lon):
        pts = np.column_stack([lat, lon])
        _, idx = self.tree.query(pts)
        return np.unravel_index(idx, (self.Ny, self.Nx))



def generate_time_steps(start, end, step_minutes=15):

    start_dt = datetime(
        start["year"], start["month"], start["day"],
        start["hour"], start["minute"]
    )

    end_dt = datetime(
        end["year"], end["month"], end["day"],
        end["hour"], end["minute"]
    )

    if end_dt < start_dt:
        raise ValueError("End time must be after start time")

    step = timedelta(minutes=step_minutes)

    times = []
    current = start_dt

    while current <= end_dt:
        times.append(dict(
            year=current.year,
            month=current.month,
            day=current.day,
            hour=current.hour,
            minute=current.minute
        ))
        current += step

    return times


def field_to_objects(
    field,
    threshold,
    grid,
    name="field",
    connectivity=8,
    min_area_km2=0.0,
    min_pixels=0
):

    mask = field >= threshold

    if not np.any(mask):
        return pd.DataFrame()

    if connectivity == 8:
        structure = np.ones((3, 3))
    elif connectivity == 4:
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]])
    else:
        raise ValueError("Connectivity must be 4 or 8")

    labeled, n = label(mask, structure=structure)

    rows = []

    for lab in range(1, n + 1):         # this automatically removes background 

        m = labeled == lab              # get all the pixels equal to the label
        n_pix = int(np.sum(m))
    
        if n_pix < min_pixels:
            continue

        weights = grid.area[m]              # get the area of each pixel
        size_km2 = float(np.sum(weights))   # total area of the core

        if size_km2 < min_area_km2:
            continue

        lat_vals = grid.lats[m]
        lon_vals = grid.lons[m]

        lat_centroid = float(np.average(lat_vals, weights=weights))
        lon_centroid = float(np.average(lon_vals, weights=weights))

        lat_min = float(np.min(lat_vals))
        lat_max = float(np.max(lat_vals))
        lon_min = float(np.min(lon_vals))
        lon_max = float(np.max(lon_vals))

        mean_val = float(np.nanmean(field[m]))
        max_val = float(np.nanmax(field[m]))

        if n_pix < 2:
            elongation = 1.0
            eccentricity = 0.0
        else:
            coords = np.column_stack((lat_vals, lon_vals))
            cov = np.cov(coords, rowvar=False, aweights=weights)

            eigvals, _ = np.linalg.eigh(cov)

            major = np.sqrt(max(eigvals.max(), 0.0))
            minor = np.sqrt(max(eigvals.min(), 0.0))

            elongation = float(major / (minor + 1e-12))
            eccentricity = float(
                np.sqrt(max(0.0, 1.0 - (minor**2 / (major**2 + 1e-12))))
            )

        rows.append((
            lab,
            lat_centroid,
            lon_centroid,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            size_km2,
            n_pix,
            mean_val,
            max_val,
            elongation,
            eccentricity
        ))

    df = pd.DataFrame(
        rows,
        columns=[
            "label",
            "lat_centroid",
            "lon_centroid",
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
            "size",
            "n_pix",
            f"mean_{name}",
            f"max_{name}",
            "elongation",
            "eccentricity"
        ]
    )

    if len(df) == 0:
        return df

    return df.sort_values("size", ascending=False).reset_index(drop=True)



def save_dataframe(df, output_dir, t):

    os.makedirs(output_dir, exist_ok=True)

    fname = (
        f"{t['year']:04d}"
        f"{t['month']:02d}"
        f"{t['day']:02d}"
        f"{t['hour']:02d}"
        f"{t['minute']:02d}.csv"
    )

    path = os.path.join(output_dir, fname)

    df.to_csv(path, index=False)


# FSS computation
def compute_fss(pred, obs, window):
    pred = np.clip(pred, 0, 1)
    obs = np.clip(obs, 0, 1)
    f_pred = uniform_filter(pred, size=window, mode="constant")
    f_obs = uniform_filter(obs, size=window, mode="constant")
    num = np.mean((f_pred - f_obs) ** 2)
    den = np.mean(f_pred ** 2 + f_obs ** 2)
    return 1 - num / (den + 1e-8)

