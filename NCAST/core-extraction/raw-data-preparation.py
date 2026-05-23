import os
import sys
import torch
import numpy as np
from glob import glob
from netCDF4 import Dataset
from scipy.ndimage import label
from datetime import datetime, timedelta

sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics

# DOMAIN SETTINGS
y_min, y_max = 1403, 1914
x_min, x_max = 66, 577

geodata = np.load("/gws/ssde/j25b/cocoon/SSA_domain/lat_lon_2268_2080.npz")
lons = geodata["lon"][y_min:y_max+1, x_min:x_max+1]
lats = geodata["lat"][y_min:y_max+1, x_min:x_max+1]

# just takes the min lat we can see in the domain
CONTEXT_LAT_MIN = float(np.min(lats))
CONTEXT_LAT_MAX = float(np.max(lats))
CONTEXT_LON_MIN = float(np.min(lons))
CONTEXT_LON_MAX = float(np.max(lons))

# BASIC HELPERS
def prepare_core(file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"The file '{file}' does not exist.")
    try:
        with Dataset(file, "r") as data:
            cores = data.variables["cores"][0, y_min:y_max+1, x_min:x_max+1]
    except OSError as e:
        raise OSError(f"Error opening NetCDF file: {file}. {e}")
    return cores

def update_hour(date_dict, hours_to_add, minutes_to_add):
    # Add hours and minutes to a datetime dictionary and return updated dict and path
    time_obj = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"])
    )

    updated = time_obj + timedelta(hours=hours_to_add, minutes=minutes_to_add)

    new_date_dict = {
        "year":   f"{updated.year:04d}",
        "month":  f"{updated.month:02d}",
        "day":    f"{updated.day:02d}",
        "hour":   f"{updated.hour:02d}",
        "minute": f"{updated.minute:02d}"
    }

    file_path = (
        f"{new_date_dict['year']}/{new_date_dict['month']}/"
        f"{new_date_dict['year']}{new_date_dict['month']}"
        f"{new_date_dict['day']}{new_date_dict['hour']}{new_date_dict['minute']}.nc"
    )

    return {'time': new_date_dict, 'path': file_path}


def extract_box(matrix, y, x, box_size=3):
    half = box_size // 2
    y_min_box = max(y - half, 0)
    y_max_box = min(y + half + 1, matrix.shape[0])
    x_min_box = max(x - half, 0)
    x_max_box = min(x + half + 1, matrix.shape[1])
    return matrix[y_min_box:y_max_box, x_min_box:x_max_box]


def create_storm_database(data_t, lats, lons):
    # Identify storm cores and extract features for each core

    cores_t = data_t["cores"][0, y_min:y_max+1, x_min:x_max+1]

    if not np.any(cores_t):
        return {}

    tir_t = data_t["tir"][0, y_min:y_max+1, x_min:x_max+1]

    Pmax_lat, Pmax_lon = data_t["max_lat"][:], data_t["max_lon"][:]

    valid = (
        (Pmax_lon >= CONTEXT_LON_MIN) & (Pmax_lon <= CONTEXT_LON_MAX) &
        (Pmax_lat >= CONTEXT_LAT_MIN) & (Pmax_lat <= CONTEXT_LAT_MAX)
    )
    Pmax_lat, Pmax_lon = Pmax_lat[valid], Pmax_lon[valid]

    labeled_array, _ = label(cores_t != 0)
    core_labels = np.unique(labeled_array[labeled_array != 0])

    dict_storm_size = {lab: np.sum(labeled_array == lab) * 9 for lab in core_labels}

    dict_storm_extent = {}
    for lab in core_labels:
        mask = labeled_array == lab
        dict_storm_extent[lab] = {
            "lat_min": float(np.nanmin(lats[mask])),
            "lat_max": float(np.nanmax(lats[mask])),
            "lon_min": float(np.nanmin(lons[mask])),
            "lon_max": float(np.nanmax(lons[mask]))
        }

    dict_storm_temperature = {}
    for lab in core_labels:
        mask = labeled_array == lab
        tir_core = tir_t[mask]
        yx_indices = np.argwhere(mask)
        y, x = yx_indices[np.argmin(tir_core)]
        box = extract_box(tir_t, y, x)
        dict_storm_temperature[lab] = float(np.mean(box))

    storm_database = {}
    for lat, lon in zip(Pmax_lat, Pmax_lon):
        try:
            y_idx, x_idx = snflics.to_yx(lat, lon, lats, lons)
            if y_idx is None or x_idx is None:
                continue
        except (IndexError, TypeError):
            continue
        lab = labeled_array[y_idx, x_idx]
        if lab == 0 or lab in storm_database:
            continue

        ext = dict_storm_extent[lab]
        storm_database[int(lab)] = {
            "lat": lat,
            "lon": lon,
            "lat_min": ext["lat_min"],
            "lat_max": ext["lat_max"],
            "lon_min": ext["lon_min"],
            "lon_max": ext["lon_max"],
            "tir": dict_storm_temperature[lab],
            "size": dict_storm_size[lab],
            "mask": 1
        }
    return storm_database


def generate_fictional_storm(context_lat_min, context_lat_max,
                             context_lon_min, context_lon_max):
    # Generate a dummy non-convective storm entry with mask=0
    lat = np.random.uniform(context_lat_min, context_lat_max)
    lon = np.random.uniform(context_lon_min, context_lon_max)

    storm = {
        "lat": lat,
        "lon": lon,
        "lat_min": lat,
        "lat_max": lat,
        "lon_min": lon,
        "lon_max": lon,
        "tir": 30.0,
        "size": 0.0,
        "mask": 0
    }

    return ("artificial", storm)


def pad_observed_storms(storm_db, nb_x0,
                        context_lat_min, context_lat_max,
                        context_lon_min, context_lon_max):
    # Ensure a fixed number of storm cores by truncating or padding

    storm_list = list(storm_db.items())

    if len(storm_list) >= nb_x0:
        sorted_db = sorted(storm_list, key=lambda item: item[1]["tir"])
        return sorted_db[:nb_x0]

    needed = nb_x0 - len(storm_list)
    for _ in range(needed):
        storm_list.append(
            generate_fictional_storm(
                context_lat_min=context_lat_min,
                context_lat_max=context_lat_max,
                context_lon_min=context_lon_min,
                context_lon_max=context_lon_max
            )
        )

    return storm_list


def transform_to_array(data):
    # Transform list of storms into an array of local per-core features

    result = []
    for _, entry in data:
        lat = float(entry["lat"])
        lon = float(entry["lon"])
        lat_min = float(entry.get("lat_min", lat))
        lat_max = float(entry.get("lat_max", lat))
        lon_min = float(entry.get("lon_min", lon))
        lon_max = float(entry.get("lon_max", lon))
        tir = float(entry["tir"])
        size = float(entry["size"])
        mask = int(entry["mask"])

        # [lat, lon, lat_min, lat_max, lon_min, lon_max, tir, size, mask]
        result.append([
            lat, lon,
            lat_min, lat_max,
            lon_min, lon_max,
            tir, size,
            mask
        ])

    return np.array(result, dtype=np.float32)

def process_file(file_t, nb_x0,
                 lats, lons,
                 CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                 CONTEXT_LON_MIN, CONTEXT_LON_MAX):
    # Process one NetCDF file and return the local per-core features

    try:
        with Dataset(file_t, "r") as data_t:
            x0_lat = data_t["max_lat"][:]
            x0_lon = data_t["max_lon"][:]
            if x0_lat.size == 0 or x0_lon.size == 0:
                return None

            storm_database = create_storm_database(data_t, lats, lons)

            X_features = pad_observed_storms(
                storm_database, nb_x0,
                CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                CONTEXT_LON_MIN, CONTEXT_LON_MAX
            )

            input_features = transform_to_array(X_features)

            input_tensor = torch.tensor(input_features, dtype=torch.float32)

        return input_tensor

    except Exception as e:
        print(f"Error processing {file_t}: {e}")
        return None


# MAIN LOOP
NB_X0 = 50

YEAR  = sys.argv[1]
MONTH = sys.argv[2]

DATA_PATH = "/gws/ssde/j25b/cocoon/SSA_domain/ch9_wavelet/"
YEAR_PATH = os.path.join(DATA_PATH, YEAR, MONTH)

print(f"Scanning {YEAR_PATH}", flush=True)

all_files = sorted(
    glob(os.path.join(YEAR_PATH, "**", "*.nc"), recursive=True)
)

print(len(all_files), flush=True)

if  len(all_files) ==0:
    sys.exit("No data found")

OUTPUT_FOLDER = "/work/scratch-nopw2/mendrika/OB/ncast/raw"

# Lags in minutes: from t0-2h to t0, every 30 minutes
lag_before_t = [120, 90, 60, 30, 0]
lead_times = [0, 60, 180, 360]

print("YEAR =", YEAR)
print("Total files found:", len(all_files))

for file_t in all_files[:]:

    time_t = snflics.get_time(file_t)
    print(time_t)

    file_before_t = [
        DATA_PATH + update_hour(time_t, hours_to_add=0, minutes_to_add=-m)["path"]
        for m in lag_before_t
    ]

    print(file_before_t)

    file_lead_times = [
        DATA_PATH + update_hour(time_t, hours_to_add=0, minutes_to_add=h)["path"]
        for h in lead_times
    ]

    print(file_lead_times)

    year   = int(time_t["year"])
    month  = int(time_t["month"])
    day    = int(time_t["day"])
    hour   = int(time_t["hour"])
    minute = int(time_t["minute"])

    NOWCAST_ORIGIN = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}"
    INPUT_LT0 = f"{OUTPUT_FOLDER}/inputs_t0/input-{NOWCAST_ORIGIN}.pt"
    OUTPUT_PATHS = {
        f"LT{m:03d}min": f"{OUTPUT_FOLDER}/targets_t{m:03d}min/target-{NOWCAST_ORIGIN}.pt"
        for m in lead_times
    }

    # Skip if this sample is already fully written
    if os.path.exists(INPUT_LT0) and all(
        os.path.exists(p) for p in OUTPUT_PATHS.values()
    ):
        continue

    # create input directory
    os.makedirs(os.path.dirname(INPUT_LT0), exist_ok=True)

    # create target directories
    for path in OUTPUT_PATHS.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)


    if not (
        all(os.path.exists(f) for f in file_lead_times)
        and all(os.path.exists(f) for f in file_before_t)
    ):
        print(f"Missing required files for {file_t}")
        continue

    try:
        core_series = [prepare_core(f) for f in file_lead_times]
    except OSError:
        print(f"Skipping {file_t}: unreadable core file.")
        continue

    if any(c is None for c in core_series):
        print(f"Skipping {file_t}: unreadable lead-time core file.")
        continue

    file_t0 = file_before_t[-1]

    with Dataset(file_t0, "r") as data_t:
        Pmax_lat = data_t["max_lat"][:]
        Pmax_lon = data_t["max_lon"][:]

        valid = (
            (Pmax_lon >= CONTEXT_LON_MIN) & (Pmax_lon <= CONTEXT_LON_MAX) &
            (Pmax_lat >= CONTEXT_LAT_MIN) & (Pmax_lat <= CONTEXT_LAT_MAX)
        )
        Pmax_lat, Pmax_lon = Pmax_lat[valid], Pmax_lon[valid]

        if Pmax_lat.size == 0:
            print(f"No core in the domain for {NOWCAST_ORIGIN}")
            continue

    lag_tensors = []
    for i, f in enumerate(file_before_t):
        t_tensor_local = process_file(
            f,
            nb_x0=NB_X0,
            lats=lats,
            lons=lons,
            CONTEXT_LAT_MIN=CONTEXT_LAT_MIN,
            CONTEXT_LAT_MAX=CONTEXT_LAT_MAX,
            CONTEXT_LON_MIN=CONTEXT_LON_MIN,
            CONTEXT_LON_MAX=CONTEXT_LON_MAX
        )
        if t_tensor_local is None:
            continue

        t_time = snflics.get_time(f)
        t_month = int(t_time["month"])
        t_hour = int(t_time["hour"])
        t_minute = int(t_time["minute"])

        month_angle = 2 * np.pi * (t_month - 1) / 12.0
        tod_angle = 2 * np.pi * (t_hour + t_minute / 60.0) / 24.0

        time_features = torch.tensor(
            [
                np.sin(month_angle), np.cos(month_angle),
                np.sin(tod_angle), np.cos(tod_angle)
            ],
            dtype=torch.float32
        ).unsqueeze(0).repeat(NB_X0, 1)

        t_tensor_full = torch.cat([time_features, t_tensor_local], dim=1)

        lag_tensors.append(t_tensor_full)

    if len(lag_tensors) != len(lag_before_t):
        print(f"Incomplete lag sequence for {NOWCAST_ORIGIN}")
        continue

    input_tensor = torch.stack(lag_tensors, dim=0)

    if input_tensor.shape == (5, NB_X0, 13):

        torch.save({
            "input_tensor": input_tensor,
            "nowcast_origin": NOWCAST_ORIGIN,
            "lags_minutes": lag_before_t,
        }, INPUT_LT0)
        print(f"Saved input tensor: {INPUT_LT0}")

        for i, (h, core) in enumerate(zip(lead_times, core_series)):
            target_tensor = torch.tensor(core != 0, dtype=torch.uint8)

            output_file_path = OUTPUT_PATHS[f"LT{h:03d}min"]

            torch.save({
                "data": target_tensor,
                "lead_time_minutes": h,
                "nowcast_origin": NOWCAST_ORIGIN
            }, output_file_path)

        print(f"Saved {len(lead_times)} targets for {NOWCAST_ORIGIN}")
