import numpy as np
import pandas as pd #for tables
import matplotlib.pyplot as plt
import os #is used in order not to hardcode the path inside the function;
import time  # Import time module for timing analysis, main function takes more than 15 min -> looking for a loop where this happens
from scipy.optimize import curve_fit #for broken power law fit / as a alternative, it basically can replace the whole function written before; and for new gaussian fit
import seaborn as sns 
from scipy.ndimage import gaussian_filter1d #to smooth the slope derivative -> can help with 'turning point towards end' problem
# import trackpy as tp #for van hove correlation; to compare with custom variant
from scipy.stats import norm #gaussian fit
import random #to pick random traj from non/hoppers


global_start = time.time()

"""
PIPELINE OVERVIEW (Jan 2026)

- msd_sum        : per-trajectory time-averaged MSD
- msd_mag_1s     : diagnostic (instructor-style), from msd_sum
- msd_mean       : ensemble mean MSD (used for gamma / plots)
- t_fit/msd_fit  : used for power-law fitting
- alpha split    : from bifurcate_by_msd (per-trajectory)
- hopper label   : Rg-based, chi^2 = 7.38 (final, defensible)

NOTE:
Hardcoded 1s cutoff below is legacy / plotting-only.
tracks_filtered = canonical tracks
"""
# !global storage for track comparison master function, is here - not to lost it. #JAN
metrics = {}


# new helper functions to find better fit:
#  the helper function for fitting a single power-law model in log-log space using np.polyfit


def single_powerlaw_fit(msd):
    """
    Fits a single power-law model (MSD = a * t^b) to the log-log MSD curve.

    Parameters:
        msd (array): Array of mean squared displacement values (non-zero).

    Returns:
        slope (float): The exponent b in the power law.
        intercept (float): The intercept in log-log space, i.e. log10(a).
    """
    # time vector: 1, 2, ..., len(msd)
    time_pt = np.arange(1, len(msd) + 1) #as before

    # converts to log-log scale (ignore any 0s or negative values to avoid -inf)
    valid = (msd > 0)
    time_log = np.log10(time_pt[valid]) #copied but with adding extra check
    msd_log = np.log10(msd[valid])

    # fit log(MSD) = b * log(t) + log(a)
    slope, intercept = np.polyfit(time_log, msd_log, 1)

    return slope, intercept

# broken power law fit: let's fit two linear regions with an automatically optimized turning point

def broken_powerlaw_fit(msd, min_segment=5):
    """
    Automatically fit a two-segment broken power law in log-log space.

    Parameters:
        msd (array): MSD values.
        min_segment (int): Minimum number of points in each segment.

    Returns:
        (tuple): slopes before and after optimal turning point, and the turning point index.
    """
    log_time = np.log10(np.arange(1, len(msd) + 1)) #time verctor; let's convert to logs init. to save time; if the motion follows a power law, this will show up as a straight line in log-log scale
    log_msd = np.log10(msd)

    # init.:
    best_error = np.inf # total least-squares error (smaller is better)
    best_slopes = (np.nan, np.nan)
    best_turn = -1 #the two slopes found at the optimal turning point.

    #  let's only test turning points that leave at least min_segment values before and after them, to avoid short segments

    for turn in range(min_segment, len(msd) - min_segment):
        # segment 1
        c1 = np.polyfit(log_time[:turn], log_msd[:turn], 1)
        # segment 2
        c2 = np.polyfit(log_time[turn:], log_msd[turn:], 1)

        # each call returns slope and intercept; for now i only care about c1[0] and c2[0], the slopes (i.e., power-law exponents α₁ and α₂)

        # calculates the total squared difference (error) between the fitted lines and the actual data
        fit1 = np.polyval(c1, log_time[:turn])
        fit2 = np.polyval(c2, log_time[turn:])
        error = np.sum((log_msd[:turn] - fit1) ** 2) + np.sum((log_msd[turn:] - fit2) ** 2)

        if error < best_error:
            best_error = error
            best_slopes = (c1[0], c2[0])
            best_turn = turn

    return best_slopes, best_turn


# not mentioned but may make fitting better => 3 step
def broken_powerlaw_fit_3step(msd, min_size=10):
    log_msd = np.log10(msd)
    log_time = np.log10(np.arange(1, len(msd) + 1))

    best_error = np.inf
    best_slopes = (0, 0, 0)
    best_breaks = (min_size, 2 * min_size)

    for i in range(min_size, len(msd) - 2 * min_size):
        for j in range(i + min_size, len(msd) - min_size):
            slope1, _ = np.polyfit(log_time[:i], log_msd[:i], 1)
            fit1 = slope1 * log_time[:i] + (log_msd[:i].mean() - slope1 * log_time[:i].mean())
            err1 = np.sum((log_msd[:i] - fit1) ** 2)

            slope2, _ = np.polyfit(log_time[i:j], log_msd[i:j], 1)
            fit2 = slope2 * log_time[i:j] + (log_msd[i:j].mean() - slope2 * log_time[i:j].mean())
            err2 = np.sum((log_msd[i:j] - fit2) ** 2)

            slope3, _ = np.polyfit(log_time[j:], log_msd[j:], 1)
            fit3 = slope3 * log_time[j:] + (log_msd[j:].mean() - slope3 * log_time[j:].mean())
            err3 = np.sum((log_msd[j:] - fit3) ** 2)

            total_err = err1 + err2 + err3

            if total_err < best_error:
                best_error = total_err
                best_slopes = (slope1, slope2, slope3)
                best_breaks = (i, j)

    return best_slopes, best_breaks

# inspired by Caden Gobat's broken power law notebook (adapted and modified)
# Source: https://gist.github.com/cgobat/12595d4e242576d4d84b1b682476317d
# Once I wanted to improve my fitting part of code and implement scipy.optimize, I looked for an example how to use broken power laws efficiently and liked this one
#  it's modified many times and lookes different (apart from the name) but mentioned above since I was using this idea as a refference
# improvement idea: eliminate trade-off of hardcoded alphas (now uses only initial guess)
# problem solved: f(x) ∝ (x / breaks[0])^α_i  # no vertical shift or normalization constant

# def. the bkn_pow evaluator
def bkn_pow(x, A, alpha1, alpha2, alpha3, break1, break2):
    """
    Three-segment broken power law: MSD = A * t^α with piecewise exponents
    """
    # Convert breaks to ints if needed
    break1 = int(round(break1))
    break2 = int(round(break2))

    # Output array
    y = np.zeros_like(x, dtype=float)

    for i, t in enumerate(x):
        if t < break1:
            alpha = alpha1
        elif t < break2:
            alpha = alpha2
        else:
            alpha = alpha3
        y[i] = A * (t ** alpha)

    return y


def bkn_pow_2seg(x, A, alpha1, alpha2, break1):
    """
    Two-segment broken power law fit with continuous intercept.
    MSD = A * t^alpha1 for t < break1,
    MSD = A2 * t^alpha2 for t >= break1,
    where A2 = A * break1^(alpha1 - alpha2) ensures continuity.
    """
    y = np.zeros_like(x, dtype=float)
    for i, t in enumerate(x):
        if t < break1:
            y[i] = A * t**alpha1
        else:
            A2 = A * break1**(alpha1 - alpha2)
            y[i] = A2 * t**alpha2
    return y, A2

def find_turning_point(msd_curve, min_idx=3):
    """
    Find turning point in MSD curve where the log-log slope changes significantly.
    """
    log_time = np.log10(np.arange(1, len(msd_curve) + 1))
    log_msd = np.log10(msd_curve)

    # apply smoothing to log(MSD)
    log_msd_smooth = gaussian_filter1d(log_msd, sigma=1)
    
    # first derivative of log(MSD) wrt log(time)
    dlogmsd = np.gradient(log_msd_smooth, log_time)
    slope_change = np.abs(np.gradient(dlogmsd))
    
    # # ignore early noisy region (e.g., first 3 points)
    # turning_pt = np.argmax(slope_change[min_idx:]) + min_idx
    
    # Restrict to middle 60% of indices => as it's always towards end -> can be more effective than above for noicier regions
    start = int(0.2 * len(slope_change))
    end = int(0.7 * len(slope_change)) #appears that 50% (-30%) filtered towards the end workeed better

    # Find maximum slope change within this window
    restricted_change = slope_change[start:end]
    turning_pt = np.argmax(restricted_change) + start

    return turning_pt

def classify_powerlaw_type(alpha1, alpha2, threshold=0.4):
    """
    Classifies an MSD curve as 'single' or 'double' power law based on the
    difference between alpha1 and alpha2.

    Parameters:
        alpha1 (float): Slope before the turning point.
        alpha2 (float): Slope after the turning point.
        threshold (float): Maximum allowed difference to consider it 'single'.
        For now it is 0.4 as it’s commonly used in literature as a cutoff

    Returns:
        str: 'single' if difference is small, 'double' otherwise.
    """
    if abs(alpha1 - alpha2) < threshold:
        return 'single'
    else:
        return 'double'



# to calculate MSD over the entire trajectory

def calc_msd_2D_longtrack(traj, timewave, time_ratio):
    """
    Calculates the Mean Squared Displacement (MSD) for a single track over its entire length.
    I tried to use the same var. names as in original script.

    Parameters:

        traj (array): 2D array of coordinates (x, y).
            => For now, I decided not to introduce commented z variable.

        timewave (array): Time array adjusted for frame intervals. New: Position_T intervals

        time_ratio (int): Det. the length of the lag time interval.

    Returns:
        (array): MSD values for each lag time.
    """
    # to adjust timewave to start at 1 instead of 0
    timewave_edit = (timewave + 1).astype(int)
    
    # max. time index
    max_index = int(np.max(timewave_edit))
    time_max = max(len(traj), max_index + 1)

    # init. coordinate arrays with NaN values, as positions are ignored during computations (to use np.nanmean() instead of np.mean()-> can produce more data points than we have)
    x = np.full(time_max, np.nan)
    y = np.full(time_max, np.nan)

    # new fix:
    # apply boundary check to avoid index errors
    valid_indices = timewave_edit[timewave_edit < time_max]  # Ensure indices are within bounds
    valid_indices = valid_indices[:len(traj)]  # Ensure length consistency with traj

    # let's populate x and y with trajectory data at the corresponding time indices

    # .astype(int) is needed because time indices have to be converted to int. vals before being used to index the x and y arrays (aviods possible errors)
    # let's populate x and y with trajectory data at the corresponding time indices
    x[valid_indices] = traj[:len(valid_indices), 0]
    y[valid_indices] = traj[:len(valid_indices), 1]

    #length of time series
    time_length = len(x)

    # the number of lag times to compute
    msd_length = time_length // time_ratio

    # setting MSD array
    # msd = np.zeros(msd_length)
    msd = np.full(msd_length, np.nan)


    # to calc. MSD for each lag time - time int. over which the displacement is measured when calc. MSD #change (0,msd_length)
    for lagtime in range(1, msd_length):

        # calc. displacements at each lag time
        delta_x = x[:-lagtime] - x[lagtime:]
        delta_y = y[:-lagtime] - y[lagtime:]

        # computes MSD, ignoring NaNs
        msd[lagtime] = np.nanmean(delta_x**2 + delta_y**2)

    return msd

# to calculate ensemble MSD using only the init. point as a reference

def calc_msd_2D_ensemble_longtrack(traj, timewave, time_ratio):
    """
    Calculates the Mean Squared Displacement (MSD) using only the initial point as a reference.
    Returns a compact, clean MSD array.
    # New "more straightforward" version of this function
    """
    num_points = len(traj)
    msd_length = num_points // time_ratio
    msd = np.zeros(msd_length)

    initial_x, initial_y = traj[0, 0], traj[0, 1]

    for i in range(1, msd_length):
        dx = traj[i, 0] - initial_x
        dy = traj[i, 1] - initial_y
        msd[i] = dx**2 + dy**2

    # # debug prints
    # print("Fixed Ensemble MSD (first 10):", msd[:10])
    return msd

# to calculate RG for the entire trajectory

def calc_Rg(traj):
    """
    Calculates the Radius of Gyration (Rg) for the entire trajectory.

    Parameters:
        traj (array): 2D array of coordinates (x, y).

    Returns:
        (float): Rg.
    """
    # to extract x and y coordinates
    x = traj[:, 0]
    y = traj[:, 1]

    # to calc. mean positions, ignoring NaNs
    #can use np.mean() here but decided to leave as it is as the difference in time to run is insignificant and this can help to avoid errors if NaNs appear due to preprocessing steps
    xc = np.nanmean(x) 
    yc = np.nanmean(y)

    # to calc. Rg
    Rg = np.sqrt(np.nanmean((x - xc)**2 + (y - yc)**2))
    return Rg


# to calculate RG for segments of the trajectory

def calc_Rg_seg(traj, seg_size):
    """
    Calculates the Radius of Gyration (Rg) for segments of the trajectory.

    Parameters:
        traj (array): 2D array of coordinates (x, y).
        seg_size (int): Size of each segment.

    Returns:
        (array): Rg values for each segment.
    """
    # to extract x and y coordinates
    x = traj[:, 0]
    y = traj[:, 1]

    # det. # of segments
    n_seg = len(traj) // seg_size
    Rg = np.zeros(n_seg) #setting array

    # calculates Rg for each segment (for loop)
    for i in range(n_seg):

        # def. segment indices
        start_idx = i * seg_size
        end_idx = start_idx + seg_size

        # getting data
        x_seg = x[start_idx:end_idx]
        y_seg = y[start_idx:end_idx]

        # mean positions, ignoring NaNs
        xc = np.nanmean(x_seg)
        yc = np.nanmean(y_seg)

        #Rg for the segment
        Rg[i] = np.sqrt(np.nanmean((x_seg - xc)**2 + (y_seg - yc)**2))

    return Rg

# data loaders (i will skip the matlab one for now, hope we won't need need as everything will be in python)
# to load and process TrackMate CSV data

def load_trackmate_track(file_path):
    """
    Load TrackMate track data from a CSV file, group by TRACK_ID, and sort by Position_T intervals.
    Handles both single-point and multi-point tracks, but only includes tracks with more than 1 point.

    Parameters:
        file_path (str): Path to the .csv file.

    Returns:
        list: List of tracks, each as a NumPy array with columns [POSITION_X, POSITION_Y, Position_T].
    """

    # Confirm column mapping with a quick inspection
    preview = pd.read_csv(file_path, nrows=1)
    print(preview.columns)

    # Corrected column indices
    # Column 0: Label (ignored)
    # Column 1: ID (ignored)
    # Column 2: TRACK_ID (this is what we need)
    # Column 3: QUALITY (can be ignored for now)
    # Column 4: POSITION_X
    # Column 5: POSITION_Y
    # Column 6: POSITION_Z (optional)
    # Column 7: FRAME

    columns = ['TRACK_ID', 'QUALITY', 'POSITION_X', 'POSITION_Y', 'POSITION_T']
    data = pd.read_csv(file_path, usecols=[2, 3, 4, 5, 7], names=columns, header=0)

    # Confirm the first few rows to ensure correct reading
    print(data.head(10))
    print(data.dtypes)

    # Ensure TRACK_ID is treated as an integer
    data['TRACK_ID'] = data['TRACK_ID'].astype(int)

    # Initialize a list to store valid tracks
    tracks = []

    # Group by TRACK_ID
    grouped = data.groupby('TRACK_ID')

    # Debug: Print number of points per TRACK_ID
    # for track_id, group in grouped:
    #     print(f"TRACK_ID: {track_id}, Points: {len(group)}")

    # Iterate through each group
    for track_id, group in grouped:
        # Sort by Position_T to ensure chronological order
        sorted_group = group.sort_values('POSITION_T')[['POSITION_X', 'POSITION_Y', 'POSITION_T']].to_numpy()

        # Include only tracks with more than 1 point
        if len(sorted_group) > 1:
            # print(f"Adding TRACK_ID {track_id} with {len(sorted_group)} points.") to debug ! TO SEE TRACKS
            tracks.append(sorted_group)

    print(f"Loaded {len(tracks)} tracks from {file_path} (Tracks with 1 point excluded)")
    return tracks


# to fit two segments of the MSD curve in log-log space

def MSD_exp_twostep(msd_temp, turning_pt):
    """
    Fits two segments of the MSD curve in log-log space and extract slopes.

    Parameters:
        msd_temp (array): Array of MSD values.
        turning_pt (int): Index marking the transition point.

    Returns:
        (array): Two slopes representing the power-law exp. before and after the turning point. (upd):Intercepts added to solve the issue with "elevated graph"
        (tuple): ((slope1, intercept1), (slope2, intercept2))
    """
    time_pt = np.arange(1, len(msd_temp) + 1) #+1 turns 4 to 3 later
    time_pt_log = np.log10(time_pt)
    msd_temp_log = np.log10(msd_temp)

    # First segment fitting
    c1 = np.polyfit(time_pt_log[3:turning_pt], msd_temp_log[3:turning_pt], 1)
    slope1, intercept1 = c1

    # Second segment
    c2 = np.polyfit(time_pt_log[turning_pt:], msd_temp_log[turning_pt:], 1)
    slope2, intercept2 = c2

    return (slope1, intercept1), (slope2, intercept2)

#  helper: log-binning on τ (frames); to make msd fit look better with more data
def log_bin_tau(tau, y, weights, bins_per_decade=10, min_pts=3, keep_first=2):
    tau = np.asarray(tau, float)
    y   = np.asarray(y,   float)
    w   = np.asarray(weights, float)
    m = np.isfinite(tau) & np.isfinite(y) & np.isfinite(w) & (tau > 0) & (w > 0)
    tau, y, w = tau[m], y[m], w[m]
    # keep the first few small-τ points unbinned
    order = np.argsort(tau)
    tau, y, w = tau[order], y[order], w[order]
    keep = min(keep_first, len(tau))
    tau_keep, y_keep, w_keep = tau[:keep], y[:keep], w[:keep]
    tau_rest, y_rest, w_rest = tau[keep:], y[keep:], w[keep:]
    if len(tau_rest) == 0:
        return tau_keep, y_keep, w_keep, np.zeros_like(y_keep)

    tmin, tmax = float(np.min(tau_rest)), float(np.max(tau_rest))
    if tmin <= 0 or tmax <= tmin:
        return tau_keep, y_keep, w_keep, np.zeros_like(y_keep)
    decades = np.log10(tmax) - np.log10(tmin)
    nbins = max(int(np.ceil(decades * bins_per_decade)), 1)
    edges = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)

    idx = np.digitize(tau_rest, edges) - 1
    T, Y, W, SE = [], [], [], []
    for b in range(nbins):
        sel = (idx == b)
        if np.sum(sel) < min_pts:
            continue
        tb, yb, wb = tau_rest[sel], y_rest[sel], w_rest[sel]
        wsum = np.sum(wb)
        ybar = np.sum(wb * yb) / wsum
        trep = np.exp(np.mean(np.log(tb)))           # geometric mean lag
        var = np.sum(wb * (yb - ybar)**2) / (wsum**2)
        se = float(np.sqrt(var))                     # SE of weighted mean (approx)
        T.append(trep); Y.append(ybar); W.append(wsum); SE.append(se)

    tau_b = np.concatenate([tau_keep, np.array(T)])
    y_b   = np.concatenate([y_keep,   np.array(Y)])
    w_b   = np.concatenate([w_keep,   np.array(W)])
    se_b  = np.concatenate([np.zeros_like(y_keep), np.array(SE)])
    order2 = np.argsort(tau_b)
    return tau_b[order2], y_b[order2], w_b[order2], se_b[order2]


# to differentiate fast traj. from slow

def bifurcate_by_msd(params_df, tracks, alpha_fast=0.6, min_break_s=None):
    """
    To split trajectories into 'fast' vs 'slow' using existing single/double MSD fits.

    Expected columns if available (function tolerates missing ones):
      - 'traj_id' (int) : index to 'tracks'
      - 'model' in {'single','double'}
      - 'alpha_single'   (float)
      - 'alpha1','alpha2' (float)  # for 2-seg
      - 'tau_break_s'     (float)  # opt. (break in seconds)

    Returns: fast_tracks, slow_tracks, summary_df
    """
    need_cols = ["traj_id", "model"]
    for c in need_cols:
        if c not in params_df.columns:
            raise KeyError(f"params_df needs column '{c}'")

    # make safe accessors
    def get(row, key, default=np.nan):
        return row[key] if key in params_df.columns and pd.notnull(row[key]) else default

    fast_ids, slow_ids, rows = [], [], []

    for _, r in params_df.iterrows():
        tid  = int(r["traj_id"])
        mdl  = str(r["model"]).lower()
        a1   = get(r, "alpha1")
        a2   = get(r, "alpha2")
        asg  = get(r, "alpha_single")
        tbrk = get(r, "tau_break_s")

        # choose which alpha to judge
        if mdl == "double" and pd.notnull(a2):
            alpha_eff = a2
            pass_break = True if (min_break_s is None or (pd.notnull(tbrk) and tbrk >= min_break_s)) else False
        else:
            alpha_eff = asg
            pass_break = True  # no break criterion for single

        is_fast = (pd.notnull(alpha_eff) and (alpha_eff >= alpha_fast) and pass_break)

        rows.append({
            "traj_id": tid,
            "model": mdl,
            "alpha_eff": alpha_eff,
            "tau_break_s": tbrk,
            "is_fast": bool(is_fast)
        })
        (fast_ids if is_fast else slow_ids).append(tid)

    summary = pd.DataFrame(rows)

    # map IDs to track arrays (skip missing/short tracks safely)
    def select(ids):
        out = []
        for i in ids:
            if i < len(tracks) and tracks[i] is not None and len(tracks[i]) >= 2:
                out.append(tracks[i])
        return out

    fast_tracks = select(fast_ids)
    slow_tracks = select(slow_ids)
    return fast_tracks, slow_tracks, summary


# main function;
# to process and analyze track data

def CalcMSD(folder_path, min_length=200, time_ratio=2, seg_size=10): #enlarge min length -> 100, 200, van hoff corelation function =>top 10-20 of longest traj
    """
    Process trajectory data and calculate MSD, non-ergodicity, and Rg.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        min_length (int): Minimum track length to consider.
        time_ratio (int): Ratio for MSD calculation.
        seg_size (int): Segment size for Rg calculation.

    Returns:
        None, shows plots. UPD: tracks_filtered for van hove corelation
    """
    #start counting
    start_time = time.time()
    print("Starting process_tracks...")

    # to load all track data, collects all relevant CSV files starting with 'export_' in the specified folder (i can delete the first part but our files start with "export_tracks_")
    track_files = [f for f in os.listdir(folder_path) if f.startswith('export_') and f.endswith('.csv')]

    #an empty list to store track data
    tracks = []

    #loading data, iterating over all files
    for file_name in track_files:
        file_path = os.path.join(folder_path, file_name)

        print(f"Reading file: {file_name}")
        read_start = time.time()

        track_data = load_trackmate_track(file_path)

        print(f"Finished reading {file_name} in {time.time() - read_start:.2f} seconds. Number of tracks: {len(track_data)}")

        tracks.extend(track_data)

    # to filter tracks by min. length as in original matlab  script
    # only tracks longer than min_length are considered for MSD calculation.

    filter_start = time.time()

    tracks_filtered = [track for track in tracks if len(track) > min_length]

    print(f"Filtered tracks in {time.time() - filter_start:.2f} seconds. Remaining tracks: {len(tracks_filtered)}")

    # helpers for 'speed' traj bifurcation init.
    # --- initialize containers and defaults ---
    param_rows = []       # to collect per-trajectory fit results
    params_df = None      # placeholder for final DataFrame
    fast_trajs = []       # to store fast-mobility trajectories
    slow_trajs = []       # to store slow-mobility trajectories
    summary = None        # to hold bifurcation summary table
    param_rows_traj = []   # one row per trajectory (for bifurcation)




    # to calculate MSD:
    msd_sum = [] #storage

    msd_start = time.time()

    for track in tracks_filtered:
        # wrong index
        timewave = (track[:, 2] - track[0, 2])/ 25 #scaling by 25 as in original code, adjusts time int. to match original scale 
        # (track[:, 2] - track[0, 2]) calculates the frame offset relative to the first frame as [:, 2] extracts the FRAME column (third column) for the current track (explained above in the code)
        # use position instead- position t
        # simply removing the scaling won't work: RuntimeWarning: Mean of empty slice => i.e. intervals become huge: I'm treating large integer values as direct time steps. 
        
        
        # to remember: [:, :2] - for X and Y and [:, 2] - for time.
        msd_temp = calc_msd_2D_longtrack(track[:, :2], timewave, time_ratio) # msd for current track, calls function
        # msd_sum.append(msd_temp) #add to storage
        # fix to avoid nans: 
        if msd_temp is not None and len(msd_temp) > 1:
            msd_sum.append(msd_temp)



    print(f"MSD calculation completed in {time.time() - msd_start:.2f} seconds.")

    # constructing MSD matrix, a bit different from matlab version but works

    # let's det. the maximum track length to understand matrix size
    max_length = max(len(msd) for msd in msd_sum)
    msd_matrix = np.full((len(msd_sum), max_length), np.nan) #creates a matrix to store MSD values filled with NaNs
    for i, msd in enumerate(msd_sum): #"replacing" nans with actual calc. vals
        msd_matrix[i, :len(msd)] = msd

    # calculating average MSD (ignore nans => nanmean)
    msd_mean = np.nanmean(msd_matrix, axis=0)

    # JAN
    # master metrics

    # =========================
    # Level 1: initialize per-trajectory metrics container
    # =========================
    metrics = {}

    for traj_id, track in enumerate(tracks_filtered):
        metrics[traj_id] = {"traj_id": traj_id, "length_frames": len(track),}

    # Level 2: MSD magnitude at fixed lag (similar Prof.'s Shi diagnostic)
    dt = 0.025
    lag_frames = int(round(1.0 / dt))
    idx = lag_frames - 1  # if msd[0] = lag 1 frame

    for traj_id, msd in enumerate(msd_sum):
        if msd is not None and len(msd) > idx:
            metrics[traj_id]["msd_mag_1s"] = float(msd[idx])
        else:
            metrics[traj_id]["msd_mag_1s"] = np.nan


    # plotting, MSD curve in log-log scale
    plt.figure()
    plt.plot(np.arange(1, len(msd_mean) + 1) * 0.025, msd_mean)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([5E-3, 1E-1])
    plt.title('Mean MSD over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('MSD')
    plt.savefig("Figure 1: mean_msd_loglog.png", dpi=300)
    # plt.show()

    msd_df = pd.DataFrame({"time_s": np.arange(1, len(msd_mean) + 1) * 0.025, "mean_msd": msd_mean })
    msd_df.to_csv("Table 1: mean_msd_loglog.csv", index=False)


    # now, let's calculate ensemble MSD and non-ergodicity
    msd_ensemble_sum = [] #storage

    ensemble_start = time.time()

    for track in tracks_filtered:  #same as above but using second function, fix added
        timewave = (track[:, 2] - track[0, 2]) / 25.0
        msd_ensemble_temp = calc_msd_2D_ensemble_longtrack(track[:, :2], timewave, time_ratio)
        
        # Only include if not None, has more than 1 point, and not all NaNs
        if (
            msd_ensemble_temp is not None
            and len(msd_ensemble_temp) > 1
            and not np.isnan(msd_ensemble_temp).all()
        ):
            msd_ensemble_sum.append(msd_ensemble_temp)
        else:
            print("Skipping a track due to invalid or empty MSD.")


    print(f"Ensemble MSD calculation completed in {time.time() - ensemble_start:.2f} seconds.")

    # to construct ensemble MSD matrix
    ensemble_matrix = np.full((len(msd_ensemble_sum), max_length), np.nan)
    for i, msd in enumerate(msd_ensemble_sum):
        ensemble_matrix[i, :len(msd)] = msd

    # to calculate the mean ensemble MSD and the non-ergodicity parameter (gamma, is called 'gamma-sum' in matlab)
    msd_ensemble_mean = np.nanmean(ensemble_matrix, axis=0)
    msd_ensemble_mean[msd_ensemble_mean == 0] = np.nan #prevents /0 => avoids nans (fix)

    # # debug:
    # print("msd_mean:", msd_mean)
    # print("msd_ensemble_mean:", msd_ensemble_mean)
    # print("Any NaNs in msd_mean?", np.isnan(msd_mean).any())
    # print("Any NaNs in msd_ensemble_mean?", np.isnan(msd_ensemble_mean).any())
    # print("Number of valid (non-nan) MSD ensemble entries:", np.sum(~np.isnan(msd_ensemble_mean)))



    gamma = (msd_ensemble_mean - msd_mean) / msd_ensemble_mean
    # here: the different. /by # data points in msd ; include both
    # make a plot of it =>how gamma changes with lag time (small)
    # print(gamma)

    # ver2 from paper mmc9:
    # corresponds to eq. 3 :
    relative_diff = (msd_ensemble_mean - msd_mean) / msd_ensemble_mean
    gamma_v2 = np.nanmean(relative_diff)
    # print(gamma_v2)
    
    # plot to visualize gamma vs lag time (use time axis matching msd) - later in code after lag times are defined

    # to calculate Rg
    Rg_all = np.array([calc_Rg(track[:, :2]) for track in tracks_filtered]) #for each track
    Rg_seg = [calc_Rg_seg(track[:, :2], seg_size) for track in tracks_filtered] #fro segmented tracks

    # save date instead of plotting => save csvs 
    # plotting Rg distributions
    plt.figure()
    plt.hist(Rg_all, bins=50, alpha=0.5, label='Overall Rg')
    plt.hist(np.hstack(Rg_seg), bins=50, alpha=0.5, label='Segmented Rg')
    plt.xlabel('Rg (units)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Radius of Gyration Distribution')
    plt.savefig("Figure 2: Rg.png", dpi=300)
    # plt.show()

    # flatten segmented Rg array just in case it's a list of arrays
    Rg_seg_flat = np.hstack(Rg_seg)

    # save Rg distributions to separate CSV files
    pd.DataFrame({"Rg": Rg_all}).to_csv("Table 2: Rg_all.csv", index=False)
    pd.DataFrame({"Rg": Rg_seg_flat}).to_csv("Table 2_1: Rg_segmented.csv", index=False)


    # output gamma
    print('Non-ergodicity parameter (gamma):', gamma)
    print('Non-ergodicity parameter (gamma-paper version):', gamma_v2)

    print(f"Total processing time: {time.time() - start_time:.2f} seconds.")
    
    # plot for gamma

    # to ensure all gamma values align and filter out invalid (NaN) values
    lag_times = np.arange(1, len(msd_ensemble_mean) + 1) * 0.025

    # mask valid points (non-NaN in both versions)
    valid_mask = (~np.isnan(gamma)) & (~np.isnan(gamma_v2)) & (~np.isnan(lag_times))

    # # Def. mask for <= 1s
    # cutoff_mask = valid_mask <= 1.0

    # final plot
    plt.figure()
    plt.plot(lag_times[valid_mask], gamma[valid_mask], label='γ - v1: Rel. Diff')
    
    # scalar paper gamma_v2 as horizontal line:
    plt.axhline(gamma_v2, linestyle='--', color='gray', label=f'γ - v3: Mean Rel. Diff (={gamma_v2:.3f})')  # scalar
    plt.xlabel('Time lag (s)')
    plt.ylabel('γ (Non-ergodicity parameter)')
    plt.title('γ vs Lag Time')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    # plt.show()
    plt.savefig("Figure 3: gamma_vs_lagtime.png", dpi=300)
    # save data
    gamma_df = pd.DataFrame({
        "lag_time_s": lag_times[valid_mask],
        "gamma_v1": gamma[valid_mask],
        "gamma_v2": gamma_v2
    })
    gamma_df.to_csv("Table 3: gamma_cutoff.csv", index=False)

    # testing differents fits, for now: only two new functions \(single break point)

    #as valid time was not defined in init. given msd two-step (which i included in the later functions i wrote),let's do it again here
    #  def. the time vector -copied
    t = np.arange(1, len(msd_mean) + 1) #as time var.; (local ) is used to check the exec. of code(debugging) => otherwise causes errors

    # # ensuring MSD has no zeros or NaNs + t<=1 s
    # mask = int(1 / 0.025)  # 1s in frame units → 40
    # valid = (msd_mean > 0) & (~np.isnan(msd_mean)) & (t <= mask)
    # time_valid = t[valid]
    # msd_valid = msd_mean[valid] #now we have valid time to use

    #JAN -keep in mind

    # NOTE (2026-01): time_valid/msd_valid use a hardcoded 1s cutoff (legacy / for plotting only).
    # Current fitting uses t_fit/msd_fit based on FIT_MAX_S below, so this cutoff should not affect fits.
    # If later we refactor, keep this separation explicit.

    # Basic clean mask
    valid = (msd_mean > 0) & (~np.isnan(msd_mean))
    t_clean = t[valid]
    msd_clean = msd_mean[valid]

    # # Then apply 1s cutoff AFTER cleaning
    cutoff_mask = (t_clean * 0.025 <= 1.0)  # apply seconds-based filter
    time_valid = t_clean[cutoff_mask]
    msd_valid = msd_clean[cutoff_mask]

    # FOR LOG BINNING
    
    # contributors per lag (how many tracks contributed to each mean)
    # -> use as weights for binning. Align with 'valid' mask.
    n_contrib_full = np.sum(~np.isnan(msd_matrix), axis=0)   # shape: (max_length,)
    n_contrib = n_contrib_full[valid]

    # (optional) apply 1s cutoff for plotting-only views => check next time
    n_contrib_valid = n_contrib[cutoff_mask]


    # to compute binned MSD on τ (still in frames)
    tau_b, msd_b, w_b, se_b = log_bin_tau(t_clean, msd_clean, n_contrib, bins_per_decade=10, min_pts=3, keep_first=2)

    # SAFE INITIALIZATION for binned fits -too many mistamashed axes errors
    dt = 0.025  # your frame interval
    # ensure these names always exist in this scope
    sb = ib = np.nan
    msd_fit_single_binned = None

    popt_2seg_b = None
    msd_fit_2seg_binned = None


    # fit only cutoff mask

    FIT_MAX_S = 10.0   # or 1.0 — your choice for the fit-only cutoff
    fit_mask = (t_clean * 0.025) <= FIT_MAX_S

    # convenience views used ONLY for fitting
    t_fit   = t_clean[fit_mask]
    msd_fit = msd_clean[fit_mask]
        

    # safe cutoff mask & sliced arrays (even if tau_b is empty)
    if len(tau_b) > 0:
        b_fit_mask = (tau_b * dt) <= FIT_MAX_S     # e.g., FIT_MAX_S = 10.0
        tau_b_fit  = tau_b[b_fit_mask]
        msd_b_fit  = msd_b[b_fit_mask]
    else:
        b_fit_mask = np.array([], dtype=bool)
        tau_b_fit  = np.array([])
        msd_b_fit  = np.array([])


    # #
    # # # we're interedsted in everything <= 1s: not were effective as well as if loops, let's edit valid instead
    # mask = time_valid <= 10.0 #or 1.0
    # time_valid_new = time_valid[mask]
    # msd_valid_new = msd_valid[mask]  # Filtered to match time_valid


    turning_pt = 30 #for msd two-step (fixed), that's why broken power law with automatic one is better

    # single power-law fit
    # slope_single, intercept_single = single_powerlaw_fit(msd_clean)
    # msd_fit_single = 10**intercept_single * (t_clean ** slope_single)


    # single power-law fit (fit only to early times)
    slope_single, intercept_single = single_powerlaw_fit(msd_fit)

    # evaluate the model only where we actually fit; NaN elsewhere
    msd_fit_single = np.full_like(msd_clean, np.nan, dtype=float)
    msd_fit_single[fit_mask] = (10**intercept_single) * (t_clean[fit_mask] ** slope_single)


    log_time = np.log10(t_clean)
    

    # new 2 segment fit:
    # --- 2-segment continuous fit using bkn_pow_2seg ---
    # break1 = find_turning_point(msd_clean)  # automatic turning point
    # A_guess = np.mean(msd_clean[:5])
    # initial_guess = [A_guess, 0.3, 1.0]
    # bounds_2seg = ([1e-5, 0.1, 0.1], [10, 3.0, 3.0])

    # def fit_wrapper(x, A, alpha1, alpha2):
    #     return bkn_pow_2seg(x, A, alpha1, alpha2, break1)[0]

    # popt_2seg, _ = curve_fit(fit_wrapper, t_clean, msd_clean, p0=initial_guess, bounds=bounds_2seg)
    # msd_fit_2seg, A2_2seg = bkn_pow_2seg(t_clean, *popt_2seg, break1)


    # 2-segment continuous fit using masked early data
    break1 = find_turning_point(msd_fit)  # pick break on the same early range
    A_guess = np.nanmean(msd_fit[:5])
    initial_guess = [A_guess, 0.3, 1.0]
    bounds_2seg = ([1e-5, 0.1, 0.1], [10, 3.0, 3.0])

    def fit_wrapper(x, A, alpha1, alpha2):
        return bkn_pow_2seg(x, A, alpha1, alpha2, break1)[0]

    popt_2seg, _ = curve_fit(fit_wrapper, t_fit, msd_fit, p0=initial_guess, bounds=bounds_2seg)

    # store as a full-length series but fill only where we actually fit
    msd_fit_2seg = np.full_like(msd_clean, np.nan, dtype=float)
    msd_fit_2seg[fit_mask], A2_2seg = bkn_pow_2seg(t_clean[fit_mask], *popt_2seg, break1)



    # respect the same cutoff in the binned series
    b_fit_mask = (tau_b * 0.025) <= FIT_MAX_S
    tau_b_fit  = tau_b[b_fit_mask]
    msd_b_fit  = msd_b[b_fit_mask]


    # NEW: single power-law on the BINNED curve
    # if len(tau_b) >= 2 and np.all(tau_b > 0) and np.all(msd_b > 0):
    #     # Straight log–log regression on (tau_b, msd_b)
    #     sb, ib = np.polyfit(np.log10(tau_b), np.log10(msd_b), 1)
    #     msd_fit_single_binned = 10**ib * (tau_b ** sb)
    # else:
    #     sb, ib = np.nan, np.nan
    #     msd_fit_single_binned = None


    # after mask
    # single binned fit (only within cutoff)
    if len(tau_b_fit) >= 2 and np.all(tau_b_fit > 0) and np.all(msd_b_fit > 0):
        sb, ib = np.polyfit(np.log10(tau_b_fit), np.log10(msd_b_fit), 1)
        msd_fit_single_binned = 10**ib * (tau_b_fit ** sb)      # plot vs tau_b_fit*0.025
    else:
        sb, ib = np.nan, np.nan
        msd_fit_single_binned = None



    # reminder (use tau_b_fit, msd_b_fit instead of tau_b, msd_b)
    # 2-segment (broken power-law) on the BINNED curve
    msd_fit_2seg_binned = None
    popt_2seg_b = None
    if (msd_fit_single_binned is not None) and (len(tau_b_fit) >= 6):
        try:
            # choose a break on the binned series; ensure enough points on both sides
            # helper find_turning_point, use it on msd_b; otherwise pick mid-index
            if 'find_turning_point' in globals():
                break_b = int(find_turning_point(msd_b_fit))
            else:
                break_b = len(tau_b_fit) // 2

            # clamp break to have at least 3 points each side (tune later!)
            break_b = int(np.clip(break_b, 3, len(tau_b_fit) - 3))

            # wrapper using same bkn_pow_2seg signature you already use
            def fit_wrapper_b(x, A, alpha1, alpha2):
                return bkn_pow_2seg(x, A, alpha1, alpha2, break_b)[0]

            # initial guess and bounds (matching your raw fit style)
            A_guess_b = max(np.mean(msd_b_fit[:min(5, len(msd_b_fit))]), 1e-5)
            initial_guess_b = [A_guess_b, 0.3, 1.0]
            bounds_2seg_b = ([1e-5, 0.1, 0.1], [10.0, 3.0, 3.0])

            popt_2seg_b, _ = curve_fit(fit_wrapper_b, tau_b_fit, msd_b_fit, p0=initial_guess_b, bounds=bounds_2seg_b, maxfev=20000)
            msd_fit_2seg_binned = bkn_pow_2seg(tau_b_fit, *popt_2seg_b, break_b)[0]

        except Exception as e:
            # Not fatal—just skip if binned 2-seg fails
            msd_fit_2seg_binned = None
            popt_2seg_b = None


    
    # make full-length (aligned to tau_b) arrays for plotting/CSV
    msd_fit_single_binned_full = None
    if len(tau_b) > 0:
        msd_fit_single_binned_full = np.full_like(msd_b, np.nan, dtype=float)
        if msd_fit_single_binned is not None:
            msd_fit_single_binned_full[b_fit_mask] = 10**ib * (tau_b_fit ** sb)

    msd_fit_2seg_binned_full = None
    if len(tau_b) > 0:
        msd_fit_2seg_binned_full = np.full_like(msd_b, np.nan, dtype=float)
        if msd_fit_2seg_binned is not None:
            msd_fit_2seg_binned_full[b_fit_mask] = msd_fit_2seg_binned

    # plotting original MSD with fits
    # plt.figure()
    plt.figure(figsize=(6, 4.5)) #to overlay binned

    # plt.plot(t_clean * 0.025, msd_fit_2seg, '--', label=f'2-Seg Fit (α₁ ≈ {popt_2seg[1]:.2f}, α₂ ≈ {popt_2seg[2]:.2f})')

    plt.plot(t_clean * 0.025, msd_clean, label='Original MSD', color='black')
    plt.plot(t_clean * 0.025, msd_fit_single, '--', label=f'Single Power Law (α ≈ {slope_single:.2f})')
    plt.plot(t_clean * 0.025, msd_fit_2seg, '--', label=f'2-Seg Fit (α₁ ≈ {popt_2seg[1]:.2f}, α₂ ≈ {popt_2seg[2]:.2f})')


    # # binned points (with tiny error bars) and binned single power-law
    # if msd_fit_single_binned is not None:
    #     plt.errorbar(tau_b * 0.025, msd_b, yerr=np.clip(se_b, 0, None), fmt='o', ms=4, capsize=2, label='MSD (log-binned)')
    #     plt.plot(tau_b * 0.025, msd_fit_single_binned, '-', lw=1.3, label=f'Single PW (binned) α≈{sb:.2f}')


    # if msd_fit_2seg_binned is not None:
    #     plt.plot(tau_b * 0.025, msd_fit_2seg_binned, '-.', label=f'2-Seg (binned) α₁≈{popt_2seg_b[1]:.2f}, α₂≈{popt_2seg_b[2]:.2f}')

    # binned points + fits (now lengths match tau_b)
    plt.errorbar(tau_b * 0.025, msd_b, yerr=np.clip(se_b, 0, None), fmt='o', ms=4, capsize=2, label='MSD (log-binned)')

    if np.isfinite(sb) and msd_fit_single_binned is not None: 
        plt.plot(tau_b * 0.025, msd_fit_single_binned_full, '-', lw=1.3, label=f'Single PW (binned) α≈{sb:.2f}')

    if (msd_fit_2seg_binned_full is not None) and (popt_2seg_b is not None):
        plt.plot(tau_b * 0.025, msd_fit_2seg_binned_full, '-.', label=f'2-Seg (binned) α₁≈{popt_2seg_b[1]:.2f}, α₂≈{popt_2seg_b[2]:.2f}')


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('MSD')
    plt.title('MSD Fit Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure4: msd_fit_comparison.png", dpi=300)
    # plt.show()

    # to print MSD exponent and magnitude
    # print(f"[Single Power Law] α = {slope_single:.3f}, A = {10**intercept_single:.3e}")
    # print(f"[2-Seg Continuous Fit] α₁ = {popt_2seg[1]:.3f}, A₁ = {popt_2seg[0]:.3e}, α₂ = {popt_2seg[2]:.3f}, A₂ = {A2_2seg:.3e}, Break = {break1}")

    # A: raw curves
    # to create DataFrame with all relevant curves
    fit_df = pd.DataFrame({
        "time_s": t_clean * 0.025,
        "msd_original": msd_clean,
        "msd_fit_single": msd_fit_single,
        "msd_fit_2seg": msd_fit_2seg
    })

    # save to CSV
    fit_df.to_csv("Table 4: msd_fits_comparison.csv", index=False)


    dt = 0.025  # frame interval in seconds

    # B: BINNED curves (only if computed binned series)
    if ("tau_b" in locals()) and (tau_b is not None) and (len(tau_b) > 0):
        binned_cols = {
            "time_s_binned":           tau_b * dt,
            "msd_binned":              msd_b
        }
        # add fits if present
        if msd_fit_single_binned_full is not None:
            binned_cols["msd_fit_single_binned"] = msd_fit_single_binned_full
        if msd_fit_2seg_binned_full is not None:
            binned_cols["msd_fit_2seg_binned"]   = msd_fit_2seg_binned_full


        fit_df_binned = pd.DataFrame(binned_cols)
        fit_df_binned.to_csv("Table 4b: msd_fits_comparison_binned.csv", index=False)

    # C: Parameter table (raw + binned; single + two-seg)
    def _safe(v, idx=None):
        try:
            return float(v if idx is None else v[idx])
        except Exception:
            return np.nan

    def _aic_like(y, yhat, k):
        try:
            if (yhat is None) or (len(y) != len(yhat)) or (len(y) < k+1):
                return np.nan
            resid = np.asarray(y) - np.asarray(yhat)
            sse = float(np.sum(resid**2))
            N = len(y)
            return (2*k + N*np.log(max(sse/N, 1e-300)))
        except Exception:
            return np.nan

    param_rows = []

    # RAW single power-law
    param_rows.append({
        "variant": "raw",
        "model":   "single_pw",
        "alpha1":  _safe(slope_single),
        "alpha2":  np.nan,
        "intercept_log10": _safe(intercept_single),
        "break_index": np.nan,
        "n_points": int(len(t_clean)),
        "tau_min_s": float((t_clean*dt).min()) if len(t_clean) else np.nan,
        "tau_max_s": float((t_clean*dt).max()) if len(t_clean) else np.nan,
        "aic_like": _aic_like(msd_b, msd_fit_single_binned_full, k=2)

    })

    # RAW two-segment
    param_rows.append({
        "variant": "raw",
        "model":   "two_seg_pw",
        "alpha1":  _safe(popt_2seg, 1),
        "alpha2":  _safe(popt_2seg, 2),
        "intercept_log10": np.nan,
        "break_index": _safe(break1),
        "n_points": int(len(t_clean)),
        "tau_min_s": float((t_clean*dt).min()) if len(t_clean) else np.nan,
        "tau_max_s": float((t_clean*dt).max()) if len(t_clean) else np.nan,
        "aic_like": _aic_like(msd_b, msd_fit_2seg_binned_full, k=3)

    })

    # BINNED single power-law
    param_rows.append({
        "variant": "binned",
        "model":   "single_pw",
        "alpha1":  _safe(sb),
        "alpha2":  np.nan,
        "intercept_log10": _safe(ib),
        "break_index": np.nan,
        "n_points": int(len(tau_b)) if ("tau_b" in locals() and tau_b is not None) else 0,
        "tau_min_s": float((tau_b*dt).min()) if ("tau_b" in locals() and len(tau_b)>0) else np.nan,
        "tau_max_s": float((tau_b*dt).max()) if ("tau_b" in locals() and len(tau_b)>0) else np.nan,
        "aic_like":  _aic_like(msd_b, msd_fit_single_binned, k=2) if ("msd_fit_single_binned" in locals() and msd_fit_single_binned is not None) else np.nan,
    })

    # BINNED two-segment
    param_rows.append({
        "variant": "binned",
        "model":   "two_seg_pw",
        "alpha1":  _safe(popt_2seg_b, 1) if ("popt_2seg_b" in locals()) else np.nan,
        "alpha2":  _safe(popt_2seg_b, 2) if ("popt_2seg_b" in locals()) else np.nan,
        "intercept_log10": np.nan,
        "break_index": _safe(break_b) if ("break_b" in locals()) else np.nan,
        "n_points": int(len(tau_b)) if ("tau_b" in locals() and tau_b is not None) else 0,
        "tau_min_s": float((tau_b*dt).min()) if ("tau_b" in locals() and len(tau_b)>0) else np.nan,
        "tau_max_s": float((tau_b*dt).max()) if ("tau_b" in locals() and len(tau_b)>0) else np.nan,
        "aic_like":  _aic_like(msd_b, msd_fit_2seg_binned, k=3) if ("msd_fit_2seg_binned" in locals() and msd_fit_2seg_binned is not None) else np.nan,
    })

    params_df = pd.DataFrame(param_rows)
    params_df.to_csv("Table 4c: msd_fit_params.csv", index=False)

    all_data = []

    if len(tracks) >= 10:
        print("Plotting MSD for 10 longest trajectories...")

        # to sort by trajectory length and pick 10 longest
        traj_lengths = [(i, len(traj)) for i, traj in enumerate(msd_sum)]
        longest_indices = sorted(traj_lengths, key=lambda x: x[1], reverse=True)[:10]

        # preparing plot
        plt.figure(figsize=(8, 6))
        msd_longest = []
        msd_longest_fit_single = []
        msd_longest_fit_2seg = []
        traj_ids = []
        time_all = []

        for idx, _ in longest_indices:
            msd = msd_sum[idx]
            tIme_t = np.arange(len(msd)) * dt

            # clean MSD: remove NaNs and nonpositive values; just to ensure
            mask = ~np.isnan(msd) & (msd > 0)
            t_clean = tIme_t[mask]
            msd_clean = np.array(msd)[mask]

            if len(msd_clean) < 5:
                continue

            # single power-law fit
            slope_single, intercept_single = single_powerlaw_fit(msd_clean)
            msd_fit_single = 10**intercept_single * (t_clean ** slope_single)

            # # two-segment power-law fit
            # break1 = find_turning_point(msd_clean)
            # A_guess = np.mean(msd_clean[:5])
            # initial_guess = [A_guess, 0.3, 1.0]
            # bounds_2seg = ([1e-5, 0.1, 0.1], [10, 3.0, 3.0])

            # def fit_wrapper(x, A, alpha1, alpha2):
            #     return bkn_pow_2seg(x, A, alpha1, alpha2, break1)[0]

            # try:
            #     popt_2seg, _ = curve_fit(fit_wrapper, t_clean, msd_clean, p0=initial_guess, bounds=bounds_2seg)
            #     msd_fit_2seg, _ = bkn_pow_2seg(t_clean, *popt_2seg, break1)
            # except Exception as e:
            #     print(f"Fit failed for trajectory {idx}: {e}")
            #     continue

            # storing for CSV
            for t, o, s, d in zip(t_clean, msd_clean, msd_fit_single, msd_fit_2seg):
                all_data.append({
                    "trajectory": idx,
                    "time_s": t,
                    "msd_original": o,
                    "msd_fit_single": s,
                    # "msd_fit_2seg": d
                })

            # plot
            plt.loglog(t_clean, msd_clean, label=f'Traj {idx}', alpha=0.5)
            plt.loglog(t_clean, msd_fit_single, '--', alpha=0.7)
            # plt.loglog(t_clean, msd_fit_2seg, '--', alpha=0.7)

        plt.xlabel("Time (s)")
        plt.ylabel("MSD (μm²)")
        plt.title("MSD + Fits for 10 Longest Trajectories")
        plt.legend(fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig("Figure 27: longest_MSD_with_fits.png", dpi=300)

        # to save CSV
        df = pd.DataFrame(all_data)
        df.to_csv("Table 27: longest_msd_fits.csv", index=False)
    
    #segments, now using msd avg. (mean) - filtered
    # plot for different trajectories
    # plotting a few individual time-averaged MSD trajectories


    # Def. the cutoff for 1 second (in frames)
    # max_frames = int(1.0 / 0.025)  # = 40

    # storages for histogram
    alpha1_vals = []
    alpha2_vals = []
    turning_pts = []
    A1_vals = []
    A2_vals = []

    plt.figure()

    # counters
    single = 0
    double = 0


    # devugging
    total_tracks = 0             # Total number of tracks processed
    skipped_short = 0            # Skipped due to too few points or bad MSD
    skipped_fit_fail = 0         # Skipped due to curve_fit failure
    skipped_class_unknown = 0    # Skipped due to unrecognized classification


    # filter parameters storage for only double power-law classified tracks
    alpha1_double = []
    alpha2_double = []
    A1_double = []
    A2_double = []
    turning_pts_double = []

    # to save:
    # creating a list to store each track's DataFrame
    all_fit_data = []

    # to plot all singlr/double traj separately:
    single_group = []
    double_group = []
    # return storages for van hove
    single_trajs = []
    double_trajs = []

    # to save fir csv
    trackmate_single_all = []  # to store ALL single track data
    trackmate_double_all = []  # to store ALL double track data





    # for i, msd in enumerate(msd_sum[:70]): #random 5 trajectories, ->change to see more
    for i, msd in enumerate(msd_sum): #!!!!!!!!!!!! ONLY FOR GRAHAM, pls don't try to run at your own computer/laptop
        # msd_trimmed_unfiltered = msd[:max_frames]
        msd_trimmed_unfiltered = msd
        # Filter valid indices
        valid_mask = ~np.isnan(msd_trimmed_unfiltered) & ~np.isinf(msd_trimmed_unfiltered)

        
        # slice the MSD to include only values within 1 s
        # plt.plot(msd[:max_frames], label=f"Track {i+1}")
        t_unfiltered = np.arange(1, len(msd_trimmed_unfiltered) + 1) #start
        t = t_unfiltered[valid_mask]
        msd_trimmed = msd_trimmed_unfiltered[valid_mask]

        total_tracks += 1

        # Optional: skip short or empty tracks
        if len(msd_trimmed) < 10 or np.any(np.isnan(msd_trimmed)) or np.any(msd_trimmed <= 0):  
            skipped_short += 1
            raise ValueError("Not enough valid data points.")

        

        try:
            # Choose fixed breakpoint, e.g. at index 20
            # break1 = 20
            break1 = find_turning_point(msd_trimmed) #finding turning point -> largest slope (derivative)

            # Initial guesses: A = first value, alpha1 and alpha2 as trial slopes
            A_guess = np.mean(msd_trimmed[:5])
            initial_guess = [A_guess, 0.3, 1.0]

            # Bounds: A positive, slopes between 0.1–3.0
            bounds = (
                [1e-5, 0.1, 0.1],
                [10, 3.0, 3.0]
            )

            def fit_wrapper(x, A, alpha1, alpha2):
                return bkn_pow_2seg(x, A, alpha1, alpha2, break1)[0]

            # fits the data
            popt, _ = curve_fit(fit_wrapper, t, msd_trimmed, p0=initial_guess, bounds=bounds)

            # to get full result with A2
            msd_fit_2seg, A2 = bkn_pow_2seg(t, *popt, break1)

            # Store all results for histograms
            A1 = popt[0]
            alpha1 = popt[1]
            alpha2 = popt[2]

            alpha1_vals.append(alpha1)
            alpha2_vals.append(alpha2)
            A1_vals.append(A1)
            A2_vals.append(A2)
            turning_pts.append(break1)


            # check if 2 segment fit applies
            classification = classify_powerlaw_type(alpha1, alpha2)
            # print(f"Track {i+1} classified as {classification} power law. Alpha_1 is:{alpha1:.2f} and Alpha2 is:{alpha2:.2f}")
            # INSIDE the per-trajectory loop, just after you set `classification`


            # Compute single-pw slope per trajectory (so singles can be fast/slow too)
            try:
                alpha_single, _ = single_powerlaw_fit(msd_trimmed)
            except Exception:
                alpha_single = np.nan

            tau_break_s = float(break1 * 0.025) if np.isfinite(break1) else np.nan

            # Single place to record per-trajectory params for bifurcation
            param_rows_traj.append({
                "traj_id":      int(i),
                "model":        "double" if classification == "double" else "single",
                "alpha_single": float(alpha_single),
                "alpha1":       float(alpha1),
                "alpha2":       float(alpha2) if classification == "double" else np.nan,
                "tau_break_s":  tau_break_s
            })

            param_rows_traj.append({
                "traj_id": int(i),                    # i must index tracks_filtered
                "model": "double" if classification == "double" else "single",
                "alpha1": float(alpha1),
                "alpha2": float(alpha2) if classification == "double" else float("nan"),
            })

            if classification == 'single':
                # Single power-law fit
                single += 1
                single_group.append(msd_trimmed) #To separate storage
                single_trajs.append(tracks_filtered[i][:, :2])  # keep full x, y trajectory

                # TrackMate-style storage
                trackmate_single_data = []  # reset for this trajectory
                for frame_idx, (x, y) in enumerate(tracks_filtered[i][:, :2]):
                    trackmate_single_data.append({
                        "TRACK_ID": single,
                        "POSITION_X": x,
                        "POSITION_Y": y,
                        "POSITION_T": frame_idx
                    })
                trackmate_single_all.extend(trackmate_single_data)  # save all

                continue  # Skip the rest of the loop for this track

            elif classification == 'double':
                
                double +=1
                double_group.append(msd_trimmed) #to separate storage
                double_trajs.append(tracks_filtered[i][:, :2])

                # add to new storage
                alpha1_double.append(alpha1)
                alpha2_double.append(alpha2)
                A1_double.append(A1)
                A2_double.append(A2)
                turning_pts_double.append(break1)

                # Plot
                plt.plot(t, msd_trimmed, label=f"Track {i+1} MSD", color="black")
                plt.plot(t, msd_fit_2seg, '--', label=f"2-Seg Fit v.2 α₁ ≈ {popt[1]:.2f}, α₂ ≈ {popt[2]:.2f}. Turning point is: {break1}")

                # to store data in a single dataframe
                df = pd.DataFrame({
                    "track": i + 1,
                    "frame": t,
                    "time_s": t * 0.025,
                    "msd": msd_trimmed,
                    "fit_2seg": msd_fit_2seg
                })
                all_fit_data.append(df)

                # TrackMate-style storage
                trackmate_double_data = []  # reset for this trajectory
                for frame_idx, (x, y) in enumerate(tracks_filtered[i][:, :2]):
                    trackmate_double_data.append({
                        "TRACK_ID": double,
                        "POSITION_X": x,
                        "POSITION_Y": y,
                        "POSITION_T": frame_idx
                    })
                trackmate_double_all.extend(trackmate_double_data)  # save all
            
            else: 
                
                skipped_class_unknown += 1
                continue

        except Exception as e:
            # print(f"Skipping 2-seg fit for Track {i+1}: {e}")
            skipped_fit_fail += 1
            continue


            # pass

        # # Plot trajectory ; coomented for now to save time debugging van Hove
        # plt.plot(t, msd_trimmed, label=f"Track {i+1}")


        # # ! move this to/from the loop for indvidual/group plot
        # # Plot the average
        # plt.plot(t_clean, msd_clean, 'k--', linewidth=2, label="Average MSD (valid)")
        

        # # Log-log and labels
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel("Lag time (frames)")
        # plt.ylabel("MSD")
        # plt.title("Single and Two-segment Fits on MSD Trajectories")
        # plt.legend(fontsize='small', loc='upper left', ncol=2)
        # plt.tight_layout()
        # plt.show()

        # concatenate all dataframes and save as a single file


    params_bif = pd.DataFrame(param_rows_traj)
    fast_trajs, slow_trajs, summary = bifurcate_by_msd(params_bif, tracks_filtered, alpha_fast=0.6) #! JAN appears to be corresponding to Prof.'s Shi variant
    summary.to_csv("Table_35_MSD_bifurcation_summary.csv", index=False)
    print(f"[bifurcation] DONE: {len(fast_trajs)} fast, {len(slow_trajs)} slow")


    # JAN
    # master metrics

    # # lvl 2
    # for _, row in summary.iterrows():
    #     tid = int(row["traj_id"])
    #     metrics[tid]["alpha_eff"] = float(row.get("alpha_eff", np.nan))
    #     metrics[tid]["is_alpha_fast"] = int(row["is_fast"])

    # to make sure indexes are correct:
    # ---- Level 3: attach alpha-bifurcation labels to metrics (bulletproof) ----
    missing_in_metrics = 0
    written = 0

    for _, row in summary.iterrows():
        try:
            tid = int(row.get("traj_id"))
        except Exception:
            continue

        if tid not in metrics:
            missing_in_metrics += 1
            continue  # don't crash long run

        metrics[tid]["alpha_eff"] = float(row.get("alpha_eff", np.nan))

        # be robust to column naming
        if "is_fast" in row:
            metrics[tid]["is_alpha_fast"] = int(row["is_fast"])
        elif "is_alpha_fast" in row:
            metrics[tid]["is_alpha_fast"] = int(row["is_alpha_fast"])
        else:
            metrics[tid]["is_alpha_fast"] = np.nan

        written += 1

    print(f"JAN Master Summary: [alpha->metrics] wrote={written}, missing_metrics_keys={missing_in_metrics}, summary_rows={len(summary)}")


    print("\nProcessing Summary/Debugging:")
    print(f"  Total tracks processed:       {total_tracks}")
    print(f"  ➤ Saved as single:            {single}")
    print(f"  ➤ Saved as double:            {double}")
    print(f"  ➤ Skipped (too short/bad):    {skipped_short}")
    print(f"  ➤ Skipped (fit failed):       {skipped_fit_fail}")
    print(f"  ➤ Skipped (unknown type):     {skipped_class_unknown}")

    # saving final TrackMate-style CSVs
    df_single = pd.DataFrame(trackmate_single_all)
    df_single.to_csv("Table 20_single_tracks_exported.csv", index=False)

    df_double = pd.DataFrame(trackmate_double_all)
    df_double.to_csv("Table 28_double_tracks_exported.csv", index=False)

    print(f"Exported {len(trackmate_single_all)} single track points.")
    print(f"Exported {len(trackmate_double_all)} double track points.")

    
    if all_fit_data:
        full_df = pd.concat(all_fit_data, ignore_index=True)
        full_df.to_csv("Table 5: all_tracks_msd_fits.csv", index=False)
    else:
        print("No valid fit data to save.")

    print (f"There are {single} single tracks and {double} double tracks")


    # to save histogram data
    df_hist = pd.DataFrame({
        "alpha1": alpha1_double,
        "alpha2": alpha2_double,
        "A1": A1_double,
        "A2": A2_double,
        "turning_point": turning_pts_double
    })

    df_hist.to_csv("Table 6: two_segment_fit_histogram_data.csv", index=False)


    # to plot histograms
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()

    sns.histplot(alpha1_double, bins=15, ax=axs[0], kde=True)
    axs[0].set_title("α₁ (First Segment Slopes)")
    axs[0].set_xlabel("α₁")
    axs[0].set_ylabel("Count")

    sns.histplot(alpha2_double, bins=15, ax=axs[1], kde=True)
    axs[1].set_title("α₂ (Second Segment Slopes)")
    axs[1].set_xlabel("α₂")
    axs[1].set_ylabel("Count")

    sns.histplot(turning_pts_double, bins=15, ax=axs[2], kde=False)
    axs[2].set_title("Turning Points")
    axs[2].set_xlabel("Frame Index")
    axs[2].set_ylabel("Count")

    sns.histplot(A1_double, bins=15, ax=axs[3], kde=True)
    axs[3].set_title("A₁ (Initial Intercepts)")
    axs[3].set_xlabel("A₁")
    axs[3].set_ylabel("Count")

    sns.histplot(A2_double, bins=15, ax=axs[4], kde=True)
    axs[4].set_title("A₂ (Adjusted Intercepts)")
    axs[4].set_xlabel("A₂")
    axs[4].set_ylabel("Count")

    axs[5].axis('off')  # Empty plot area

    plt.tight_layout()
    plt.suptitle("Distributions of Two-Segment Fit Parameters Across Trajectories", y=1.03)
    # Save figure to file
    fig.savefig("Figure 6: two_segment_fit_histograms.png", dpi=300, bbox_inches='tight')
    # plt.show()


    # plotting single msds
    single_rows = []
    plt.figure(figsize=(8, 5))
    for traj_id, msd in enumerate(single_group):
        t = np.arange(1, len(msd) + 1)
        try:
            slope, intercept = single_powerlaw_fit(msd)
            fit = 10**intercept * (t ** slope)

            # plot
            plt.plot(t * 0.025, msd, color='gray', alpha=0.3)
            plt.plot(t * 0.025, fit, '--', color='blue', alpha=0.4)

            # to save to storage
            for i in range(len(msd)):
                single_rows.append({
                    "trajectory_id": traj_id,
                    "time_s": t[i] * 0.025,
                    "msd_original": msd[i],
                    "msd_fit_single": fit[i]
                })

        except Exception:
            continue

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("MSD")
    plt.title("Single Power-Law–Like Trajectories (Fitted)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figure 8_Single Power-Law–Like Trajectories.png", dpi=300)
    # plt.show()

    # to save single group to CSV
    df_single = pd.DataFrame(single_rows)
    df_single.to_csv("Table 8_single_powerlaw_fits.csv", index=False)

    # double
    double_rows = []

    plt.figure(figsize=(8, 5))
    for traj_id, msd in enumerate(double_group):
        t = np.arange(1, len(msd) + 1)
        try:
            break1 = find_turning_point(msd)
            A_guess = np.mean(msd[:5])
            initial_guess = [A_guess, 0.3, 1.0]
            bounds = ([1e-5, 0.1, 0.1], [10, 3.0, 3.0])

            def fit_wrapper(x, A, alpha1, alpha2):
                return bkn_pow_2seg(x, A, alpha1, alpha2, break1)[0]

            popt, _ = curve_fit(fit_wrapper, t, msd, p0=initial_guess, bounds=bounds)
            fit, _ = bkn_pow_2seg(t, *popt, break1)

            # plot
            plt.plot(t * 0.025, msd, color='gray', alpha=0.3)
            plt.plot(t * 0.025, fit, '--', color='red', alpha=0.4)

            # to save to storage
            for i in range(len(msd)):
                double_rows.append({
                    "trajectory_id": traj_id,
                    "time_s": t[i] * 0.025,
                    "msd_original": msd[i],
                    "msd_fit_2seg": fit[i]
                })

        except Exception:
            continue

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("MSD")
    plt.title("Two-Segment Power-Law–Like Trajectories (Fitted)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Figure 9: Two-Segment Power-Law–Like Trajectories.png", dpi=300)
    # plt.show()

    # to save double group to CSV
    df_double = pd.DataFrame(double_rows)
    df_double.to_csv("Table 9_double_powerlaw_fits.csv", index=False)

    # entire_single/double traj:
    # single
    max_len = max(map(len, single_group))
    msd_mat = np.full((len(single_group), max_len), np.nan)
    for i, msd in enumerate(single_group):
        msd_mat[i, :len(msd)] = msd
    mean_msd = np.nanmean(msd_mat, axis=0)
    t = np.arange(1, len(mean_msd) + 1)

    slope, intercept = single_powerlaw_fit(mean_msd)
    fit_single = 10**intercept * (t ** slope)

    plt.figure(figsize=(8, 5))
    plt.plot(t * 0.025, mean_msd, 'k-', lw=2, label="Mean MSD (Single)")
    plt.plot(t * 0.025, fit_single, '--b', lw=2, label=f"Fit α ≈ {slope:.2f}")
    plt.xscale("log"), plt.yscale("log")
    plt.xlabel("Time (s)"); plt.ylabel("MSD")
    plt.title("Single‑Power‑Law Group")
    plt.legend(); plt.grid(True); plt.tight_layout()
    # plt.show()
    plt.savefig("Figure 10: Mean MSD (single).png", dpi=300)

    df = pd.DataFrame({
        "time_s": t * 0.025,
        "msd_mean": mean_msd,
        "msd_fit": fit_single
    })
    df.to_csv("Table 10_avg_single_group.csv", index=False)

    # double
    max_len = max(map(len, double_group))
    msd_mat = np.full((len(double_group), max_len), np.nan)
    for i, msd in enumerate(double_group):
        msd_mat[i, :len(msd)] = msd
    mean_msd = np.nanmean(msd_mat, axis=0)
    t = np.arange(1, len(mean_msd) + 1)

    break1 = find_turning_point(mean_msd)
    A_guess = np.mean(mean_msd[:5])
    bounds = ([1e-5, 0.1, 0.1], [10, 3.0, 3.0])
    popt, _ = curve_fit(
        lambda x, A, a1, a2: bkn_pow_2seg(x, A, a1, a2, break1)[0],
        t, mean_msd, p0=[A_guess, 0.3, 1.0], bounds=bounds
    )
    fit_double, _ = bkn_pow_2seg(t, *popt, break1)

    plt.figure(figsize=(8, 5))
    plt.plot(t * 0.025, mean_msd, 'k-', lw=2, label="Mean MSD (Double)")
    plt.plot(t * 0.025, fit_double, '--r', lw=2, label=f"Fit α₁ ≈ {popt[1]:.2f}, α₂ ≈ {popt[2]:.2f}")
    plt.xscale("log"), plt.yscale("log")
    plt.xlabel("Time (s)"); plt.ylabel("MSD")
    plt.title("Two‑Segment Power‑Law Group")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("Figure 11: Mean MSD (double).png", dpi=300)
    # plt.show()

    df = pd.DataFrame({
        "time_s": t * 0.025,
        "msd_mean": mean_msd,
        "msd_fit": fit_double
    })
    df.to_csv("Table 11_avg_double_group.csv", index=False)



    




    return tracks_filtered, single_trajs, double_trajs


# to ensure the path exists (and has needed file type)
def check_folder(path):
    """
    Ensures if the path exists, checks if needed .csv files are in selected folder.

    Args:
        path (str): Path to the folder containing CSV files.

    Returns:
        (str): Found files with first line printed. Raises ValueError if something went wrong.
    """

    if not os.path.exists(path):
        raise ValueError(f"Invalid path. The folder '{path}' does not exist.")

    else:
        print(f"Listing .csv files in '{path}':\n")

        for file_name in os.listdir(path):
            # constructing the full file path
            file_path = os.path.join(path, file_name)
            # check for .csv files only
            if os.path.isfile(file_path) and file_path.endswith(".csv"):
                print(f"Found CSV file: {file_name}")

                # check if the file exists
                if os.path.exists(file_path):
                    print(f"Reading file: {file_name}")
                    # read and print the first line (for demonstration)
                    with open(file_path, "r") as file:
                        first_line = file.readline().strip()
                        print(f"First line: {first_line}\n")
                else:
                    raise ValueError(".csv files not found. Please check if the folder is chosen correctly.")
                

#test
path = os.getcwd()
check_folder(path)

# CalcMSD(path)

# gausssian fit

def gauss(x, H, A, x0, sigma): 
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))
    H0 = min(y)
    A0 = max(y) - H0
    popt, pcov = curve_fit(gauss, x, y, p0=[H0, A0, mean, sigma])
    return popt


def convert_tracks_to_df(tracks):
    """
    Converts list of 2D numpy arrays to a trackpy-compatible DataFrame.

    Each track is assumed to be of shape (T x 2), with columns (x, y).
    """
    records = []

    # to check how many empty tracks
    empty = 0
    valid =0
    for pid, traj in enumerate(tracks):
        # if traj.shape[0] < 2 or np.isnan(traj[:, 0]).all():
        #     empty+=1
        #     continue  # skip short or empty x-trajectories

        x_positions = traj[:, 0] #for filter
        if (
            traj.shape[0] < 2 or
            np.allclose(x_positions, x_positions[0])  # i.e., no change over time
        ):
            empty += 1
            continue

        else:
            valid+=1
            for frame_idx, (x, y) in enumerate(traj[:, :2]):
                records.append({'frame': frame_idx, 'particle': pid, 'x': x, 'y': y})
    
    print(f"{empty} out of {valid} empty tracks were in trajectories used to plot van Hove corrrelation")
    return pd.DataFrame(records)

# alternative Methods of Cage hopping Detection #UPD. Replaced by a new version (below)
# "Many other robust estimators of location and scatter have been presented in the literature. The first such estimator was proposed by Stahel15 and Donoho16 (see also Ref 17). 
# They defined the socalled Stahel–Donoho outlyingness of a data point" - Robust statistics for outlierdetection Peter J. Rousseeuw and Mia Hubert

# const
EPS = 1e-12 # small epsilon to prevent division by zero
MAD_SCALE = 1.4826  # to make MAD comparable to std under Gaussian (wiki) / Scale factor so MAD ≈ standard deviation for normal data

# helpers

def mad(x):
    """
    To compute Median Absolute Deviation (MAD):
    a robust estimate of variability that is less sensitive to outliers than std.
    """
    med = np.median(x)                       # to find the median (robust center)
    return np.median(np.abs(x - med))        # takes median of absolute deviations



def robust_z_univariate(x):
    """
    To compute robust z-scores using median and MAD. (from article)
    |x - median| / (1.4826*MAD + eps)
    Values > ~3 usually indicate 'outliers' or rare events (e.g. hopping).
    """
    x = np.asarray(x, dtype=float)           # to ensure it's a numeric NumPy array
    med = np.median(x)                       # Robust center of distribution
    s = MAD_SCALE * mad(x) + EPS             # Robust scale (like std), add eps to avoid /0
    return np.abs(x - med) / s               # Robust z-score (magnitude of deviation)

def segment_outlier_fraction(log_rg_segments, k=3.0):
    """
    To calculate the fraction of trajectory segments that are unusually large in log(Rg).
    Biological idea: how often does a nucleosome escape its cage?
    """
    x = np.asarray(log_rg_segments, dtype=float)     # to nsure array form
    med = np.median(x)                               # typical cage size (in log-space)
    z = robust_z_univariate(x)                       # robust deviation for each segment
    # Count how many segments exceed threshold k AND are above median (right-tail)
    return np.mean((z > k) & (x > med)) if x.size else 0.0

def traj_level_features(log_rg_segments):
    """
    To compute per-trajectory summary: 'location' (median log Rg) and 'scatter' (MAD).
    These correspond to the Stahel–Donoho 'location' and 'scatter' parameters.
    """
    x = np.asarray(log_rg_segments, dtype=float)
    return np.median(x), MAD_SCALE * mad(x)          # return (location, scatter)

def robust_mahalanobis_2d(features): #need because of eq. 7 in main source
    """
    To compute a simple robust Mahalanobis-like distance across trajectories
    in the 2D (location, scatter) space using coordinate-wise medians and MADs.
    Biological meaning: how unusual a trajectory is in both its cage size (location)
    and internal variability (scatter) compared to the population.
    """
    Z = np.asarray(features, dtype=float)            # shape (M, 2)
    med = np.median(Z, axis=0)                       # robust mean per column
    scale = MAD_SCALE * np.array([mad(Z[:,0]), mad(Z[:,1])]) + EPS  # scales per axis
    Zs = (Z - med) / scale                           # to normalize deviations per coordinate
    D = np.sqrt(np.sum(Zs**2, axis=1))               # Euclidean/decartian distance in norm. space
    return D

# MAIN CLASSIFIER

def classify_hoppers(Rg_segments_per_traj, k=3.0, p=0.1, use_global=False):
    """
    To identify cage-hopping trajectories using robust stats.
    
    Parameters:
    Rg_segments_per_traj (list of 1D arrays):
        Each array contains Rg values for segments of a single trajectory.
    k (float):
        Threshold on robust z-score (≈ 3 corresponds to 99.7% cutoff for Gaussian data).
    p (float):
        Minimum fraction of large-Rg segments to call a trajectory a hopper.
    use_global (bool):
        Whether to include global comparison (across trajectories) using robust Mahalanobis distance.
    
    Returns:
    --------
    DataFrame with:
        traj   : trajectory index
        frac_hi: fraction of high-Rg segments
        med    : median log(Rg)
        mad    : MAD log(Rg)
        D      : robust 2D distance (if use_global)
        is_hopper : boolean label (True if classified as hopping)
    """
    rows = []    # to store per-trajectory metrics
    feats = []   # to store (location, scatter) features for later global comparison

    for idx, rg in enumerate(Rg_segments_per_traj):
        rg = np.asarray(rg, dtype=float)                          # converts to array
        rg = rg[np.isfinite(rg) & (rg > 0)]                       # filters out NaNs/negatives

        if rg.size < 3:                                           # if too short for robust stats
            rows.append(dict(traj=idx, frac_hi=0.0, med=np.nan, mad=np.nan,
                             D=np.nan, is_hopper=False))
            feats.append([np.nan, np.nan])
            continue

        x = np.log(rg)                                            # Work in log-space
        frac_hi = segment_outlier_fraction(x, k=k)                # Fraction of “escaped” segments
        loc, sca = traj_level_features(x)                         # to get (median, scatter)
        feats.append([loc, sca])                                  # to store feature pair
        rows.append(dict(traj=idx, frac_hi=frac_hi, med=loc, mad=sca,
                         D=np.nan, is_hopper=False))

    df = pd.DataFrame(rows)                                       # converts to table

    # opt. global step: compare all trajectories jointly in 2D (location, scatter) space
    if use_global and len(Rg_segments_per_traj) >= 5:
        feats_arr = np.array(feats, dtype=float)                  # to convert to NumPy array
        mask = np.all(np.isfinite(feats_arr), axis=1)             # to keep valid ones only
        D = np.full(len(feats_arr), np.nan)                       # preallocate
        if np.sum(mask) >= 5:
            D[mask] = robust_mahalanobis_2d(feats_arr[mask])      # computes distances
        df["D"] = D                                               # sdds to DataFrame

        # Chi-square 97.5% threshold for df=2 → sqrt(7.3778) ≈ 2.718
        df["is_hopper"] = (df["frac_hi"] >= p) | (df["D"] > np.sqrt(7.3778))
    else:
        df["is_hopper"] = (df["frac_hi"] >= p)

    return df

# simplified version to better use inside van Hove function
# helper
def rg_of_segment(xy):
    # radius of gyration (2D) for points in rows of xy
    c = xy.mean(axis=0)
    dif = xy - c
    return np.sqrt((dif**2).sum(axis=1).mean())

def robust_rg_hopper_split(tracks, seg_size=10, alpha=0.975):
    """
    To plit trajectories into 'hoppers' vs 'non-hoppers' using a robust
    2D feature + robust (diagonal) Mahalanobis distance.

    Per trajectory m:
      z1 = median(log Rg_seg),  z2 = 1.4826 * MAD(log Rg_seg)
      D_m^2 = sum_j ((z_mj - med_j) / (1.4826*MAD_j))^2
    Hopper if D_m^2 > chi2_{p=2, alpha}  (≈ 7.377 for alpha=0.975)

    Returns: hoppers, non_hoppers, diagnostics_df
    """

    # 1) build per-trajectory log Rg segment series
    feats = []      # (z1, z2)
    keep_idx = []   # indices of trajectories that produced features
    all_logRg = []  # store the logRg arrays to avoid recomputation
    for idx, traj in enumerate(tracks):
        if traj is None or traj.shape[0] < seg_size:
            continue
        # sliding, non-overlapping segments of length seg_size
        nseg = traj.shape[0] // seg_size
        if nseg < 1:
            continue
        rg_vals = []
        for k in range(nseg):
            seg = traj[k*seg_size:(k+1)*seg_size, :2]
            if np.isnan(seg).any():
                continue
            rg_vals.append(rg_of_segment(seg))
        rg_vals = np.array(rg_vals, float)
        rg_vals = rg_vals[np.isfinite(rg_vals) & (rg_vals > 0)]
        if rg_vals.size == 0:
            continue
        logRg = np.log(rg_vals)
        med = np.median(logRg)
        mad = np.median(np.abs(logRg - med))
        z1 = med
        z2 = 1.4826 * mad
        feats.append([z1, z2])
        keep_idx.append(idx)
        all_logRg.append(logRg)

    if len(feats) == 0:
        return [], [], None  # nothing usable

    Z = np.asarray(feats)                     # shape (M,2)
    col_meds = np.median(Z, axis=0)          # robust center
    col_mads = 1.4826 * np.median(np.abs(Z - col_meds), axis=0)
    col_mads[col_mads == 0] = 1e-12          # guard

    # robust (diagonal) Mahalanobis distance
    RD2 = np.sum(((Z - col_meds) / col_mads)**2, axis=1)

    # chi2_{2, alpha}
    # 0.975 → ~7.377; 0.99 → ~9.210; 0.95 → ~5.991
    chi2_cut = {0.95: 5.991, 0.975: 7.377, 0.99: 9.210}.get(alpha, 7.377)

    # we can now call the function with alpha=0.95, alpha=0.975, or alpha=0.99.

    # If someone forgets to pass alpha, it defaults to 7.377 (97.5%), which matches your literature.

    # adding new thresholds later becomes trivial (e.g., 0.999 → 13.82).

    hoppers_idx = [keep_idx[i] for i in range(len(keep_idx)) if RD2[i] > chi2_cut]
    nonhop_idx  = [keep_idx[i] for i in range(len(keep_idx)) if RD2[i] <= chi2_cut]

    hoppers = [tracks[i] for i in hoppers_idx]
    nonhop  = [tracks[i] for i in nonhop_idx]

    # opt. diagnostics to compare with your old rule
    diag = pd.DataFrame({
        # "traj_index": keep_idx,
        # JAN: metrix, same index
        "traj_id": keep_idx,
        "z1_med_logRg": Z[:,0],
        "z2_MAD_logRg": Z[:,1],
        "RD2": RD2,
        "is_hopper": RD2 > chi2_cut
    })

    # NEW: print how many hoppers you’d get for each standard cutoff
    print_rg_threshold_summary(diag)
    
    return hoppers, nonhop, diag


# to see how many traj pass each hopping threshold (%)
def print_rg_threshold_summary(diag):
    """
    Print how many trajectories would be called 'hoppers' for several RD^2 cutoffs.
    Expects a DataFrame `diag` with a column 'RD2'.
    """
    chi2_table = {
        0.95: 5.991,   # χ²₂,0.95
        0.975: 7.377,  # χ²₂,0.975
        0.99: 9.210    # χ²₂,0.99
    }

    if "RD2" not in diag.columns:
        print("[rg-threshold] diag has no 'RD2' column, skipping summary.")
        return

    total = len(diag)
    if total == 0:
        print("[rg-threshold] diag is empty, skipping summary.")
        return

    print("[rg-threshold] Mahalanobis RD^2 counts per cutoff:")
    for alpha, cut in chi2_table.items():
        n = int((diag["RD2"] > cut).sum())
        frac = n / total
        print(f"  alpha={alpha:.3f}, cutoff={cut:.3f}: "
              f"{n}/{total} trajectories ({frac:.1%} hoppers)")


# helper to modify vanHove to take not only 3 integer vals
# def _normalize_lags(lags_to_plot, dt=0.025, max_frames=None, units="frames"):
#     """
#     Convert user-specified lags (frames or seconds) to valid integer frame lags.
#     - lags_to_plot: list of numbers (frames if units='frames', seconds if units='seconds')
#     - dt: seconds per frame (used only when units='seconds')
#     - max_frames: optional cap; if given we clip to <= max_frames-1
#     - returns: sorted unique np.array of integer lags >= 1
#     """
#     import numpy as np

#     arr = np.asarray(lags_to_plot, dtype=float)
#     if units.lower().startswith("sec"):
#         # seconds -> frames
#         arr = np.round(arr / float(dt))

#     # frames as int, valid (>=1)
#     arr = arr.astype(int)
#     arr = arr[arr >= 1]
#     if max_frames is not None:
#         arr = arr[arr < int(max_frames)]  # need at least lag+1 samples

#     # unique + sorted
#     arr = np.unique(arr)
#     return arr

# same helper but works for any type of input provided: v3. now i don't need separate helpers for different functions
def normalize_lags(
    lag_spec,
    dt=0.025,
    units="auto",          # "auto" | "frames" | "seconds"
    max_frames=None,       # clip to < max_frames (need at least lag+1 samples)
    return_labels=True,    # also return legend strings
    label_seconds_precision=3
):
    """
    To normalize user-specified lags to valid integer frame lags (>=1), optionally
    producing legend-ready labels that include BOTH seconds and frames.

    Parameters:
        lag_spec : iterable (float or int)
            Lags specified either in frames or seconds (or mixed).
        dt (float):
            Seconds per frame.
        units (str):
            "auto"   : ints→frames, floats→seconds
            "frames" : treat all values as frames
            "seconds": treat all values as seconds
        max_frames (int or None):
            If given, drop lags >= max_frames (need lag+1 samples).
        return_labels (bool):
            If True, also return labels like "Δt = 0.100s (4 fr)".
        label_seconds_precision (int):
            Decimal places for seconds in labels.

    Returns
    -------
    frames : array [int]
        Sorted unique integer frame lags (>=1).
    labels : list[str] or None
        Legend strings for each frame lag (if return_labels=True).
    """

    # to array (keeps NaNs out)
    arr = np.asarray(list(lag_spec), dtype=float)
    if arr.size == 0:
        return np.array([], dtype=int), ([] if return_labels else None)

    # decide how to interpret inputs
    if units == "auto":
        # integers → frames; non-integers → seconds
        is_intlike = np.isclose(arr, np.round(arr))
        frames = np.round(arr[is_intlike]).astype(int)
        secs   = arr[~is_intlike]
        frames_from_secs = np.round(secs / float(dt)).astype(int)
        frames = np.concatenate([frames, frames_from_secs])
    elif units == "frames":
        frames = np.round(arr).astype(int)
    elif units == "seconds":
        frames = np.round(arr / float(dt)).astype(int)
    else:
        raise ValueError("units must be 'auto', 'frames', or 'seconds'")

    # enforce valid range (>=1)
    frames = frames[frames >= 1]
    if frames.size == 0:
        return np.array([], dtype=int), ([] if return_labels else None)

    # optional upper clip
    if max_frames is not None:
        frames = frames[frames < int(max_frames)]

    # unique + sorted
    frames = np.unique(frames)

    if not return_labels:
        return frames, None

    # build legend labels with both units
    sec_vals = frames * float(dt)
    fmt = f"{{:.{label_seconds_precision}f}}"
    labels = [f"Δt = {fmt.format(s)}s ({f} fr)" for s, f in zip(sec_vals, frames)]
    return frames, labels


# test helpers to check van Hove plots

def pdf_from_values(values, bin_edges):
    """Width-normalized PDF (works for linear or log bins)."""
    counts, edges = np.histogram(values, bins=bin_edges)  # no density=True
    widths = np.diff(edges)
    pdf = counts / (counts.sum() * widths)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, pdf

def vh_consistency_check_v1(dx, bins_linear=400, xlim_abs=15.0):
    dx = np.asarray(dx)

    # Two-sided (signed); linear bins just for display range
    edges_signed = np.linspace(-xlim_abs, xlim_abs, bins_linear + 1)
    xc_signed, pdf_signed = pdf_from_values(dx, edges_signed)

    # One-sided (absolute); log bins to show tails (same max range)
    abs_dx = np.abs(dx)
    lo = max(abs_dx[abs_dx > 0].min(), 1e-3)
    hi = min(np.percentile(abs_dx, 99), xlim_abs)
    edges_abs_log = np.logspace(np.log10(lo), np.log10(hi), bins_linear + 1)
    xc_abs, pdf_abs = pdf_from_values(abs_dx, edges_abs_log)

    #Numeric checks
    # A) PDFs integrate to ~1
    i_signed = np.sum(pdf_signed * np.diff(edges_signed))
    i_abs    = np.sum(pdf_abs    * np.diff(edges_abs_log))
    print(f"[norm] ∫P(Δx)dx ≈ {i_signed:.5f}, ∫P(|Δx|)d|x| ≈ {i_abs:.5f}")

    # B) Plateau ratio near 0: one-sided should be ~ 2× two-sided (same units)
    mask_plateau_signed = (np.abs(xc_signed) <= 0.2)
    plateau_signed = pdf_signed[mask_plateau_signed].mean() if mask_plateau_signed.any() else np.nan

    mask_plateau_abs = (xc_abs <= 0.2)
    plateau_abs = pdf_abs[mask_plateau_abs].mean() if mask_plateau_abs.any() else np.nan

    print(f"[plateau] two-sided ~ {plateau_signed:.3e}, one-sided ~ {plateau_abs:.3e}, ratio (abs / signed) ≈ {plateau_abs/plateau_signed:.2f}")

    # C) Symmetry of two-sided PDF
    # (quick sanity: P(+x) ~ P(-x))
    mid = len(xc_signed)//2
    sym_n = min(mid, len(xc_signed)-mid-1, 50)
    if sym_n > 5:
        left  = pdf_signed[mid-sym_n:mid][::-1]
        right = pdf_signed[mid+1:mid+1+sym_n]
        sym_err = np.mean(np.abs(left-right)/(left+right+1e-30))
        print(f"[symmetry] mean relative asymmetry near center ≈ {sym_err:.3f}")

    # Plots from the same PDFs
    # 1) Two-sided: linear X, log Y
    plt.figure(figsize=(7.0, 4.5))
    plt.semilogy(xc_signed, pdf_signed, label="two-sided P(Δx)")
    plt.xlabel("Scaled Δx (signed)")
    plt.ylabel("P(Δx)")
    plt.title("Van Hove two-sided (same PDF; semi-log display)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("vh_check_two_sided_semilog.png", dpi=200)
    plt.close()

    # 2) One-sided: log-log
    plt.figure(figsize=(7.0, 4.5))
    plt.loglog(xc_abs, pdf_abs, label="one-sided P(|Δx|)")
    plt.xlabel("Scaled |Δx|")
    plt.ylabel("P(|Δx|)")
    plt.title("Van Hove one-sided (same PDF; log-log display)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("vh_check_one_sided_loglog.png", dpi=200)
    plt.close()

    return (xc_signed, pdf_signed, xc_abs, pdf_abs)

def vh_consistency_check(dx, bins_linear=400, xlim_abs=15.0):

    dx = np.asarray(dx)
    edges_lin = np.linspace(-xlim_abs, xlim_abs, bins_linear + 1)  # SAME bins

    # signed PDF (two-sided) on linear bins
    cs, es = np.histogram(dx, bins=edges_lin)
    ws = np.diff(edges_lin)
    pdf_signed = cs / np.maximum(cs.sum() * ws, 1e-300)
    xc_signed = 0.5 * (edges_lin[:-1] + edges_lin[1:])

    # One-sided PDF derived in TWO ways using SAME binning near 0:
    # (A) Direct abs on linear bins
    abs_dx = np.abs(dx)
    edges_abs_lin = np.linspace(0.0, xlim_abs, bins_linear//2 + 1)
    ca, ea = np.histogram(abs_dx, bins=edges_abs_lin)
    wa = np.diff(ea)
    pdf_abs_direct = ca / np.maximum(ca.sum() * wa, 1e-300)
    xc_abs = 0.5 * (ea[:-1] + ea[1:])

    # (B) Fold the two-sided PDF onto |x| using the SAME linear bins
    #     Map positive side of signed histogram onto abs grid, add the mirrored negative.
    # Build a signed-grid version at |x|>0
    pos_mask_s = xc_signed > 0
    xc_pos = xc_signed[pos_mask_s]
    pdf_pos = pdf_signed[pos_mask_s]

    # To interpolate negative side onto positive centers (symmetric)
    neg_mask_s = xc_signed < 0
    xc_neg = -xc_signed[neg_mask_s]  # mirror
    pdf_neg = pdf_signed[neg_mask_s]

    # to interpolate both onto the abs centers ea
    pdf_pos_i = np.interp(xc_abs, xc_pos, pdf_pos, left=np.nan, right=np.nan)
    pdf_neg_i = np.interp(xc_abs, xc_neg, pdf_neg, left=np.nan, right=np.nan)
    pdf_abs_fold = np.nan_to_num(pdf_pos_i) + np.nan_to_num(pdf_neg_i)

    # Diagnostics
    area_signed = float(np.sum(pdf_signed * ws))
    area_abs_dir = float(np.sum(pdf_abs_direct * wa))
    area_abs_fld = float(np.sum(pdf_abs_fold * wa))
    print(f"[norm] ∫P(Δx)dx ≈ {area_signed:.5f}, "
          f"∫P(|Δx|)dir ≈ {area_abs_dir:.5f}, ∫P(|Δx|)fold ≈ {area_abs_fld:.5f}")

    # Plateau near 0 with SAME binning
    m_signed = np.abs(xc_signed) <= 0.2
    m_abs = xc_abs <= 0.2
    plat_signed = np.nanmean(pdf_signed[m_signed]) if m_signed.any() else np.nan
    plat_abs_dir = np.nanmean(pdf_abs_direct[m_abs]) if m_abs.any() else np.nan
    plat_abs_fld = np.nanmean(pdf_abs_fold[m_abs]) if m_abs.any() else np.nan
    print(f"[plateau] two-sided ~ {plat_signed:.3e} | "
          f"abs-direct ~ {plat_abs_dir:.3e} | abs-fold ~ {plat_abs_fld:.3e} | "
          f"ratios: dir/signed ≈ {plat_abs_dir/plat_signed:.2f}, "
          f"fold/signed ≈ {plat_abs_fld/plat_signed:.2f}")

    # Plots (for sanity) 
    # Semi-log two-sided
    plt.figure(figsize=(6.6,4.2))
    plt.semilogy(xc_signed, pdf_signed, label="two-sided (linear bins)")
    plt.xlabel("Δx"); plt.ylabel("P(Δx)"); plt.title("Two-sided (semi-log)")
    plt.grid(True, which="both", ls="--", alpha=0.5); plt.legend(); plt.tight_layout()
    plt.savefig("vh_check_two_sided_semilog_matched.png", dpi=200); plt.close()

    # Log-log one-sided (for visualization): you can still use log bins here separately
    lo = max(abs_dx[abs_dx>0].min(), 1e-3); hi = min(np.percentile(abs_dx,99), xlim_abs)
    edges_abs_log = np.logspace(np.log10(lo), np.log10(hi), bins_linear + 1)
    ca_log, ea_log = np.histogram(abs_dx, bins=edges_abs_log)
    wa_log = np.diff(ea_log)
    pdf_abs_log = ca_log / np.maximum(ca_log.sum() * wa_log, 1e-300)
    xc_abs_log = 0.5 * (ea_log[:-1] + ea_log[1:])

    plt.figure(figsize=(6.6,4.2))
    plt.loglog(xc_abs_log, pdf_abs_log, label="one-sided (log bins)")
    plt.xlabel("|Δx|"); plt.ylabel("P(|Δx|)"); plt.title("One-sided (log–log)")
    plt.grid(True, which="both", ls="--", alpha=0.5); plt.legend(); plt.tight_layout()
    plt.savefig("vh_check_one_sided_loglog_visual.png", dpi=200); plt.close()

# new fast version
def vh_consistency_check_v2(dx, bins_linear=400, xlim_abs=15.0):
    """
    Fast numeric check (no plotting):
    - Two-sided P(Δx) on linear bins
    - One-sided P(|Δx|) computed two ways on matched linear bins:
        (A) direct abs histogram
        (B) fold two-sided PDF (with proper sorting before interp)
    Prints: normalization, plateau values, and ratios.
    """
    import numpy as np

    dx = np.asarray(dx)

    # two-sided on linear bins (source of truth)
    edges_lin = np.linspace(-xlim_abs, xlim_abs, bins_linear + 1)
    cs, _ = np.histogram(dx, bins=edges_lin)
    ws = np.diff(edges_lin)
    pdf_signed = cs / np.maximum(cs.sum() * ws, 1e-300)
    xc_signed = 0.5 * (edges_lin[:-1] + edges_lin[1:])

    # One-sided, (A) direct abs on matched linear bins [0, xlim]
    abs_dx = np.abs(dx)
    edges_abs_lin = np.linspace(0.0, xlim_abs, bins_linear//2 + 1)
    ca, _ = np.histogram(abs_dx, bins=edges_abs_lin)
    wa = np.diff(edges_abs_lin)
    pdf_abs_direct = ca / np.maximum(ca.sum() * wa, 1e-300)
    xc_abs = 0.5 * (edges_abs_lin[:-1] + edges_abs_lin[1:])

    # One-sided, (B) fold two-sided PDF -> |x| (with sorting before interp)
    pos_mask = xc_signed > 0
    xc_pos = xc_signed[pos_mask]
    pdf_pos = pdf_signed[pos_mask]
    i_pos = np.argsort(xc_pos)       # <- NEW: ensure increasing
    xc_pos = xc_pos[i_pos]
    pdf_pos = pdf_pos[i_pos]

    neg_mask = xc_signed < 0
    xc_neg = -xc_signed[neg_mask]    # mirror to positive
    pdf_neg =  pdf_signed[neg_mask]
    i_neg = np.argsort(xc_neg)       # <- NEW: ensure increasing
    xc_neg = xc_neg[i_neg]
    pdf_neg = pdf_neg[i_neg]

    pdf_pos_i = np.interp(xc_abs, xc_pos, pdf_pos, left=np.nan, right=np.nan)
    pdf_neg_i = np.interp(xc_abs, xc_neg, pdf_neg, left=np.nan, right=np.nan)
    pdf_abs_fold = np.nan_to_num(pdf_pos_i) + np.nan_to_num(pdf_neg_i)

    # Diagnostics
    area_signed = float(np.sum(pdf_signed * ws))
    area_abs_dir = float(np.sum(pdf_abs_direct * wa))
    area_abs_fld = float(np.sum(pdf_abs_fold   * wa))
    print(f"[norm] ∫P(Δx)dx ≈ {area_signed:.5f}, "
          f"∫P(|Δx|)dir ≈ {area_abs_dir:.5f}, ∫P(|Δx|)fold ≈ {area_abs_fld:.5f}")

    # plateau near 0 (same linear bins on both sides)
    m_signed = (np.abs(xc_signed) <= 0.2)
    m_abs    = (xc_abs <= 0.2)
    plat_signed  = np.nanmean(pdf_signed[m_signed])  if m_signed.any() else np.nan
    plat_abs_dir = np.nanmean(pdf_abs_direct[m_abs]) if m_abs.any() else np.nan
    plat_abs_fld = np.nanmean(pdf_abs_fold[m_abs])   if m_abs.any() else np.nan
    print(f"[plateau] two-sided ~ {plat_signed:.3e} | "
          f"abs-direct ~ {plat_abs_dir:.3e} | abs-fold ~ {plat_abs_fld:.3e} | "
          f"ratios: dir/signed ≈ {plat_abs_dir/plat_signed:.2f}, "
          f"fold/signed ≈ {plat_abs_fld/plat_signed:.2f}")

# same for savers
def vh_assert_consistency(dx, tol_area=1e-2, tol_ratio=0.15, bins=400, xlim_abs=15.0):
    """
    Lightweight consistency check for Van Hove functions.
    Confirms that:
      - ∫P(Δx)dx ≈ 1
      - ∫P(|Δx|) ≈ 1 (both direct and folded)
      - plateau(|Δx|)/plateau(Δx) ≈ 2
    """

    dx = np.asarray(dx)
    if bins % 2 == 1:
        bins += 1

    # Two-sided PDF
    edges = np.linspace(-xlim_abs, xlim_abs, bins + 1)
    counts_signed, _ = np.histogram(dx, bins=edges)
    widths = np.diff(edges)
    pdf_signed = counts_signed / (counts_signed.sum() * widths)
    xc_signed = 0.5 * (edges[:-1] + edges[1:])

    # One-sided (direct)
    abs_dx = np.abs(dx)
    edges_abs = edges[edges >= 0.0]
    if edges_abs[0] > 0:
        edges_abs = np.r_[0.0, edges_abs]
    counts_abs, _ = np.histogram(abs_dx, bins=edges_abs)
    widths_abs = np.diff(edges_abs)
    pdf_abs = counts_abs / (counts_abs.sum() * widths_abs)
    xc_abs = 0.5 * (edges_abs[:-1] + edges_abs[1:])

    # Folded
    i0 = np.argmin(np.abs(edges))
    neg, pos = counts_signed[:i0], counts_signed[i0:]
    neg_flip = neg[::-1]
    m = min(len(neg_flip), len(pos))
    counts_fold = pos[:m] + neg_flip[:m]
    edges_fold = edges[i0:i0+m+1]
    widths_fold = np.diff(edges_fold)
    pdf_fold = counts_fold / (counts_signed.sum() * widths_fold)

    # Integrals
    area_signed = float(np.sum(pdf_signed * widths))
    area_abs = float(np.sum(pdf_abs * widths_abs))
    area_fold = float(np.sum(pdf_fold * widths_fold))

    # Plateaus near zero
    m_signed = np.abs(xc_signed) <= 0.2
    m_abs = xc_abs <= 0.2
    plat_signed = np.nanmean(pdf_signed[m_signed])
    plat_abs = np.nanmean(pdf_abs[m_abs])
    plat_fold = np.nanmean(pdf_fold[m_abs])

    # Print summary
    print(f"[vh_check] areas: signed={area_signed:.5f}, abs={area_abs:.5f}, fold={area_fold:.5f}")
    print(f"[vh_check] plateaus: signed={plat_signed:.3e}, abs={plat_abs:.3e}, fold={plat_fold:.3e}")
    print(f"[vh_check] ratios: abs/signed={plat_abs/plat_signed:.2f}, fold/signed={plat_fold/plat_signed:.2f}")

    # Simple flag
    if (abs(area_signed-1)>tol_area or abs(area_abs-1)>tol_area or abs(area_fold-1)>tol_area
        or abs(plat_abs/plat_signed - 2) > tol_ratio
        or abs(plat_fold/plat_signed - 2) > tol_ratio):
        print("⚠️  [WARNING] Van Hove consistency outside tolerance.\n")
    else:
        print("✅ Van Hove normalization and symmetry consistent.\n")

def vh_consistency_check_v3(dx, bins_linear=400, xlim_abs=15.0):
    import numpy as np
    dx = np.asarray(dx)

    # --- ensure even number of bins so 0.0 is an EDGE, not a center ---
    if bins_linear % 2 == 1:
        bins_linear += 1

    # two-sided on linear bins
    edges_lin = np.linspace(-xlim_abs, xlim_abs, bins_linear + 1)
    counts_signed, _ = np.histogram(dx, bins=edges_lin)   # raw counts
    widths_lin = np.diff(edges_lin)
    pdf_signed = counts_signed / (counts_signed.sum() * widths_lin)
    xc_signed  = 0.5 * (edges_lin[:-1] + edges_lin[1:])

    # one-sided (A) direct |dx| on linear bins [0, X]
    abs_dx = np.abs(dx)
    edges_abs = edges_lin[edges_lin >= 0.0]               # same edges from 0..X
    # If for any reason 0.0 is not exactly present (floating), enforce it:
    if edges_abs[0] > 0:
        edges_abs = np.r_[0.0, edges_abs]
    counts_abs_dir, _ = np.histogram(abs_dx, bins=edges_abs)
    widths_abs = np.diff(edges_abs)
    pdf_abs_direct = counts_abs_dir / (counts_abs_dir.sum() * widths_abs)
    xc_abs = 0.5 * (edges_abs[:-1] + edges_abs[1:])

    # one-sided (B) fold two-sided COUNTS onto [0, X] with matched bins
    # find the index of the zero edge
    i0 = np.argmin(np.abs(edges_lin))  # edge index nearest 0
    # split counts around 0-edge: negatives on the left, non-negative on the right
    counts_neg = counts_signed[:i0]    # bins entirely < 0
    counts_pos = counts_signed[i0:]    # bins with lower edge >= 0
    # mirror negative counts to positive order
    counts_neg_flip = counts_neg[::-1]
    # align lengths (they should match if 0 is an edge; trim to be safe)
    m = min(len(counts_neg_flip), len(counts_pos))
    counts_abs_fold = counts_pos[:m] + counts_neg_flip[:m]
    edges_abs_fold  = edges_lin[i0:i0+m+1]  # the matching positive-side edges
    widths_abs_fold = np.diff(edges_abs_fold)
    pdf_abs_fold = counts_abs_fold / (counts_signed.sum() * widths_abs_fold)

    # --- diagnostics ---
    area_signed  = float(np.sum(pdf_signed     * widths_lin))
    area_abs_dir = float(np.sum(pdf_abs_direct * widths_abs))
    area_abs_fld = float(np.sum(pdf_abs_fold   * widths_abs_fold))
    print(f"[norm] ∫P(Δx)dx ≈ {area_signed:.5f}, "
          f"∫P(|Δx|)dir ≈ {area_abs_dir:.5f}, ∫P(|Δx|)fold ≈ {area_abs_fld:.5f}")

    # plateau near 0 using SAME linear bins
    m_signed = (np.abs(xc_signed) <= 0.2)
    m_abs    = (xc_abs <= 0.2)
    plat_signed  = np.nanmean(pdf_signed[m_signed])       if m_signed.any() else np.nan
    plat_abs_dir = np.nanmean(pdf_abs_direct[m_abs])      if m_abs.any() else np.nan
    # interpolate folded onto the same abs centers only for the print (no renorm)
    pdf_abs_fold_i = np.interp(xc_abs, 0.5*(edges_abs_fold[:-1]+edges_abs_fold[1:]),
                               pdf_abs_fold, left=np.nan, right=np.nan)
    plat_abs_fld = np.nanmean(pdf_abs_fold_i[m_abs])      if m_abs.any() else np.nan

    print(f"[plateau] two-sided ~ {plat_signed:.3e} | "
          f"abs-direct ~ {plat_abs_dir:.3e} | abs-fold ~ {plat_abs_fld:.3e} | "
          f"ratios: dir/signed ≈ {plat_abs_dir/plat_signed:.2f}, "
          f"fold/signed ≈ {plat_abs_fld/plat_signed:.2f}")

# van hove for x, custom
tracks_filtered, single_trajs, double_trajs = CalcMSD(path)

def linear_pooled_log_scaled_van_hove_per_lag_old(tracks, lags_to_plot=[1, 10, 30], bins=400, range_max=15.0): #15,30,46
    # tracks = CalcMSD(path)
    bin_edges = np.linspace(-range_max, range_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    all_data = []  # collect for CSV

    plt.figure(figsize=(8, 5))

    for lag in lags_to_plot:
        all_scaled_dx = []

        for traj in tracks:
            if traj.shape[0] < lag + 1:
                continue
            x = traj[:, 0] #x component  y = traj[:, 1]
            if np.isnan(x).all() or np.allclose(x, x[0]):
                continue

            dx = x[lag:] - x[:-lag]
            safe_log = np.log(np.abs(dx[dx != 0]))
            if len(safe_log) == 0:
                continue
            xi = np.exp(np.mean(safe_log))
            scaled_dx = dx / xi
            all_scaled_dx.extend(scaled_dx)

        if not all_scaled_dx:
            continue
        
        # histogram of all scaled displacements
        hist, _ = np.histogram(all_scaled_dx, bins=bin_edges, density=True)
        H, A, mu, sigma = gauss_fit(bin_centers, hist)
        #gaussian fit
        gauss_curve = gauss(bin_centers, H, A, mu, sigma)

        plt.plot(bin_centers, hist, label=f"Δt = {lag}")
        plt.plot(bin_centers, gauss_curve, '--', label=f"Fit Δt = {lag}, σ={sigma:.2f}")

        # to save each point to all_data
        for x_val, h_val, g_val in zip(bin_centers, hist, gauss_curve):
            all_data.append({
                "lag_time": lag,
                "bin_center": x_val,
                "P(Δx)": h_val,
                "gaussian_fit": g_val
            })

    plt.xlabel("Scaled Δx")
    plt.ylabel("P(Δx)")
    plt.title("Van Hove (per lag) with Log-Scaling plotted together with Gaussian fits")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("Figure 7: vanhove_scaled_fits.png", dpi=300)
    # plt.show()
    # save data as CSV
    # df = pd.DataFrame(all_data)
    # df.to_csv("Table 7: vanhove_scaled_fits_data.csv", index=False)

    return all_data

def linear_pooled_log_scaled_van_hove_per_lag(
    tracks,
    lags_to_plot=(0.1, 0.5, 1, 10, 30, 60),   # seconds or frames (mixed allowed) (0.1, 0.15, 0.22, 0.33, 0.5, 0.75, 1, 1.5, 2.2, 3.3, 5, 10, 30)
    bins=400,
    range_max=15.0,
    dt=0.025
):
    """
    Two-sided Van Hove (signed Δx), linear X + linear Y.
    Uses geometric-mean scaling per lag for robustness.
    Returns: all_data (list of rows for CSV)
    """
    # normalize lags & labels
    max_len = max((traj.shape[0] for traj in tracks if traj is not None), default=0)
    lags_frames, lag_labels = normalize_lags(lags_to_plot, dt=dt, units="auto",
                                             max_frames=max_len, return_labels=True)

    bin_edges   = np.linspace(-range_max, +range_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    all_data = []
    plt.figure(figsize=(8, 5))

    for fr, lab in zip(lags_frames, lag_labels):
        all_scaled_dx = []

        for traj in tracks:
            if traj.shape[0] < fr + 1:
                continue
            x = traj[:, 0]
            if np.isnan(x).all() or np.allclose(x, x[0]):
                continue

            dx = x[fr:] - x[:-fr]
            nz = np.abs(dx[dx != 0.0])
            if nz.size == 0:
                continue
            xi = np.exp(np.mean(np.log(nz)))
            all_scaled_dx.extend(dx / xi)

        if not all_scaled_dx:
            continue

        all_scaled_dx = np.asarray(all_scaled_dx)
        hist, _ = np.histogram(all_scaled_dx, bins=bin_edges, density=True)

        H, A, mu, sigma = gauss_fit(bin_centers, hist)
        gauss_curve = gauss(bin_centers, H, A, mu, sigma)

        # plot (fully linear)
        plt.plot(bin_centers, hist, label=lab)
        plt.plot(bin_centers, gauss_curve, '--',
                 label=f"Fit {lab}, σ={sigma:.2f}")

        lag_s = fr * dt
        for x_val, h_val, g_val in zip(bin_centers, hist, gauss_curve):
            all_data.append({
                "lag_time_frames": int(fr),
                "lag_time_s": float(lag_s),
                "bin_center": float(x_val),
                "P(Δx)": float(h_val),
                "gaussian_fit": float(g_val),
            })

    plt.xlabel("Scaled Δx (signed)")
    plt.ylabel("P(Δx)")
    plt.title("Van Hove (two-sided) with Gaussian fits (linear axes)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # (optional) save a CSV mirroring what was plotted
    if all_data:
        df = pd.DataFrame(all_data).sort_values(["lag_time_frames", "bin_center"])
        df.to_csv("Table_vanhove_two_sided_linear.csv", index=False)

    return all_data


# log-linear for the two side VanHove; nearly same as above
def linearLog_pooled_log_scaled_van_hove_per_lag_old(tracks, lags_to_plot=[1, 10, 30], bins=400, range_max=15.0): #15,30,46  #used to be bins=100, range_max=10.0; increased to 400 and 15 to see fat tails #0.1 1 30
    # tracks = CalcMSD(path)
    bin_edges = np.linspace(-range_max, range_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    all_data = []  # collect for CSV

    plt.figure(figsize=(8, 5))

    for lag in lags_to_plot:
        all_scaled_dx = []

        for traj in tracks:
            if traj.shape[0] < lag + 1:
                continue
            x = traj[:, 0] #x component  y = traj[:, 1]
            if np.isnan(x).all() or np.allclose(x, x[0]):
                continue

            dx = x[lag:] - x[:-lag]
            safe_log = np.log(np.abs(dx[dx != 0]))
            if len(safe_log) == 0:
                continue
            xi = np.exp(np.mean(safe_log))
            scaled_dx = dx / xi
            all_scaled_dx.extend(scaled_dx)

        if not all_scaled_dx:
            continue
        
        # histogram of all scaled displacements
        hist, _ = np.histogram(all_scaled_dx, bins=bin_edges, density=True)
        H, A, mu, sigma = gauss_fit(bin_centers, hist)
        #gaussian fit
        gauss_curve = gauss(bin_centers, H, A, mu, sigma)

        plt.plot(bin_centers, hist, label=f"Δt = {lag}")
        plt.plot(bin_centers, gauss_curve, '--', label=f"Fit Δt = {lag}, σ={sigma:.2f}")

        # to save each point to all_data
        for x_val, h_val, g_val in zip(bin_centers, hist, gauss_curve):
            all_data.append({
                "lag_time": lag,
                "bin_center": x_val,
                "P(Δx)": h_val,
                "gaussian_fit": g_val
            })

    plt.yscale("log")  # to change Y to log to highlight tails
    plt.xlabel("Scaled Δx")
    plt.ylabel("P(Δx)")
    plt.title("Van Hove (two-sided) with log-scaled Y and Gaussian fits")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("Figure 7: vanhove_scaled_fits.png", dpi=300)
    # plt.show()
    # save data as CSV
    # df = pd.DataFrame(all_data)
    # df.to_csv("Table 7: vanhove_scaled_fits_data.csv", index=False)

    return all_data

def linearLog_pooled_log_scaled_van_hove_per_lag(
    tracks,
    lags_to_plot= (0.1, 0.5, 1, 10, 30, 60),  # seconds or frames (mixed allowed) # 0.22, 0.33, 0.5, 0.75, 1, 1.5 (0.1, 0.15,  1.5, 3, 5, 10, 30), 
    bins=400,
    range_max=15.0,
    dt=0.025
):
    """
    Two-sided Van Hove (signed Δx), linear X + log Y.
    Uses geometric-mean scaling per lag for robustness.
    Returns: all_data (list of rows for CSV)
    """
    # normalize lags & labels
    max_len = max((traj.shape[0] for traj in tracks if traj is not None), default=0)
    lags_frames, lag_labels = normalize_lags(lags_to_plot, dt=dt, units="auto",
                                             max_frames=max_len, return_labels=True)

    # two-sided linear bins over signed displacements
    bin_edges   = np.linspace(-range_max, +range_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    all_data = []
    plt.figure(figsize=(8, 5))
    eps = 1e-12  # only for display on log-Y

    for fr, lab in zip(lags_frames, lag_labels):
        all_scaled_dx = []

        for traj in tracks:
            if traj.shape[0] < fr + 1:
                continue
            x = traj[:, 0]
            if np.isnan(x).all() or np.allclose(x, x[0]):
                continue

            dx = x[fr:] - x[:-fr]

            # check
            # Example inside your loop for a chosen lag:
            # vh_consistency_check(dx, bins_linear=400, xlim_abs=15.0)


            nz = np.abs(dx[dx != 0.0])
            if nz.size == 0:
                continue
            xi = np.exp(np.mean(np.log(nz)))
            all_scaled_dx.extend(dx / xi)

        if not all_scaled_dx:
            continue

        all_scaled_dx = np.asarray(all_scaled_dx)
        hist, _ = np.histogram(all_scaled_dx, bins=bin_edges, density=True)

        H, A, mu, sigma = gauss_fit(bin_centers, hist)
        gauss_curve = gauss(bin_centers, H, A, mu, sigma)

        # plot (log Y)
        plt.plot(bin_centers, np.clip(hist, eps, None), label=lab)
        plt.plot(bin_centers, np.clip(gauss_curve, eps, None), '--',
                 label=f"Fit {lab}, σ={sigma:.2f}")

        lag_s = fr * dt
        for x_val, h_val, g_val in zip(bin_centers, hist, gauss_curve):
            all_data.append({
                "lag_time_frames": int(fr),
                "lag_time_s": float(lag_s),
                "bin_center": float(x_val),
                "P(Δx)": float(h_val),
                "gaussian_fit": float(g_val),
            })

    plt.yscale("log")                      # linear X + log Y
    plt.xlabel("Scaled Δx (signed)")
    plt.ylabel("P(Δx)")
    plt.title("Van Hove (two-sided) with log-scaled Y and Gaussian fits")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # (optional) save a CSV mirroring what was plotted
    if all_data:
        df = pd.DataFrame(all_data).sort_values(["lag_time_frames", "bin_center"])
        df.to_csv("Table_vanhove_two_sided_logY.csv", index=False)

    return all_data


# same as above but with log scale + added cage hopping
#JAN

def pooled_log_scaled_van_hove_per_lag(
    tracks,
    lags_to_plot=(0.1, 0.5, 1, 10, 30, 60), # seconds or frames (mixed allowed) #(0.1, 1.0, 30.0) (0.1, 0.15, 0.22, 0.33, 0.5, 0.75, 1, 1.5, 2.2, 3.3, 5, 10, 30) 0.1, 0.15,  1.5, 3, 5, 10, 30), 
    bins=100,
    range_max=10.0,  # seconds per frame
    dt=0.025,
    lag_units="seconds", # kept for backward compat; we use normalize_lags(auto)
    *,
    metrics_dict=None,      # <-- to write is_hopper
    track_ids=None,         # <-- map local_index -> metrics_key
    attach_hoppers=True     # <-- to turn off single/double runs
):

    """
    One-sided Van Hove with log–log axes. Displacements are scaled per lag by a
    geometric-mean factor to make curves comparable across Δt.
    Accepts lag list in seconds or frames; legends show both.
    Returns: (all_data, Rg_hoppers, Rg_non_hoppers)
    """
    all_data = []
    

    # decide the longest usable length to clip lags safely
    max_len = max((traj.shape[0] for traj in tracks if traj is not None), default=0)

    # unified helper: frames (int) for computation + prebuilt labels for legend
    lags_frames, lag_labels = normalize_lags(
        lags_to_plot, dt=dt, units="auto", max_frames=max_len, return_labels=True
    )

    plt.figure(figsize=(8, 5))

    for fr, lab in zip(lags_frames, lag_labels):
        all_scaled_dx = []

        for traj in tracks:
            if traj.shape[0] < fr + 1:
                continue
            x = traj[:, 0]
            if np.isnan(x).all() or np.allclose(x, x[0]):
                continue

            dx = x[fr:] - x[:-fr]

            # #  check
            # vh_consistency_check_v3(dx, bins_linear=400, xlim_abs=15.0)

            # robust per-lag scale (geometric mean of |dx| over nonzeros)
            nz = np.abs(dx[dx != 0.0])
            if nz.size == 0:
                continue
            xi = np.exp(np.mean(np.log(nz)))
            scaled_dx = dx / xi
            all_scaled_dx.extend(scaled_dx)

        if not all_scaled_dx:
            continue

        # build |Δx|, exclude zeros for log binning
        all_scaled_dx = np.abs(np.asarray(all_scaled_dx))
        all_scaled_dx = all_scaled_dx[all_scaled_dx > 0.0]
        if all_scaled_dx.size == 0:
            continue

        # log-spaced bins for |Δx|
        max_abs_dx  = min(np.percentile(all_scaled_dx, 99), range_max)
        min_nonzero = max(np.min(all_scaled_dx), 1e-3)
        if min_nonzero >= max_abs_dx:   # safety for pathological cases
            continue

        bin_edges   = np.logspace(np.log10(min_nonzero), np.log10(max_abs_dx), bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist, _     = np.histogram(all_scaled_dx, bins=bin_edges, density=True)

        # gaussian fit in the plotted coordinates
        H, A, mu, sigma = gauss_fit(bin_centers, hist)
        gauss_curve      = gauss(bin_centers, H, A, mu, sigma)

        # plot main + dashed fit (labels already include both units)
        plt.plot(bin_centers, hist, label=lab)
        plt.plot(bin_centers, gauss_curve, '--', label=f"Fit {lab}, σ={sigma:.2f}")

        # stash for CSV
        lag_s = fr * dt
        for x_val, h_val, g_val in zip(bin_centers, hist, gauss_curve):
            all_data.append({
                "lag_time_frames": int(fr),
                "lag_time_s": float(lag_s),
                "bin_center": float(x_val),
                "P(|Δx|)": float(h_val),
                "gaussian_fit": float(g_val),
            })

    # styling to match in-function look
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Scaled |Δx|")
    plt.ylabel("P(|Δx|)")
    plt.title("Van Hove (log–log) per lag with Gaussian fits")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # save the data (same table name you used)
    if all_data:
        df = pd.DataFrame(all_data).sort_values(["lag_time_frames", "bin_center"])
        df.to_csv("Table 7: vanhove_log_scaled_fits_data.csv", index=False)

    # --- cage hopping classification (unchanged) ---
    # seg_size = 10
    # Rg_hoppers, Rg_non_hoppers = [], []
    # for traj in tracks:
    #     if traj.shape[0] < seg_size:
    #         continue
    #     Rg_values = calc_Rg_seg(traj, seg_size)
    #     if (len(Rg_values) == 0) or np.isnan(Rg_values).all():
    #         continue

    #     mean_Rg = np.nanmean(Rg_values)
    #     std_Rg  = np.nanstd(Rg_values)
    #     threshold = mean_Rg + 2 * std_Rg

    #     (Rg_hoppers if np.any(Rg_values > threshold) else Rg_non_hoppers).append(traj)

    # v2
    seg_size = 10
    Rg_hoppers, Rg_non_hoppers, rg_diag = robust_rg_hopper_split(tracks, seg_size=seg_size, alpha=0.975)
    # (opt.) save diagnostics for later inspection
    try:
        rg_diag.to_csv("TableTest_Rg_robust_threshold_diag.csv", index=False)
    except Exception:
        pass

    #JAN 
    # master metrics
    # # ---- attach hopper labels to metrics (bulletproof) ----


    # ---- attach hopper labels to metrics (bulletproof) ----
    if attach_hoppers and (metrics_dict is not None):
        # ---- attach hopper labels to metrics (with index→ID mapping) ----
        missing = 0
        written = 0

        id_col = "traj_id" if "traj_id" in rg_diag.columns else (
                "traj_index" if "traj_index" in rg_diag.columns else None)

        if id_col is None:
            print("[hopper->metrics] ERROR: rg_diag has no traj_id/traj_index column. Columns:",
                list(rg_diag.columns))
        else:
            for _, row in rg_diag.iterrows():
                try:
                    tid = int(row[id_col])
                except Exception:
                    continue

                # 🔑 ВАЖЛИВО: мапінг індекс → реальний track_id
                if track_ids is not None:
                    if 0 <= tid < len(track_ids):
                        tid = track_ids[tid]

                if tid not in metrics:
                    missing += 1
                    continue

                metrics[tid]["is_hopper"] = int(bool(row.get("is_hopper", False)))
                written += 1

        print(f"[hopper->metrics] wrote={written}, missing_metrics_keys={missing}, rg_diag_rows={len(rg_diag)}")
        print("hopper counts in metrics:",
            sum(v.get("is_hopper", 0) == 1 for v in metrics.values()))
        
    return all_data, Rg_hoppers, Rg_non_hoppers




def pooled_log_scaled_van_hove_per_lag_v1(tracks, lags_to_plot=[0.1, 1, 30], bins=100, range_max=10.0, dt=0.025, lag_units="seconds"  ): # 'frames' or 'seconds'):  #already log scale
    all_data = []  # collect for CSV
    plt.figure(figsize=(8, 5))

     # (optional) use the longest usable length across tracks to clip lags

    max_len = 0

    for traj in tracks:

        if traj.shape[0] > max_len:

            max_len = traj.shape[0]



    lags_frames = normalize_lags(lags_to_plot, dt=dt, max_frames=max_len, units=lag_units) #needed _normalize_lags

    if lags_frames.size == 0:

        raise ValueError("No valid lags after normalization—check dt/units and track lengths.")



    for lag in lags_frames:
    # for lag in lags_to_plot:
        all_scaled_dx = []

        for traj in tracks:
            if traj.shape[0] < lag + 1:
                continue
            x = traj[:, 0] ##x component  y = traj[:, 1]
            if np.isnan(x).all() or np.allclose(x, x[0]):
                continue

            dx = x[lag:] - x[:-lag]
            safe_log = np.log(np.abs(dx[dx != 0]))
            if len(safe_log) == 0:
                continue
            xi = np.exp(np.mean(safe_log))
            scaled_dx = dx / xi
            all_scaled_dx.extend(scaled_dx)

        if not all_scaled_dx:
            continue

        # all_scaled_dx = np.abs(all_scaled_dx) 
        # all_scaled_dx = all_scaled_dx[all_scaled_dx > 0]# remove exact zeros to avoid log(0)
        
        all_scaled_dx = np.abs(np.asarray(all_scaled_dx))
        all_scaled_dx = all_scaled_dx[all_scaled_dx > 0]

        if len(all_scaled_dx) == 0:
            raise ValueError("All displacements are zero; cannot create log-log plot.")
        #  continue

        max_abs_dx = min(np.percentile(all_scaled_dx, 99), 50) # clip at 99th percentile
        min_nonzero = max(np.min(all_scaled_dx), 1e-3) # set a reasonable lower limit -> # lower limit for it not to be too small/cause issues

        bin_edges = np.logspace(np.log10(min_nonzero), np.log10(max_abs_dx), bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        hist, _ = np.histogram(all_scaled_dx, bins=bin_edges, density=True)
        H, A, mu, sigma = gauss_fit(bin_centers, hist) #may not be gaussioan anymore -> because 
        gauss_curve = gauss(bin_centers, H, A, mu, sigma)

        # plt.plot(bin_centers, hist, label=f"Δt = {lag}")
        # plt.plot(bin_centers, gauss_curve, '--', label=f"Fit Δt = {lag}, σ={sigma:.2f}")
        
        # label in both units

        lag_s = lag * dt

        plt.plot(bin_centers, hist, label=f"Δt = {lag_s:g}s ({lag} fr)")
        plt.plot(bin_centers, gauss_curve, '--', label=f"Fit Δt = {lag_s:g}s, σ={sigma:.2f}")


        for x_val, h_val, g_val in zip(bin_centers, hist, gauss_curve):
            all_data.append({
                
                "lag_time_frames": int(lag),
                "lag_time_s": lag_s,
                # "lag_time": lag,
                "bin_center": x_val,
                "P(Δx)": h_val,
                "gaussian_fit": g_val
            })

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Scaled |Δx|")
    plt.ylabel("P(|Δx|)")
    plt.title("Van Hove (log-log) per lag with Gaussian fits")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    # plt.savefig("Figure 7: vanhove_log_scaled_fits.png", dpi=300)


    # plt.show()

    # save data as CSV

    df = pd.DataFrame(all_data)
    df.to_csv("Table 7: vanhove_log_scaled_fits_data.csv", index=False)

    #cage hopping: 
    seg_size = 10  # to adjust as needed
    Rg_hoppers = []
    Rg_non_hoppers = []

    for traj in tracks:
        if traj.shape[0] < seg_size:
            continue  # too short to segment

        Rg_values = calc_Rg_seg(traj, seg_size) #applied to each traj.

        if len(Rg_values) == 0 or np.isnan(Rg_values).all():
            continue

        mean_Rg = np.nanmean(Rg_values)
        std_Rg = np.nanstd(Rg_values)
        threshold = mean_Rg + 2 * std_Rg  #mean + 2×std. =>10 -extreme

        if np.any(Rg_values > threshold):
            Rg_hoppers.append(traj) #save the idx; look for %
        else:
            Rg_non_hoppers.append(traj)

    return all_data, Rg_hoppers, Rg_non_hoppers

# old variant
def pooled_log_scaled_van_hove_per_lag_old(tracks, lags_to_plot=[0.1, 1, 30], bins=100, range_max=10.0):  #already log scale

    all_data = []  # collect for CSV


    plt.figure(figsize=(8, 5))



    for lag in lags_to_plot:

        all_scaled_dx = []



        for traj in tracks:

            if traj.shape[0] < lag + 1:

                continue

            x = traj[:, 0] ##x component  y = traj[:, 1]

            if np.isnan(x).all() or np.allclose(x, x[0]):

                continue



            dx = x[lag:] - x[:-lag]

            safe_log = np.log(np.abs(dx[dx != 0]))

            if len(safe_log) == 0:

                continue

            xi = np.exp(np.mean(safe_log))

            scaled_dx = dx / xi

            all_scaled_dx.extend(scaled_dx)



        if not all_scaled_dx:

            continue



        all_scaled_dx = np.abs(all_scaled_dx) 

        all_scaled_dx = all_scaled_dx[all_scaled_dx > 0]# remove exact zeros to avoid log(0)



        if len(all_scaled_dx) == 0:

            raise ValueError("All displacements are zero; cannot create log-log plot.")



        max_abs_dx = min(np.percentile(all_scaled_dx, 99), 50) # clip at 99th percentile

        min_nonzero = max(np.min(all_scaled_dx), 1e-3) # set a reasonable lower limit -> # lower limit for it not to be too small/cause issues



        bin_edges = np.logspace(np.log10(min_nonzero), np.log10(max_abs_dx), bins + 1)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])



        hist, _ = np.histogram(all_scaled_dx, bins=bin_edges, density=True)

        H, A, mu, sigma = gauss_fit(bin_centers, hist) #may not be gaussioan anymore -> because 

        gauss_curve = gauss(bin_centers, H, A, mu, sigma)



        plt.plot(bin_centers, hist, label=f"Δt = {lag}")

        plt.plot(bin_centers, gauss_curve, '--', label=f"Fit Δt = {lag}, σ={sigma:.2f}")



        for x_val, h_val, g_val in zip(bin_centers, hist, gauss_curve):

            all_data.append({

                "lag_time": lag,

                "bin_center": x_val,

                "P(Δx)": h_val,

                "gaussian_fit": g_val

            })



    plt.xscale("log")

    plt.yscale("log")

    plt.xlabel("Scaled |Δx|")

    plt.ylabel("P(|Δx|)")

    plt.title("Van Hove (log-log) per lag with Gaussian fits")

    plt.legend()

    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()

    # plt.savefig("Figure 7: vanhove_log_scaled_fits.png", dpi=300)





    # plt.show()



    # save data as CSV



    df = pd.DataFrame(all_data)

    df.to_csv("Table 7: vanhove_log_scaled_fits_data.csv", index=False)



    #cage hopping: 

    seg_size = 10  # to adjust as needed

    Rg_hoppers = []

    Rg_non_hoppers = []



    for traj in tracks:

        if traj.shape[0] < seg_size:

            continue  # too short to segment



        Rg_values = calc_Rg_seg(traj, seg_size) #applied to each traj.



        if len(Rg_values) == 0 or np.isnan(Rg_values).all():

            continue



        mean_Rg = np.nanmean(Rg_values)

        std_Rg = np.nanstd(Rg_values)

        threshold = mean_Rg + 2 * std_Rg  #mean + 2×std. =>10 -extreme



        if np.any(Rg_values > threshold):

            Rg_hoppers.append(traj) #save the idx; look for %

        else:

            Rg_non_hoppers.append(traj)



    return all_data, Rg_hoppers, Rg_non_hoppers

# to save all plots instead of 1 by 1

# linear
def save_van_hove_results(all_data, csv_filename="Table_vanHove.csv", fig_filename="Figure_vanHove.png"):
    """
    Save van Hove results from pooled_log_scaled_van_hove_per_lag to CSV and figure.
    Input:
        all_data – list of dicts returned from pooled_log_scaled_van_hove_per_lag()
    """
    if not all_data:
        print("No data to save.")
        return

    # converting to DataFrame
    df = pd.DataFrame(all_data)
    df.to_csv(csv_filename, index=False)

    # plot
    plt.figure(figsize=(8, 5))
    for lag in sorted(df["lag_time"].unique()):
        sub = df[df["lag_time"] == lag]
        plt.plot(sub["bin_center"], sub["P(Δx)"], label=f"Δt = {lag}")
        plt.plot(sub["bin_center"], sub["gaussian_fit"], '--', label=f"Fit Δt = {lag}, σ≈{sub['gaussian_fit'].std():.2f}")

    plt.xlabel("Scaled Δx")
    plt.ylabel("P(Δx)")
    plt.title("Van Hove (log-scaled) with Gaussian fits")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

def save_van_hove_results_linear_difColors(
    all_data,
    csv_filename="Table_vanHove_linear.csv",
    fig_filename="Figure_vanHove_linear.png",
    dt=0.025,              # used if lag_time_s missing
    refit_sigma=False      # True => refit sigma from (x,y)
):
    """
    Replot two-sided Van Hove (signed Δx) with linear X and linear Y.
    Compatible with outputs from linear_pooled_log_scaled_van_hove_per_lag().

    Expects rows with:
      - 'lag_time_frames' (int) and/or 'lag_time_s' (float)
      - 'bin_center'
      - 'P(Δx)' OR 'P(|Δx|)' (probability density column)
      - optional 'gaussian_fit'
    """
    if not all_data:
        print("No data to save.")
        return

    df = pd.DataFrame(all_data).copy()
    # Upd: make saver compatible with all Van Hove function v
    # Older versions used "lag_time", newer ones use "lag_time_s" or "lag_time_frames".
    if "lag_time" not in df.columns:
        if "lag_time_s" in df.columns:
            df["lag_time"] = df["lag_time_s"]
        elif "lag_time_frames" in df.columns:
            df["lag_time"] = df["lag_time_frames"]
        else:
            raise KeyError(
                "Expected one of 'lag_time', 'lag_time_s', or 'lag_time_frames' in the data."
            )
    # --- end update ---


    # pick probability column robustly
    pcol = None
    for cand in ("P(Δx)", "P(|Δx|)", "P(dx)", "P(|dx|)"):
        if cand in df.columns:
            pcol = cand
            break
    if pcol is None:
        raise KeyError("Expected a probability column like 'P(Δx)' or 'P(|Δx|)' not found in all_data.")

    # ensure lag columns
    if "lag_time_frames" not in df.columns:
        if "lag_time" in df.columns:  # legacy
            df["lag_time_frames"] = df["lag_time"].astype(int)
        else:
            raise KeyError("No 'lag_time_frames' (or fallback 'lag_time') in all_data.")
    if "lag_time_s" not in df.columns:
        df["lag_time_s"] = df["lag_time_frames"] * float(dt)

    # sort & persist
    df = df.sort_values(["lag_time_frames", "bin_center"])
    df.to_csv(csv_filename, index=False)

    # plot
    plt.figure(figsize=(8, 5))
    for fr, sub in df.groupby("lag_time_frames"):
        x = sub["bin_center"].to_numpy()
        y = sub[pcol].to_numpy()

        # main curve (no clipping needed on linear axes)
        plt.plot(x, y, label=f"Δt = {sub['lag_time_s'].iloc[0]:.3f}s ({int(fr)} fr)")

        # dashed fit
        do_refit = refit_sigma or ("gaussian_fit" not in sub.columns)
        if do_refit:
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                g = gauss(x, H, A, mu, sigma)
                plt.plot(x, g, "--",
                         label=f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma:.2f}")
            except Exception:
                pass
        else:
            g = sub["gaussian_fit"].to_numpy()
            # try to annotate σ (non-fatal if it fails)
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                lbl = f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma:.2f}"
            except Exception:
                lbl = f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s"
            plt.plot(x, g, "--", label=lbl)

    plt.xlabel("Scaled Δx (signed)")
    plt.ylabel("P(Δx)")
    plt.title("Van Hove (two-sided) with Gaussian fits (linear axes)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

def save_van_hove_results_linear(
    all_data,
    csv_filename="Table_vanHove_linear.csv",
    fig_filename="Figure_vanHove_linear.png",
    dt=0.025,              # used if lag_time_s missing
    refit_sigma=False      # True => refit sigma from (x,y)
):
    """
    Replot two-sided Van Hove (signed Δx) with linear X and linear Y.
    Compatible with outputs from linear_pooled_log_scaled_van_hove_per_lag().

    Expects rows with:
      - 'lag_time_frames' (int) and/or 'lag_time_s' (float)
      - 'bin_center'
      - 'P(Δx)' OR 'P(|Δx|)' (probability density column)
      - optional 'gaussian_fit'
    """
    if not all_data:
        print("No data to save.")
        return

    df = pd.DataFrame(all_data).copy()
    # Upd: make saver compatible with all Van Hove function versions —
    if "lag_time" not in df.columns:
        if "lag_time_s" in df.columns:
            df["lag_time"] = df["lag_time_s"]
        elif "lag_time_frames" in df.columns:
            df["lag_time"] = df["lag_time_frames"]
        else:
            raise KeyError(
                "Expected one of 'lag_time', 'lag_time_s', or 'lag_time_frames' in the data."
            )
    # --- end update ---

    # pick probability column robustly
    pcol = None
    for cand in ("P(Δx)", "P(|Δx|)", "P(dx)", "P(|dx|)"):
        if cand in df.columns:
            pcol = cand
            break
    if pcol is None:
        raise KeyError("Expected a probability column like 'P(Δx)' or 'P(|Δx|)' not found in all_data.")

    # ensure lag columns
    if "lag_time_frames" not in df.columns:
        if "lag_time" in df.columns:
            df["lag_time_frames"] = df["lag_time"].astype(int)
        else:
            raise KeyError("No 'lag_time_frames' in all_data.")
    if "lag_time_s" not in df.columns:
        df["lag_time_s"] = df["lag_time_frames"] * float(dt)

    # sort & persist
    df = df.sort_values(["lag_time_frames", "bin_center"])
    df.to_csv(csv_filename, index=False)

    # plot
    plt.figure(figsize=(8, 5))
    for fr, sub in df.groupby("lag_time_frames"):

        x = sub["bin_center"].to_numpy()
        y = sub[pcol].to_numpy()

        # main curve  --------------------------------------------------- #
        line, = plt.plot(    #new
            x,
            y,
            label=f"Δt = {sub['lag_time_s'].iloc[0]:.3f}s ({int(fr)} fr)"
        )  #new
        color = line.get_color()   #new  extract chosen color

        # dashed fit  --------------------------------------------------- #
        do_refit = refit_sigma or ("gaussian_fit" not in sub.columns)
        if do_refit:
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                g = gauss(x, H, A, mu, sigma)
                plt.plot(
                    x,
                    g,
                    "--",             #new
                    color=color,     #new  match main curve color
                    label=f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma:.2f}"
                )  #new
            except Exception:
                pass
        else:
            g = sub["gaussian_fit"].to_numpy()
            # annotate σ if possible
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                lbl = f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma:.2f}"
            except Exception:
                lbl = f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s"

            plt.plot(
                x,
                g,
                "--",             #new
                color=color,     #new
                label=lbl
            )  #new

    plt.xlabel("Scaled Δx (signed)")
    plt.ylabel("P(Δx)")
    plt.title("Van Hove (two-sided) with Gaussian fits (linear axes)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()


def save_van_hove_results_abs_old(all_data, csv_filename="Table_vanHove.csv", fig_filename="Figure_vanHove.png"):
    """
    Replot Van Hove from pooled_log_scaled_van_hove_per_lag() to match its log–log look.
    - Sorts by (lag_time, bin_center)
    - Uses log–log axes and consistent labels/grid/title
    - If 'gaussian_fit' is in the data, plots it; otherwise refits to get the dashed curve + sigma label
    """
    if not all_data:
        print("No data to save.")
        return

    # to convert and sort (ensures smooth lines)
    df = pd.DataFrame(all_data).copy()
    # Upd: make saver compatible with all Van Hove function versions
    # Older versions used "lag_time", newer ones use "lag_time_s" or "lag_time_frames".
    if "lag_time" not in df.columns:
        if "lag_time_s" in df.columns:
            df["lag_time"] = df["lag_time_s"]
        elif "lag_time_frames" in df.columns:
            df["lag_time"] = df["lag_time_frames"]
        else:
            raise KeyError(
                "Expected one of 'lag_time', 'lag_time_s', or 'lag_time_frames' in the data."
            )
    #

    pcol = "P(Δx)" if "P(Δx)" in df.columns else ("P(|Δx|)" if "P(|Δx|)" in df.columns else None)
    if pcol is None:
        raise KeyError("Expected column 'P(Δx)' or 'P(|Δx|)' not found in all_data.")

    df = df.sort_values(["lag_time", "bin_center"])
    df.to_csv(csv_filename, index=False)

    eps = 1e-12  # avoid log(0)

    plt.figure(figsize=(8, 5))
    for lag, sub in df.groupby("lag_time"):
        x = sub["bin_center"].to_numpy()
        y = np.clip(sub[pcol].to_numpy(), eps, None)

        # main curve
        # plt.plot(x, y, label=f"Δt = {int(lag)}")
        # to plot decimals (frames)

        # dashed fit: prefer existing column; otherwise refit to get sigma for label
        fit_label = f"Fit Δt = {int(lag)}"
        if "gaussian_fit" in sub.columns:
            g = np.clip(sub["gaussian_fit"].to_numpy(), eps, None)
            # try to recover sigma for the legend by refitting (non-fatal if fails)
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                fit_label = f"Fit Δt = {int(lag)}, σ={sigma:.2f}"
            except Exception:
                pass
            plt.plot(x, g, "--", label=fit_label)
        else:
            # no stored fit—compute/plot one so the figure still looks like the in-function plot
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                g = np.clip(gauss(x, H, A, mu, sigma), eps, None)
                plt.plot(x, g, "--", label=f"Fit Δt = {int(lag)}, σ={sigma:.2f}")
            except Exception:
                # silently skip if fitting fails
                pass

    # to match the pooled function's styling
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Scaled |Δx|")
    plt.ylabel("P(|Δx|)")
    plt.title("Van Hove (log-log) per lag with Gaussian fits")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()
# try change y-axis to log scale (instead of linear) - highlight differences


def save_van_hove_results_abs_v1(
    all_data,
    csv_filename="Table_vanHove.csv",
    fig_filename="Figure_vanHove.png",
    prefer_seconds=True,   # choose which unit to show first in legend
    dt=0.025               # only used if we must derive seconds from frames
):
    """
    Replot Van Hove from pooled_*() outputs to match in-function log–log look.

    - Accepts dict list or DataFrame with columns:
        bin_center, P(Δx) or P(|Δx|), (optional) gaussian_fit,
        lag_time_frames and/or lag_time_s (or legacy lag_time).
    - Sorts within each lag by bin_center to ensure smooth lines.
    - Uses log–log axes with consistent labels/grid/title.
    - If 'gaussian_fit' is present, plots it; otherwise refits a Gaussian on |Δx|.
    - Saves a clean CSV identical to what it plotted.
    """

    if not all_data:
        print("No data to save.")
        return

    # Normalize input -> DataFrame
    df = pd.DataFrame(all_data).copy()

    # pick probability column name
    pcol = None
    for cand in ["P(Δx)", "P(|Δx|)", "P(deltax)", "P_abs"]:
        if cand in df.columns:
            pcol = cand
            break
    if pcol is None:
        raise KeyError("Expected a probability column like 'P(Δx)' or 'P(|Δx|)' not found.")

    # normalize lag columns (support old/new schemas)
    # priority: explicit lag_time_frames / lag_time_s; fall back to legacy lag_time
    if "lag_time_frames" not in df.columns:
        if "lag_time" in df.columns and np.issubdtype(df["lag_time"].dtype, np.number):
            df["lag_time_frames"] = df["lag_time"].astype(int)
        else:
            # if we only have seconds, we can infer frames using dt
            if "lag_time_s" in df.columns:
                df["lag_time_frames"] = np.round(df["lag_time_s"].to_numpy() / float(dt)).astype(int)
            else:
                raise KeyError("No lag column found (need 'lag_time_frames' or 'lag_time' or 'lag_time_s').")

    if "lag_time_s" not in df.columns:
        # if seconds not present, derive from frames
        df["lag_time_s"] = df["lag_time_frames"].to_numpy() * float(dt)

    # sort for smooth lines
    df = df.sort_values(["lag_time_frames", "bin_center"]).reset_index(drop=True)

    # write exactly what we plotted
    df.to_csv(csv_filename, index=False)

    eps = 1e-12  # avoid log(0)

    plt.figure(figsize=(8, 5))
    for lag_frames, sub in df.groupby("lag_time_frames", sort=True):
        x = sub["bin_center"].to_numpy()
        y = np.clip(sub[pcol].to_numpy(), eps, None)

        # legend label with both units (seconds first by default)
        lag_sec = float(np.unique(sub["lag_time_s"])[0])
        if prefer_seconds:
            main_lab = f"Δt = {lag_sec:g}s ({int(lag_frames)} fr)"
        else:
            main_lab = f"Δt = {int(lag_frames)} fr ({lag_sec:g}s)"

        plt.plot(x, y, label=main_lab)

        # dashed fit
        fit_label = f"Fit {main_lab}"
        if "gaussian_fit" in sub.columns:
            g = np.clip(sub["gaussian_fit"].to_numpy(), eps, None)
            # try to recover sigma for legend by refitting (non-fatal if it fails)
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                fit_label = f"{fit_label}, σ={sigma:.2f}"
            except Exception:
                pass
            plt.plot(x, g, "--", label=fit_label)
        else:
            # no stored fit → compute a quick one for the dashed curve
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                g = np.clip(gauss(x, H, A, mu, sigma), eps, None)
                plt.plot(x, g, "--", label=f"{fit_label}, σ={sigma:.2f}")
            except Exception:
                # silently skip if fitting fails (keeps main curve)
                pass

    # match pooled function styling
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Scaled |Δx|")
    plt.ylabel("P(|Δx|)")
    plt.title("Van Hove (log-log) per lag with Gaussian fits")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

def save_van_hove_results_abs_notLimited(
    all_data,
    csv_filename="Table_vanHove_abs.csv",
    fig_filename="Figure_vanHove_abs.png",
    dt=0.025,                 # only used for labels if lag_time_s is missing
    refit_sigma=False,        # True => refit sigma from (x,y) instead of trusting stored gaussian_fit
    clip_eps_for_plot=1e-12   # avoid log(0) on display; CSV stays unmodified
):
    """
    Replot one-sided Van Hove (|Δx|) results to match the in-function log–log look.
    Works with data produced by the new pooled_* function(s).

    Expects (rows) with:
      - 'lag_time_frames' (int) and/or 'lag_time_s' (float)
      - 'bin_center'
      - 'P(|Δx|)'  OR 'P(Δx)'    (probability density column name)
      - optional 'gaussian_fit'

    Behavior:
      - Sorts by (lag, bin_center) for smooth lines
      - Labels show both seconds and frames: Δt = {secs:.3f}s ({frames} fr)
      - Plots log–log axes
      - If refit_sigma=True OR no 'gaussian_fit' column, refits Gaussian for dashed curve
      - Otherwise uses stored 'gaussian_fit' as the dashed curve
    """
    if not all_data:
        print("No data to save.")
        return

    # to convert + pick probability column name robustly
    df = pd.DataFrame(all_data).copy()

    # Udp: make saver compatible with all Van Hove function versions ---
    # Older versions used "lag_time", newer ones use "lag_time_s" or "lag_time_frames".
    if "lag_time" not in df.columns:
        if "lag_time_s" in df.columns:
            df["lag_time"] = df["lag_time_s"]
        elif "lag_time_frames" in df.columns:
            df["lag_time"] = df["lag_time_frames"]
        else:
            raise KeyError(
                "Expected one of 'lag_time', 'lag_time_s', or 'lag_time_frames' in the data."
            )
    # 

    pcol = None
    for cand in ("P(|Δx|)", "P(Δx)", "P(|dx|)", "P(dx)"):
        if cand in df.columns:
            pcol = cand
            break
    if pcol is None:
        raise KeyError("Expected a probability column like 'P(|Δx|)' or 'P(Δx)' not found in all_data.")

    # to ensure lag columns exist (at least frames; seconds for nicer labels)
    has_frames = "lag_time_frames" in df.columns
    has_secs   = "lag_time_s" in df.columns
    if not has_frames and "lag_time" in df.columns:
        # back-compat: older code used 'lag_time' (frames)
        df["lag_time_frames"] = df["lag_time"].astype(int)
        has_frames = True
    if not has_secs and has_frames:
        df["lag_time_s"] = df["lag_time_frames"] * float(dt)
        has_secs = True

    if not has_frames:
        raise KeyError("No 'lag_time_frames' (or fallback 'lag_time') in all_data; cannot group by lag.")
    if "bin_center" not in df.columns:
        raise KeyError("No 'bin_center' in all_data; cannot replot curves.")

    # to sort for smooth lines and save the (possibly augmented) table
    df = df.sort_values(["lag_time_frames", "bin_center"])
    df.to_csv(csv_filename, index=False)

    #plotingt
    plt.figure(figsize=(8, 5))

    # to group by frames so legends are stable; format label with both units
    for fr, sub in df.groupby("lag_time_frames"):
        x = sub["bin_center"].to_numpy()
        y = sub[pcol].to_numpy()

        # Main curve
        plt.plot(x, np.clip(y, clip_eps_for_plot, None),
                 label=f"Δt = {sub['lag_time_s'].iloc[0]:.3f}s ({int(fr)} fr)")

        # Dashed curve: use stored 'gaussian_fit' if available and we don't refit
        do_refit = refit_sigma or ("gaussian_fit" not in sub.columns)
        if do_refit:
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                g = gauss(x, H, A, mu, sigma)
                plt.plot(x, np.clip(g, clip_eps_for_plot, None), "--",
                         label=f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma:.2f}")
            except Exception:
                # if fitting fails, skip dashed curve 
                pass
        else:
            g = sub["gaussian_fit"].to_numpy()
            # try to recover sigma to annotate legend (non-fatal if it fails)
            sigma_lbl = None
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                sigma_lbl = f"{sigma:.2f}"
            except Exception:
                pass
            if sigma_lbl is None:
                plt.plot(x, np.clip(g, clip_eps_for_plot, None), "--",
                         label=f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s")
            else:
                plt.plot(x, np.clip(g, clip_eps_for_plot, None), "--",
                         label=f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma_lbl}")
                


    # sanity check
    try:
        xvals = df["bin_center"].to_numpy()
        yvals = df[pcol].to_numpy()
        # approximate PDF-based sample by weighting x by P(x)
        pseudo_dx = np.repeat(xvals, np.maximum((yvals * 1e4).astype(int), 1))
        vh_assert_consistency(pseudo_dx)
    except Exception as e:
        print(f"[vh_check] skipped: {e}")


    # to match the pooled function styling
    plt.xscale("log")
    plt.yscale("log")
 # tighter range
    plt.xlabel("Scaled |Δx|")
    plt.ylabel("P(|Δx|)")
    plt.title("Van Hove (log–log) per lag with Gaussian fits")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

def save_van_hove_results_abs(
    all_data,
    csv_filename="Table_vanHove_abs.csv",
    fig_filename="Figure_vanHove_abs.png",
    dt=0.025,                 # only used for labels if lag_time_s is missing
    refit_sigma=False,        # True => refit sigma from (x,y) instead of trusting stored gaussian_fit
    clip_eps_for_plot=1e-12   # avoid log(0) on display; CSV stays unmodified
):
    """
    Replot one-sided Van Hove (|Δx|) results to match the in-function log–log look.
    Works with data produced by the new pooled_* function(s).

    Expects (rows) with:
      - 'lag_time_frames' (int) and/or 'lag_time_s' (float)
      - 'bin_center'
      - 'P(|Δx|)'  OR 'P(Δx)'    (probability density column name)
      - optional 'gaussian_fit'

    Behavior:
      - Sorts by (lag, bin_center) for smooth lines
      - Labels show both seconds and frames: Δt = {secs:.3f}s ({frames} fr)
      - Plots log–log axes
      - If refit_sigma=True OR no 'gaussian_fit' column, refits Gaussian for dashed curve
      - Otherwise uses stored 'gaussian_fit' as the dashed curve
    """
    if not all_data:
        print("No data to save.")
        return

    # to convert + pick probability column name robustly
    df = pd.DataFrame(all_data).copy()

    # Udp: make saver compatible with all Van Hove function versions ---
    # Older versions used "lag_time", newer ones use "lag_time_s" or "lag_time_frames".
    if "lag_time" not in df.columns:
        if "lag_time_s" in df.columns:
            df["lag_time"] = df["lag_time_s"]
        elif "lag_time_frames" in df.columns:
            df["lag_time"] = df["lag_time_frames"]
        else:
            raise KeyError(
                "Expected one of 'lag_time', 'lag_time_s', or 'lag_time_frames' in the data."
            )
    # 

    pcol = None
    for cand in ("P(|Δx|)", "P(Δx)", "P(|dx|)", "P(dx)"):
        if cand in df.columns:
            pcol = cand
            break
    if pcol is None:
        raise KeyError("Expected a probability column like 'P(|Δx|)' or 'P(Δx)' not found in all_data.")

    # to ensure lag columns exist (at least frames; seconds for nicer labels)
    has_frames = "lag_time_frames" in df.columns
    has_secs   = "lag_time_s" in df.columns
    if not has_frames and "lag_time" in df.columns:
        # back-compat: older code used 'lag_time' (frames)
        df["lag_time_frames"] = df["lag_time"].astype(int)
        has_frames = True
    if not has_secs and has_frames:
        df["lag_time_s"] = df["lag_time_frames"] * float(dt)
        has_secs = True

    if not has_frames:
        raise KeyError("No 'lag_time_frames' (or fallback 'lag_time') in all_data; cannot group by lag.")
    if "bin_center" not in df.columns:
        raise KeyError("No 'bin_center' in all_data; cannot replot curves.")

    # to sort for smooth lines and save the (possibly augmented) table
    df = df.sort_values(["lag_time_frames", "bin_center"])
    df.to_csv(csv_filename, index=False)

    #plotingt
    plt.figure(figsize=(8, 5))

    # to group by frames so legends are stable; format label with both units
    for fr, sub in df.groupby("lag_time_frames"):
        x = sub["bin_center"].to_numpy()
        y = sub[pcol].to_numpy()

        # Main curve
        line, = plt.plot(  #new
            x,
            np.clip(y, clip_eps_for_plot, None),
            label=f"Δt = {sub['lag_time_s'].iloc[0]:.3f}s ({int(fr)} fr)"
        )  #new
        color = line.get_color()  #new

        # Dashed curve: use stored 'gaussian_fit' if available and we don't refit
        do_refit = refit_sigma or ("gaussian_fit" not in sub.columns)
        if do_refit:
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                g = gauss(x, H, A, mu, sigma)
                plt.plot(
                    x,
                    np.clip(g, clip_eps_for_plot, None),
                    "--",                     #new
                    color=color,             #new same color as data
                    label=f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma:.2f}"
                )  #new
            except Exception:
                # if fitting fails, skip dashed curve 
                pass
        else:
            g = sub["gaussian_fit"].to_numpy()
            # try to recover sigma to annotate legend (non-fatal if it fails)
            sigma_lbl = None
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                sigma_lbl = f"{sigma:.2f}"
            except Exception:
                pass
            if sigma_lbl is None:
                plt.plot(
                    x,
                    np.clip(g, clip_eps_for_plot, None),
                    "--",                 #new
                    color=color,         #new
                    label=f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s"
                )  #new
            else:
                plt.plot(
                    x,
                    np.clip(g, clip_eps_for_plot, None),
                    "--",                 #new
                    color=color,         #new
                    label=f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma_lbl}"
                )  #new
                


    # sanity check
    try:
        xvals = df["bin_center"].to_numpy()
        yvals = df[pcol].to_numpy()
        # approximate PDF-based sample by weighting x by P(x)
        pseudo_dx = np.repeat(xvals, np.maximum((yvals * 1e4).astype(int), 1))
        vh_assert_consistency(pseudo_dx)
    except Exception as e:
        print(f"[vh_check] skipped: {e}")


    # to match the pooled function styling
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-3, 10)  #new tighter y-range to highlight deviations
    plt.xlabel("Scaled |Δx|")
    plt.ylabel("P(|Δx|)")
    plt.title("Van Hove (log–log) per lag with Gaussian fits")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()



def save_van_hove_results_logScaledY_old(all_data, csv_filename="Table_vanHove.csv", fig_filename="Figure_vanHove.png"):
    """
    Save and replot two-sided Van Hove with log-scaled Y axis.
    """
    if not all_data:
        print("No data to save.")
        return
    
    df = pd.DataFrame(all_data).sort_values(["lag_time", "bin_center"])
    df.to_csv(csv_filename, index=False)

    plt.figure(figsize=(8, 5))
    for lag, sub in df.groupby("lag_time"):
        x = sub["bin_center"].to_numpy()
        y = np.clip(sub["P(Δx)"].to_numpy(), 1e-12, None)

        plt.plot(x, y, label=f"Δt = {int(lag)}")
        if "gaussian_fit" in sub.columns:
            g = np.clip(sub["gaussian_fit"].to_numpy(), 1e-12, None)
            sigma = sub["sigma"].iloc[0] if "sigma" in sub else np.nan
            fit_label = f"Fit Δt = {int(lag)}, σ={sigma:.2f}" if not np.isnan(sigma) else f"Fit Δt = {int(lag)}"
            plt.plot(x, g, "--", label=fit_label)

    plt.yscale("log")  # only Y axis log
    plt.xlabel("Scaled Δx (signed)")
    plt.ylabel("P(Δx)")
    plt.title("Van Hove (two-sided) with log-scaled Y and Gaussian fits")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

def save_van_hove_results_logScaledY_notLimited(
    all_data,
    csv_filename="Table_vanHove_logY.csv",
    fig_filename="Figure_vanHove_logY.png",
    dt=0.025,                 # used if lag_time_s missing
    refit_sigma=False,        # True => refit sigma from (x,y)
    clip_eps_for_plot=1e-12   # avoid log(0) on display; CSV unchanged
):
    """
    Replot two-sided Van Hove (signed Δx) with linear X and log-scaled Y.
    Compatible with outputs from linearLog_pooled_log_scaled_van_hove_per_lag().

    Expects rows with:
      - 'lag_time_frames' (int) and/or 'lag_time_s' (float)
      - 'bin_center'
      - 'P(Δx)'  OR 'P(|Δx|)'  (probability density column)
      - optional 'gaussian_fit'
    """
    if not all_data:
        print("No data to save.")
        return


    df = pd.DataFrame(all_data).copy()

    # Upd: make saver compatible with all Van Hove function v
    # Older versions used "lag_time", newer ones use "lag_time_s" or "lag_time_frames".
    if "lag_time" not in df.columns:
        if "lag_time_s" in df.columns:
            df["lag_time"] = df["lag_time_s"]
        elif "lag_time_frames" in df.columns:
            df["lag_time"] = df["lag_time_frames"]
        else:
            raise KeyError(
                "Expected one of 'lag_time', 'lag_time_s', or 'lag_time_frames' in the data."
            )
    # ---


    # pick probability column robustly
    pcol = None
    for cand in ("P(Δx)", "P(|Δx|)", "P(dx)", "P(|dx|)"):
        if cand in df.columns:
            pcol = cand
            break
    if pcol is None:
        raise KeyError("Expected a probability column like 'P(Δx)' or 'P(|Δx|)' not found in all_data.")

    # ensure lag columns
    if "lag_time_frames" not in df.columns:
        if "lag_time" in df.columns:  # legacy
            df["lag_time_frames"] = df["lag_time"].astype(int)
        else:
            raise KeyError("No 'lag_time_frames' (or fallback 'lag_time') in all_data.")
    if "lag_time_s" not in df.columns:
        df["lag_time_s"] = df["lag_time_frames"] * float(dt)

    # sort for smooth lines & persist the (possibly augmented) table
    df = df.sort_values(["lag_time_frames", "bin_center"])
    df.to_csv(csv_filename, index=False)

    # plot
    plt.figure(figsize=(8, 5))
    for fr, sub in df.groupby("lag_time_frames"):
        x = sub["bin_center"].to_numpy()
        y = sub[pcol].to_numpy()

        # main curve (clip only for display on logY)
        plt.plot(x, np.clip(y, clip_eps_for_plot, None),
                 label=f"Δt = {sub['lag_time_s'].iloc[0]:.3f}s ({int(fr)} fr)")

        # dashed fit
        do_refit = refit_sigma or ("gaussian_fit" not in sub.columns)
        if do_refit:
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                g = gauss(x, H, A, mu, sigma)
                plt.plot(x, np.clip(g, clip_eps_for_plot, None), "--",
                         label=f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma:.2f}")
            except Exception:
                pass
        else:
            g = sub["gaussian_fit"].to_numpy()
            # try to annotate σ (non-fatal if it fails)
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                lbl = f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma:.2f}"
            except Exception:
                lbl = f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s"
            plt.plot(x, np.clip(g, clip_eps_for_plot, None), "--", label=lbl)

    plt.yscale("log")  # linear X, log Y
    plt.xlabel("Scaled Δx (signed)")
    plt.ylabel("P(Δx)")
    # ax.set_ylim(1e-5, 10)   # tighter range
    plt.title("Van Hove (two-sided) with log-scaled Y and Gaussian fits")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

def save_van_hove_results_logScaledY(
    all_data,
    csv_filename="Table_vanHove_logY.csv",
    fig_filename="Figure_vanHove_logY.png",
    dt=0.025,                 # used if lag_time_s missing
    refit_sigma=False,        # True => refit sigma from (x,y)
    clip_eps_for_plot=1e-12   # avoid log(0) on display; CSV unchanged
):
    """
    Replot two-sided Van Hove (signed Δx) with linear X and log-scaled Y.
    Compatible with outputs from linearLog_pooled_log_scaled_van_hove_per_lag().

    Expects rows with:
      - 'lag_time_frames' (int) and/or 'lag_time_s' (float)
      - 'bin_center'
      - 'P(Δx)'  OR 'P(|Δx|)'  (probability density column)
      - optional 'gaussian_fit'
    """
    if not all_data:
        print("No data to save.")
        return


    df = pd.DataFrame(all_data).copy()

    # Upd: make saver compatible with all Van Hove function v
    # Older versions used "lag_time", newer ones use "lag_time_s" or "lag_time_frames".
    if "lag_time" not in df.columns:
        if "lag_time_s" in df.columns:
            df["lag_time"] = df["lag_time_s"]
        elif "lag_time_frames" in df.columns:
            df["lag_time"] = df["lag_time_frames"]
        else:
            raise KeyError(
                "Expected one of 'lag_time', 'lag_time_s', or 'lag_time_frames' in the data."
            )
    # ---


    # pick probability column robustly
    pcol = None
    for cand in ("P(Δx)", "P(|Δx|)", "P(dx)", "P(|dx|)"):
        if cand in df.columns:
            pcol = cand
            break
    if pcol is None:
        raise KeyError("Expected a probability column like 'P(Δx)' or 'P(|Δx|)' not found in all_data.")

    # ensure lag columns
    if "lag_time_frames" not in df.columns:
        if "lag_time" in df.columns:  # legacy
            df["lag_time_frames"] = df["lag_time"].astype(int)
        else:
            raise KeyError("No 'lag_time_frames' (or fallback 'lag_time') in all_data.")
    if "lag_time_s" not in df.columns:
        df["lag_time_s"] = df["lag_time_frames"] * float(dt)

    # sort for smooth lines & persist the (possibly augmented) table
    df = df.sort_values(["lag_time_frames", "bin_center"])
    df.to_csv(csv_filename, index=False)

    # plot
    plt.figure(figsize=(8, 5))
    for fr, sub in df.groupby("lag_time_frames"):
        x = sub["bin_center"].to_numpy()
        y = sub[pcol].to_numpy()

        # main curve (clip only for display on logY)
        line, = plt.plot(  #new
            x,
            np.clip(y, clip_eps_for_plot, None),
            label=f"Δt = {sub['lag_time_s'].iloc[0]:.3f}s ({int(fr)} fr)"
        )  #new
        color = line.get_color()  #new

        # dashed fit
        do_refit = refit_sigma or ("gaussian_fit" not in sub.columns)
        if do_refit:
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                g = gauss(x, H, A, mu, sigma)
                plt.plot(
                    x,
                    np.clip(g, clip_eps_for_plot, None),
                    "--",                 #new
                    color=color,         #new same color as main curve
                    label=f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma:.2f}"
                )  #new
            except Exception:
                pass
        else:
            g = sub["gaussian_fit"].to_numpy()
            # try to annotate σ (non-fatal if it fails)
            try:
                H, A, mu, sigma = gauss_fit(x, y)
                lbl = f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s, σ={sigma:.2f}"
            except Exception:
                lbl = f"Fit Δt = {sub['lag_time_s'].iloc[0]:.3f}s"
            plt.plot(
                x,
                np.clip(g, clip_eps_for_plot, None),
                "--",             #new
                color=color,     #new
                label=lbl
            )  #new

    plt.yscale("log")  # linear X, log Y
    plt.xlabel("Scaled Δx (signed)")
    plt.ylabel("P(Δx)")
    ax = plt.gca()                #new #If there is currently no Axes on this Figure, a new one is created using Figure.add_subplot
    ax.set_ylim(1e-5, 10)       #new tighter y-range
    plt.title("Van Hove (two-sided) with log-scaled Y and Gaussian fits")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close()

# overall

# init version/linear
data = linear_pooled_log_scaled_van_hove_per_lag(tracks_filtered) #vanhove data +cagging
save_van_hove_results_linear(data, csv_filename="Table 7: vanhove_scaled_fits_data.csv", fig_filename="Figure 7: vanhove_scaled_fits.png")

data_single = linear_pooled_log_scaled_van_hove_per_lag(single_trajs)
save_van_hove_results_linear(data_single, csv_filename="Table 12: vanhove_scaled_fits_data_single.csv", fig_filename="Figure 12: vanhove_scaled_fits_single.png")

data_double = linear_pooled_log_scaled_van_hove_per_lag(double_trajs)
save_van_hove_results_linear(data_double, csv_filename="Table 13: vanhove_scaled_fits_data_double.csv", fig_filename="Figure 13: vanhove_scaled_fits_double.png")


# log-log for the one side VanHove
# data, Rg_hoppers, Rg_non_hoppers = pooled_log_scaled_van_hove_per_lag(tracks_filtered) #vanhove data +cagging
# save_van_hove_results_abs(data, csv_filename="Table 29: log-log_1side_vanhove_scaled_fits_data.csv", fig_filename="Figure 29: vanhove_scaled_fits.png")

# data_single, Rg_hoppers_single, Rg_non_hoppers_single = pooled_log_scaled_van_hove_per_lag(single_trajs)
# save_van_hove_results_abs(data_single, csv_filename="Table 30: log-log_1side_vanhove_scaled_fits_data_single.csv", fig_filename="Figure 30: vanhove_scaled_fits_single.png")

# data_double, Rg_hoppers_double, Rg_non_hoppers_double = pooled_log_scaled_van_hove_per_lag(double_trajs)
# save_van_hove_results_abs(data_double, csv_filename="Table 31: log-log_1side_vanhove_scaled_fits_data_double.csv", fig_filename="Figure 31: vanhove_scaled_fits_double.png")

# JAN
# after metrics added
track_ids_filtered = list(range(len(tracks_filtered)))

data, Rg_hoppers, Rg_non_hoppers = pooled_log_scaled_van_hove_per_lag(
    tracks_filtered,
    lags_to_plot=[0.1, 1, 30],
    bins=100,
    range_max=10.0,
    dt=0.025,
    metrics_dict=metrics,
    track_ids=track_ids_filtered,
    attach_hoppers=True
)

data_single, _, _ = pooled_log_scaled_van_hove_per_lag(
    single_trajs,
    lags_to_plot=[0.1, 1, 30],
    bins=100,
    range_max=10.0,
    dt=0.025,
    metrics_dict=None,
    attach_hoppers=False
)

data_double, _, _ = pooled_log_scaled_van_hove_per_lag(
    double_trajs,
    lags_to_plot=[0.1, 1, 30],
    bins=100,
    range_max=10.0,
    dt=0.025,
    metrics_dict=None,
    attach_hoppers=False
)




# log-linear for the two side VanHove
data = linearLog_pooled_log_scaled_van_hove_per_lag(tracks_filtered) #vanhove data +cagging
save_van_hove_results_logScaledY(data, csv_filename="Table 32: log-linear_2side_vanhove_scaled_fits_data.csv", fig_filename="Figure 32: vanhove_scaled_fits.png")

data_single = linearLog_pooled_log_scaled_van_hove_per_lag(single_trajs)
save_van_hove_results_logScaledY(data_single, csv_filename="Table 33: log-linear_2side_vanhove_scaled_fits_data_single.csv", fig_filename="Figure 33: vanhove_scaled_fits_single.png")

data_double = linearLog_pooled_log_scaled_van_hove_per_lag(double_trajs)
save_van_hove_results_logScaledY(data_double, csv_filename="Table 34: log-linear_2side_vanhove_scaled_fits_data_double.csv", fig_filename="Figure 34: vanhove_scaled_fits_double.png")


def save_rg_classified_tracks_to_csv(Rg_hoppers, Rg_non_hoppers, output_prefix="RgClassified", number=1):
    """
    Saves cage-hopping and non-hopping tracks to separate CSV files in TrackMate-like format.

    Parameters:
        Rg_hoppers (list of array): List of (N x 2) arrays for hopping tracks (x, y).
        Rg_non_hoppers (list of array): List of (N x 2) arrays for non-hopping tracks (x, y).
        output_prefix (str): Prefix for output file names.
    """
    def make_df(rg_list, label):
        records = []
        for traj_id, xy in enumerate(rg_list): #rg_traj is (N,2) or (N,>=2) with [x,y,(t)] =>xy
            # for frame, rg_val in enumerate(rg_traj):
            #     records.append({
            #         "trajectory_id": traj_id,
            #         "frame": frame,
            #         "Rg": rg_val,
            #         "class": label
            #     })
            x = xy[:, 0]
            y = xy[:, 1]
            t = xy[:, 2] if xy.shape[1] >= 3 else np.arange(len(x))
            for frame, (ti, xi, yi) in enumerate(zip(t, x, y)):
                records.append({
                    "trajectory_id": traj_id,
                    "frame": frame,
                    "time": float(ti),
                    "x": float(xi),
                    "y": float(yi),
                    "class": label
                })
        return pd.DataFrame(records)

    # creating DataFrames
    df_hop = make_df(Rg_hoppers, "HOP")
    df_nonhop = make_df(Rg_non_hoppers, "NONHOP")
    
    # saving to CSV
    df_hop.to_csv(f"Table {number}_{output_prefix}_hoppers.csv", index=False)
    df_nonhop.to_csv(f"Table {number+1}_{output_prefix}_nonhoppers.csv", index=False)

    print(f"Saved: {output_prefix}_hoppers.csv and {output_prefix}_nonhoppers.csv")

save_rg_classified_tracks_to_csv(Rg_hoppers=Rg_hoppers, Rg_non_hoppers=Rg_non_hoppers, output_prefix="overall", number=21)
# JAN commented for now
# save_rg_classified_tracks_to_csv(Rg_hoppers=Rg_hoppers_single, Rg_non_hoppers=Rg_non_hoppers_single, output_prefix="single", number=23)
# save_rg_classified_tracks_to_csv(Rg_hoppers=Rg_hoppers_double, Rg_non_hoppers=Rg_non_hoppers_double, output_prefix="double", number=25)




def old_compute_gamma_rg_from_group(group_tracks, time_step=0.025, seg_size=10):

    # mainly copied from loop above
    # to compute ensemble MSD:
    max_length = max(traj.shape[0] for traj in group_tracks)
    timewave = np.arange(max_length)
    time_ratio = 1

    msd_ensemble_sum = []

    for track in group_tracks:
        msd_ensemble_temp = calc_msd_2D_ensemble_longtrack(track[:, :2], timewave, time_ratio)

        if (
            msd_ensemble_temp is not None
            and len(msd_ensemble_temp) > 1
            and not np.isnan(msd_ensemble_temp).all()
        ):
            msd_ensemble_sum.append(msd_ensemble_temp)
        else:
            print("Skipping a track due to invalid or empty MSD.")

    ensemble_matrix = np.full((len(msd_ensemble_sum), max_length), np.nan)
    for i, msd in enumerate(msd_ensemble_sum):
        ensemble_matrix[i, :len(msd)] = msd

    msd_ensemble_mean = np.nanmean(ensemble_matrix, axis=0)
    msd_ensemble_mean[msd_ensemble_mean == 0] = np.nan

    # to compute MSD mean across individual trajectories
    msd_matrix = np.full((len(group_tracks), max_length), np.nan)
    for i, traj in enumerate(group_tracks):
        msd = calc_msd_2D_ensemble_longtrack(traj[:, :2], timewave, time_ratio)
        if msd is not None  and len(msd_ensemble_temp) > 1 and not np.isnan(msd_ensemble_temp).all():
            msd_matrix[i, :len(msd)] = msd

    msd_mean = np.nanmean(msd_matrix, axis=0)
    # debug
    print("msd_mean[:10]:", msd_mean[:10])
    print("msd_ensemble_mean[:10]:", msd_ensemble_mean[:10])
    print("Difference:", msd_ensemble_mean[:10] - msd_mean[:10])
    


    gamma = (msd_ensemble_mean - msd_mean) / msd_ensemble_mean
    relative_diff = (msd_ensemble_mean - msd_mean) / msd_ensemble_mean
    gamma_v2 = np.nanmean(relative_diff)

    lag_times = np.arange(1, len(msd_ensemble_mean) + 1) * time_step #converted to seconds
    valid_mask = (~np.isnan(gamma)) & (~np.isnan(gamma_v2)) & (~np.isnan(lag_times))

    # plt.figure()
    # plt.plot(lag_times[valid_mask], gamma[valid_mask], label='γ - v1: Rel. Diff')
    # plt.axhline(gamma_v2, linestyle='--', color='gray', label=f'γ - v3: Mean Rel. Diff (={gamma_v2:.3f})')
    # plt.xlabel('Time lag (s)')
    # plt.ylabel('γ (Non-ergodicity parameter)')
    # plt.title(f'γ vs Lag Time ({prefix})')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.grid(True, which="both", ls="--", alpha=0.5)
    # plt.legend()
    # plt.savefig(f"Figure_gamma_{prefix}.png", dpi=300)

    # gamma_df = pd.DataFrame({
    #     "lag_time_s": lag_times[valid_mask],
    #     "gamma_v1": gamma[valid_mask],
    #     "gamma_v2": gamma_v2
    # })
    # gamma_df.to_csv(f"Table_gamma_{prefix}.csv", index=False)

    # Rg calculations
    Rg_all = np.array([calc_Rg(track[:, :2]) for track in group_tracks])
    Rg_seg = [calc_Rg_seg(track[:, :2], seg_size) for track in group_tracks]
    Rg_seg_flat = np.hstack(Rg_seg)

    # plt.figure()
    # plt.hist(Rg_all, bins=50, alpha=0.5, label='Overall Rg')
    # plt.hist(Rg_seg_flat, bins=50, alpha=0.5, label='Segmented Rg')
    # plt.xlabel('Rg (units)')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.title(f'Radius of Gyration Distribution ({prefix})')
    # plt.savefig(f"Figure_rg_{prefix}.png", dpi=300)

    # pd.DataFrame({"Rg": Rg_all}).to_csv(f"Table_rg_{prefix}_all.csv", index=False)
    # pd.DataFrame({"Rg": Rg_seg_flat}).to_csv(f"Table_rg_{prefix}_segmented.csv", index=False)

    # Optionally return computed values


    print("msd_mean[:10]:", msd_mean[:10])
    print("msd_ensemble_mean[:10]:", msd_ensemble_mean[:10])
    print("gamma[:10]:", gamma[:10])
    print("Are they equal?:", np.allclose(msd_mean, msd_ensemble_mean, equal_nan=True))
    print("gamma_v2:", gamma_v2)

    print("msd_mean mean/std:", np.nanmean(msd_mean), np.nanstd(msd_mean))
    print("msd_ensemble_mean mean/std:", np.nanmean(msd_ensemble_mean), np.nanstd(msd_ensemble_mean))
    print("gamma mean/std:", np.nanmean(gamma), np.nanstd(gamma))
    
    plt.plot(msd_mean, label="MSD mean")
    plt.plot(msd_ensemble_mean, label="MSD ensemble mean")
    plt.legend()
    plt.savefig("debug_plot_msd_comparison.png")

    print("Number of tracks:", len(tracks_filtered))
    print("Lengths of first few tracks:", [len(tr) for tr in tracks_filtered[:5]])


    # to save gamma numerator and denominator for debugging
    debug_df = pd.DataFrame({
        "lag_time_s": lag_times,
        "msd_mean": msd_mean,
        "msd_ensemble": msd_ensemble_mean,
        "abs_diff": np.abs(msd_mean - msd_ensemble_mean),
        "gamma": gamma
    })

    debug_df.to_csv(f"Table _debug_gamma.csv", index=False)

    # matrix shape
    print("Matrix shapes:", msd_matrix.shape, ensemble_matrix.shape)
    print("Row 0:", msd_matrix[0, :10])
    print("Row 1:", msd_matrix[1, :10])
    print("Row 2:", ensemble_matrix[2, :10])



    return gamma, gamma_v2, Rg_all, Rg_seg_flat, msd_ensemble_mean, msd_mean, lag_times, valid_mask

def compute_gamma_rg_from_group(group_tracks, time_step=0.025, seg_size=10):

    # mainly copied from loop above
    # to compute ensemble MSD:
    max_length = max(traj.shape[0] for traj in group_tracks)
    timewave = np.arange(max_length)
    time_ratio = 1

    msd_ensemble_sum = []

    for track in group_tracks:
        msd_ensemble_temp = calc_msd_2D_ensemble_longtrack(track[:, :2], timewave, time_ratio)

        if (
            msd_ensemble_temp is not None
            and len(msd_ensemble_temp) > 1
            and not np.isnan(msd_ensemble_temp).all()
        ):
            msd_ensemble_sum.append(msd_ensemble_temp)
        else:
            print("Skipping a track due to invalid or empty MSD.")

    ensemble_matrix = np.full((len(msd_ensemble_sum), max_length), np.nan)
    for i, msd in enumerate(msd_ensemble_sum):
        ensemble_matrix[i, :len(msd)] = msd

    msd_ensemble_mean = np.nanmean(ensemble_matrix, axis=0)
    msd_ensemble_mean[msd_ensemble_mean == 0] = np.nan

    # to compute MSD mean across individual trajectories !!!!!!!!!
    skipped = 0
    msd_sum = []  # storage

    for i, traj in enumerate(group_tracks):
        # Convert frame index to time using frame column (3rd column) and scale

        if traj.shape[1] >= 3:
            timewave = (traj[:, 2] - traj[0, 2]) / 25.0  # frame-based
        else:
            timewave = np.arange(traj.shape[0]) * time_step  # uniform spacing fallback

        # timewave = (traj[:, 2] - traj[0, 2]) / 25.0  # adjust as needed to match original scale
        msd = calc_msd_2D_longtrack(traj[:, :2], timewave, time_ratio)

        if msd is not None and len(msd) > 1 and not np.isnan(msd).all():
            msd_sum.append(msd)
        else:
            skipped += 1
            print(f"Skipping track {i} due to invalid or empty MSD.")

    print(f"Skipped {skipped} tracks during MSD mean calculation.")

    # Construct MSD matrix
    max_length = max(len(msd) for msd in msd_sum)
    msd_matrix = np.full((len(msd_sum), max_length), np.nan)

    for i, msd in enumerate(msd_sum):
        msd_matrix[i, :len(msd)] = msd

    # Average across valid trajectories
    msd_mean = np.nanmean(msd_matrix, axis=0)


    # debug
    print("msd_mean[:10]:", msd_mean[:10])
    print("msd_ensemble_mean[:10]:", msd_ensemble_mean[:10])
    print("Difference:", msd_ensemble_mean[:10] - msd_mean[:10])
    

    # Ensure both MSD arrays are the same length , in case if data somehow has shorter tracks
    min_len = min(len(msd_ensemble_mean), len(msd_mean))
    msd_ensemble_mean = msd_ensemble_mean[:min_len]
    msd_mean = msd_mean[:min_len]

    gamma = (msd_ensemble_mean - msd_mean) / msd_ensemble_mean
    relative_diff = gamma.copy()
    gamma_v2 = np.nanmean(relative_diff)


    # gamma = (msd_ensemble_mean - msd_mean) / msd_ensemble_mean
    # relative_diff = (msd_ensemble_mean - msd_mean) / msd_ensemble_mean
    # gamma_v2 = np.nanmean(relative_diff)
    # print(f"gamma 1: {gamma}, gamma 2 {gamma_v2}")
    print("gamma shape:", gamma.shape)
    print("valid gamma points:", np.sum(~np.isnan(gamma)))


    lag_times = np.arange(1, len(msd_ensemble_mean) + 1) * time_step #converted to seconds
    # valid_mask = (~np.isnan(gamma)) & (~np.isnan(gamma_v2)) & (~np.isnan(lag_times))
    valid_mask = (~np.isnan(gamma)) & (~np.isnan(lag_times))


    # plt.figure()
    # plt.plot(lag_times[valid_mask], gamma[valid_mask], label='γ - v1: Rel. Diff')
    # plt.axhline(gamma_v2, linestyle='--', color='gray', label=f'γ - v3: Mean Rel. Diff (={gamma_v2:.3f})')
    # plt.xlabel('Time lag (s)')
    # plt.ylabel('γ (Non-ergodicity parameter)')
    # plt.title(f'γ vs Lag Time ({prefix})')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.grid(True, which="both", ls="--", alpha=0.5)
    # plt.legend()
    # plt.savefig(f"Figure_gamma_{prefix}.png", dpi=300)

    # gamma_df = pd.DataFrame({
    #     "lag_time_s": lag_times[valid_mask],
    #     "gamma_v1": gamma[valid_mask],
    #     "gamma_v2": gamma_v2
    # })
    # gamma_df.to_csv(f"Table_gamma_{prefix}.csv", index=False)


    # # single track test:
    # # Select a single track
    # track = group_tracks[0]  # or any other index, e.g., group_tracks[5]

    # # to generate timewave — use either of these depending on your data structure
    # timewave = np.arange(track.shape[0])  # generic, evenly spaced
    # # timewave = (track[:, 2] - track[0, 2]) / 25  # use if third column is frame/time

    # time_ratio = 1  # usually 1 unless you scale time steps

    # # to calculate MSDs
    # msd_ensemble = calc_msd_2D_ensemble_longtrack(track[:, :2], timewave, time_ratio)
    # msd_mean = calc_msd_2D_longtrack(track[:, :2], timewave, time_ratio)

    # # print first few values for inspection
    # print("msd_ensemble[:10]:", msd_ensemble[:10] if msd_ensemble is not None else "None")
    # print("msd_mean[:10]:", msd_mean[:10] if msd_mean is not None else "None")

    # # to plot both on the same graph
    # plt.figure(figsize=(6, 4))
    # if msd_mean is not None:
    #     plt.plot(msd_mean, label='Time-averaged MSD', lw=2)
    # if msd_ensemble is not None:
    #     plt.plot(msd_ensemble, label='Ensemble MSD', lw=2, linestyle='--')

    # plt.xlabel('Lag step')
    # plt.ylabel('MSD')
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.title('Single Track: Time-averaged vs Ensemble MSD')
    # plt.grid(True, which='both', ls='--', alpha=0.5)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Rg calculations
    Rg_all = np.array([calc_Rg(track[:, :2]) for track in group_tracks])
    Rg_seg = [calc_Rg_seg(track[:, :2], seg_size) for track in group_tracks]
    Rg_seg_flat = np.hstack(Rg_seg)

    return gamma, gamma_v2, Rg_all, Rg_seg_flat, msd_ensemble_mean, msd_mean, lag_times, valid_mask

def plot_and_save_gamma_rg_results(gamma, gamma_v2, Rg_all, Rg_seg_flat, msd_ensemble_mean, msd_mean, lag_times, valid_mask, prefix, N):

    #plot for gamma
    plt.figure()
    plt.plot(lag_times[valid_mask], gamma[valid_mask], label='γ - v1: Rel. Diff')
    plt.axhline(gamma_v2, linestyle='--', color='gray', label=f'γ - v3: Mean Rel. Diff (={gamma_v2:.3f})')
    plt.xlabel('Time lag (s)')
    plt.ylabel('γ (Non-ergodicity parameter)')
    plt.title(f'γ vs Lag Time ({prefix})')
    plt.xscale('log')
    plt.yscale('log') #remove this log scale
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Figure {N}_gamma_{prefix}.png", dpi=300)

    # to save γ data to CSV
    gamma_df = pd.DataFrame({
        "lag_time_s": lag_times[valid_mask],
        "gamma_v1": gamma[valid_mask],
        "gamma_v2": gamma_v2
    })
    gamma_df.to_csv(f"Table {N}_gamma_{prefix}.csv", index=False)

    # plot for Rg distributions
    plt.figure()
    plt.hist(Rg_all, bins=50, alpha=0.5, label='Overall Rg')
    plt.hist(Rg_seg_flat, bins=50, alpha=0.5, label='Segmented Rg')
    plt.xlabel('Rg (units)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f'Radius of Gyration Distribution ({prefix})')
    plt.tight_layout()
    M = N+1
    K = M +1
    plt.savefig(f"Figure{M}_rg_{prefix}.png", dpi=300)

    # to save Rg data to CSV
    pd.DataFrame({"Rg": Rg_all}).to_csv(f"Table {M}_rg_{prefix}_all.csv", index=False)
    pd.DataFrame({"Rg": Rg_seg_flat}).to_csv(f"Table {K}_rg_{prefix}_segmented.csv", index=False)

    print(f"Plots and CSVs saved for group '{prefix}'.")

# single
gamma, gamma_v2, Rg_all, Rg_seg_flat, msd_ensemble_mean, msd_mean, lag_times, valid_mask = compute_gamma_rg_from_group(single_trajs, time_step=0.025, seg_size=10)
plot_and_save_gamma_rg_results(gamma, gamma_v2, Rg_all, Rg_seg_flat, msd_ensemble_mean, msd_mean, lag_times, valid_mask, prefix = "single", N = 14)

# double
gamma, gamma_v2, Rg_all, Rg_seg_flat, msd_ensemble_mean, msd_mean, lag_times, valid_mask = compute_gamma_rg_from_group(double_trajs, time_step=0.025, seg_size=10)
plot_and_save_gamma_rg_results(gamma, gamma_v2, Rg_all, Rg_seg_flat, msd_ensemble_mean, msd_mean, lag_times, valid_mask, prefix = "double", N = 17)

# plotting random hoppers/nonhoppers
# creating plot pack, i am using global storages from above so no additional param. needed
def pack(range_max=10.0, seg_size=10, k=2.0, return_plot_samples=True, n_plot_each=3, seed=0):
    rng = random.Random(seed)
    # pick a few samples from each class (if they exist)

    # sample safely from each pool
    def pick(lst):
        return rng.sample(lst, min(n_plot_each, len(lst))) if lst else []

    h_samples         = pick(Rg_hoppers)
    n_samples         = pick(Rg_non_hoppers)
    # JAN 
    # commented for now
    # h_single_samples  = pick(Rg_hoppers_single)        # fixed: length of *_single
    # n_single_samples  = pick(Rg_non_hoppers_single)
    # h_double_samples  = pick(Rg_hoppers_double)
    # n_double_samples  = pick(Rg_non_hoppers_double)

    # store everything needed to plot later
    plot_pack = {
            "seg_size": seg_size,
            "k": k,
            "hoppers": h_samples,        # list of arrays (N,2) or (N,>=3)
            "nonhoppers": n_samples,     # list of arrays

            # JAN, commmented for now
            # "hoppers_single": h_single_samples,        # list of arrays (N,2) or (N,>=3)
            # "nonhoppers_single": n_single_samples,     # list of arrays
            # "hoppers_double": h_double_samples,        # list of arrays (N,2) or (N,>=3)
            # "nonhoppers_double": n_double_samples,     # list of arrays

        }
    
    return plot_pack


# opt. diagnostics, may be useful -> skipped, too much trouble (but considered using dbscan for cages)

# plotting from pack
def plot_from_pack_simple(plot_pack, max_per_group=None, prefix="traj"):
    """
    Minimal plotting: for each group in plot_pack, save XY path colored by time.
    Files:{group}_{cls}_{idx:03d}.png
    """

    groups = [
        ("all",    "hopper",     plot_pack.get("hoppers", [])),
        ("all",    "nonhopper",  plot_pack.get("nonhoppers", [])),
        ("single", "hopper",     plot_pack.get("hoppers_single", [])),
        ("single", "nonhopper",  plot_pack.get("nonhoppers_single", [])),
        ("double", "hopper",     plot_pack.get("hoppers_double", [])),
        ("double", "nonhopper",  plot_pack.get("nonhoppers_double", [])),
    ]

    for group_label, cls_label, traj_list in groups:
        if not traj_list:
            print(f"[skip] {group_label}/{cls_label}: none in pack")
            continue

        use_list = traj_list[:max_per_group] if (max_per_group is not None) else traj_list
        print(f"[plot] {group_label}/{cls_label}: {len(use_list)} sample(s)")

        N = 35

        for i, traj in enumerate(use_list):
            xy = traj[:, :2]
            t  = traj[:, 2] if traj.shape[1] >= 3 else np.arange(traj.shape[0])

            plt.figure(figsize=(5, 5))
            sc = plt.scatter(xy[:,0], xy[:,1], c=np.linspace(0,1,len(xy)), s=6, cmap="viridis")
            plt.plot([xy[0,0]], [xy[0,1]], 'o', ms=6, label='start')
            plt.plot([xy[-1,0]], [xy[-1,1]], 's', ms=6, label='end')
            plt.gca().set_aspect('equal', adjustable='datalim')
            plt.xlabel('x'); plt.ylabel('y')
            plt.title(f'{group_label} / {cls_label}')
            cb = plt.colorbar(sc, fraction=0.046, pad=0.02); cb.set_label('relative time')
            plt.legend(fontsize=8)
            plt.tight_layout()

            fname = f"Figure {N}_CageHoppersExamples_{prefix}_{group_label}_{cls_label}_{i:03d}.png"
            N+=1
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close()


def plot_from_pack_simple_v2(
    plot_pack,
    max_per_group=None,
    prefix="traj",
    seg_size=10,          # for boundary markers in variant B
    line_alpha=0.5,       # line transparency (variant B)
    line_lw=0.8,          # line width (variant B)
    boundary_every=1,     # label every k-th boundary (variant B)
    start_N=35            # starting figure counter to match naming
):
    """
    For each group in plot_pack, save two files per trajectory:
      A) Figure {N}_CageHoppersExamples_{prefix}_{group}_{cls}_{idx:03d}_scatter.png
      B) Figure {N+1}_CageHoppersExamples_{prefix}_{group}_{cls}_{idx:03d}_scatter_line_segments.png
    """

    groups = [
        ("all",    "hopper",     plot_pack.get("hoppers", [])),
        ("all",    "nonhopper",  plot_pack.get("nonhoppers", [])),
        ("single", "hopper",     plot_pack.get("hoppers_single", [])),
        ("single", "nonhopper",  plot_pack.get("nonhoppers_single", [])),
        ("double", "hopper",     plot_pack.get("hoppers_double", [])),
        ("double", "nonhopper",  plot_pack.get("nonhoppers_double", [])),
    ]

    N = start_N

    for group_label, cls_label, traj_list in groups:
        if not traj_list:
            print(f"[skip] {group_label}/{cls_label}: none in pack")
            continue

        use_list = traj_list[:max_per_group] if (max_per_group is not None) else traj_list
        print(f"[plot] {group_label}/{cls_label}: {len(use_list)} sample(s)")

        for i, traj in enumerate(use_list):
            xy = traj[:, :2]
            t  = traj[:, 2] if traj.shape[1] >= 3 else np.arange(traj.shape[0])
            rel = np.linspace(0, 1, len(xy))

            # Variant A: scatter only
            plt.figure(figsize=(5, 5))
            sc = plt.scatter(xy[:,0], xy[:,1], c=rel, s=6, cmap="viridis")
            plt.plot([xy[0,0]], [xy[0,1]], 'o', ms=6, label='start')
            plt.plot([xy[-1,0]], [xy[-1,1]], 's', ms=6, label='end')
            plt.gca().set_aspect('equal', adjustable='datalim')
            plt.xlabel('x'); plt.ylabel('y')
            plt.title(f'{group_label} / {cls_label}')
            cb = plt.colorbar(sc, fraction=0.046, pad=0.02); cb.set_label('relative time')
            plt.legend(fontsize=8); plt.tight_layout()

            fnameA = f"Figure {N}_CageHoppersExamples_{prefix}_{group_label}_{cls_label}_{i:03d}_scatter.png"
            plt.savefig(fnameA, dpi=300, bbox_inches="tight")
            plt.close()
            N += 1

            # Variant B: scatter + thin line + segment boundaries
            plt.figure(figsize=(5, 5))
            sc = plt.scatter(xy[:,0], xy[:,1], c=rel, s=6, cmap="viridis")
            plt.plot(xy[:,0], xy[:,1], '-', lw=line_lw, alpha=line_alpha)  # path line
            plt.plot([xy[0,0]], [xy[0,1]], 'o', ms=6, label='start')
            plt.plot([xy[-1,0]], [xy[-1,1]], 's', ms=6, label='end')

            # segment boundaries (every seg_size)
            if seg_size and seg_size > 0 and len(xy) > seg_size:
                bounds = np.arange(seg_size, len(xy), seg_size)
                plt.scatter(xy[bounds,0], xy[bounds,1], s=24,
                            facecolors='none', edgecolors='k', linewidths=0.8,
                            label='segment boundary')
                if boundary_every >= 1:
                    for k, idx_b in enumerate(bounds):
                        if (k % boundary_every) != 0:
                            continue
                        plt.annotate(str(k+1), (xy[idx_b,0], xy[idx_b,1]),
                                     textcoords="offset points", xytext=(2,2),
                                     fontsize=7, alpha=0.7)

            plt.gca().set_aspect('equal', adjustable='datalim')
            plt.xlabel('x'); plt.ylabel('y')
            plt.title(f'{group_label} / {cls_label} (line + segment boundaries)')
            cb = plt.colorbar(sc, fraction=0.046, pad=0.02); cb.set_label('relative time')
            plt.legend(fontsize=8); plt.tight_layout()

            fnameB = f"Figure {N}_CageHoppersExamples_{prefix}_{group_label}_{cls_label}_{i:03d}_scatter_line_segments.png"
            plt.savefig(fnameB, dpi=300, bbox_inches="tight")
            plt.close()
            N += 1



pp = pack(seg_size=10, k=2.0, n_plot_each=3, seed=0)   #sampler
plot_from_pack_simple_v2(pp, max_per_group=3, prefix="traj")

# JAN

# ---------- FINAL: master metrics table ----------
out_dir = "diagnostics_output"
os.makedirs(out_dir, exist_ok=True)

master_df = pd.DataFrame(list(metrics.values()))

# (optional but recommended) keep only rows where key labels exist + msd_mag exists
required_cols = ["msd_mag_1s", "is_hopper", "is_alpha_fast"]
missing_cols = [c for c in required_cols if c not in master_df.columns]
if missing_cols:
    print("[master_df] WARNING: missing columns:", missing_cols)
else:
    master_df = master_df[
        np.isfinite(master_df["msd_mag_1s"]) &
        (master_df["is_hopper"].isin([0, 1])) &
        (master_df["is_alpha_fast"].isin([0, 1]))
    ].copy()

# save full master table
master_path = os.path.join(out_dir, "master_metrics_table.csv")
master_df.to_csv(master_path, index=False)
print(f"[master_df] saved: {master_path}  (rows={len(master_df)})")

# quick sanity counts (super useful at 2am)
if "is_hopper" in master_df.columns:
    print("[master_df] hopper count:", int((master_df["is_hopper"] == 1).sum()))
if "is_alpha_fast" in master_df.columns:
    print("[master_df] alpha_fast count:", int((master_df["is_alpha_fast"] == 1).sum()))
if all(c in master_df.columns for c in ["is_hopper", "is_alpha_fast"]):
    print("[master_df] cross-tab (hopper vs alpha_fast):")
    print(pd.crosstab(master_df["is_hopper"], master_df["is_alpha_fast"]))

print(f"Total time: {time.time() - global_start:.2f}s")