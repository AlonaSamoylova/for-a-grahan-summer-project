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

import uuid

# Backup original show
for i in plt.get_fignums():
    fig = plt.figure(i)
    fig.savefig(f"plot_{i}.png")
    plt.close(fig)


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
    end = int(0.9 * len(slope_change)) #appears that 50% (-30%) filtered towards the end workeed better

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


# main function;
# to process and analyze track data

def CalcMSD(folder_path, min_length=50, time_ratio=2, seg_size=10): #enlarge min length -> 100, 200, van hoff corelation function
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

    # plotting, MSD curve in log-log scale
    plt.figure()
    plt.plot(np.arange(1, len(msd_mean) + 1) * 0.025, msd_mean)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([5E-3, 1E-1])
    plt.title('Mean MSD over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('MSD')
    plt.show()

    # now, let's calculate ensemble MSD and non-ergodicity
    msd_ensemble_sum = [] #storage

    ensemble_start = time.time()

    for track in tracks_filtered:  #same as above but using second function, fix added
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
    print(gamma)

    # to calculate Rg
    Rg_all = np.array([calc_Rg(track[:, :2]) for track in tracks_filtered]) #for each track
    Rg_seg = [calc_Rg_seg(track[:, :2], seg_size) for track in tracks_filtered] #fro segmented tracks

    # plotting Rg distributions
    plt.figure()
    plt.hist(Rg_all, bins=50, alpha=0.5, label='Overall Rg')
    plt.hist(np.hstack(Rg_seg), bins=50, alpha=0.5, label='Segmented Rg')
    plt.xlabel('Rg (units)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Radius of Gyration Distribution')
    plt.show()

    # output gamma
    print('Non-ergodicity parameter (gamma):', gamma)

    print(f"Total processing time: {time.time() - start_time:.2f} seconds.")

    # testing differents fits, for now: only two new functions \(single break point)

    #as valid time was not defined in init. given msd two-step (which i included in the later functions i wrote),let's do it again here
    #  def. the time vector -copied
    t = np.arange(1, len(msd_mean) + 1) #as time var.; (local ) is used to check the exec. of code(debugging) => otherwise causes errors

    # # ensuring MSD has no zeros or NaNs + t<=1 s
    # mask = int(1 / 0.025)  # 1s in frame units → 40
    # valid = (msd_mean > 0) & (~np.isnan(msd_mean)) & (t <= mask)
    # time_valid = t[valid]
    # msd_valid = msd_mean[valid] #now we have valid time to use


    # Basic clean mask
    valid = (msd_mean > 0) & (~np.isnan(msd_mean))
    t_clean = t[valid]
    msd_clean = msd_mean[valid]

    # Then apply 1s cutoff AFTER cleaning
    cutoff_mask = (t_clean * 0.025 <= 1.0)  # apply seconds-based filter
    time_valid = t_clean[cutoff_mask]
    msd_valid = msd_clean[cutoff_mask]

    #
    # # we're interedsted in everything <= 1s: not were effective as well as if loops, let's edit valid instead
    # mask = time_valid_unfiltered <= 1.0
    # time_valid = time_valid_unfiltered[mask]
    # msd_valid = msd_valid_unfiltered[mask]  # Filtered to match time_valid
    

    turning_pt = 30 #for msd two-step (fixed), that's why broken power law with automatic one is better

    # single power-law fit
    slope_single, intercept_single = single_powerlaw_fit(msd_valid)
    msd_fit_single = 10**intercept_single * (time_valid ** slope_single)

    #two step
    # (slope1, slope2), c1, c2 = MSD_exp_twostep(msd_valid, turning_pt) if interceptswill be later added , but matlab file doesn't have it like this
    # (slope1, intercept1), (slope2, intercept2) = MSD_exp_twostep(msd_valid, turning_pt)
    log_time = np.log10(time_valid)
    # # intercept1 = c1[1]
    # # intercept2 = c2[1]

    msd_fit_twostep = np.empty_like(msd_valid)
    # msd_fit_twostep[:turning_pt] = 10**(slope1 * log_time[:turning_pt])
    # msd_fit_twostep[turning_pt:] = 10**(slope2 * log_time[turning_pt:])
    (slope1, intercept1), (slope2, intercept2) = MSD_exp_twostep(msd_valid, turning_pt)
    fit1 = slope1 * log_time[:turning_pt] + intercept1
    fit2 = slope2 * log_time[turning_pt:] + intercept2
    msd_fit_twostep[:turning_pt] = 10**fit1
    msd_fit_twostep[turning_pt:] = 10**fit2


    # broken power-law fit
    (slope1, slope2), turn = broken_powerlaw_fit(msd_valid)
    msd_fit_broken = np.empty_like(msd_valid, dtype=float)
    log_time = np.log10(time_valid)
    log_msd = np.log10(msd_valid) #for 3 steo

    # fitting segments
    fit1 = slope1 * log_time[:turn] + (np.log10(msd_valid[:turn]).mean() - slope1 * log_time[:turn].mean())
    fit2 = slope2 * log_time[turn:] + (np.log10(msd_valid[turn:]).mean() - slope2 * log_time[turn:].mean())

    # combining
    msd_fit_broken[:turn] = 10**fit1
    msd_fit_broken[turn:] = 10**fit2


    # 3 step
    (s1, s2, s3), (b1, b2) = broken_powerlaw_fit_3step(msd_valid)
    msd_fit_broken_3 = np.empty_like(msd_valid)

    msd_fit_broken_3[:b1] = 10 ** (s1 * log_time[:b1] + (log_msd[:b1].mean() - s1 * log_time[:b1].mean()))
    msd_fit_broken_3[b1:b2] = 10 ** (s2 * log_time[b1:b2] + (log_msd[b1:b2].mean() - s2 * log_time[b1:b2].mean()))
    msd_fit_broken_3[b2:] = 10 ** (s3 * log_time[b2:] + (log_msd[b2:].mean() - s3 * log_time[b2:].mean()))

    # bkn_pow

    #  Initial guess: slightly off from truth
    guess_alphas = [0.22, 1.11, 1.17] #from 3 step - graph, which suggests three segments: subdiffusive → nearly diffusive → slightly superdiffusive -> helpful to select bonds
    N = len(time_valid)
    # guess_breaks = [15, 50]  #just random # ex.
    guess_breaks = [int(N * 0.3), int(N * 0.7)] #def. dynamically   !important


    # guess_A = msd_valid[0]  # or just 1e-2
    guess_A = np.mean(msd_valid[:3])

    initial_guess = [guess_A] + guess_alphas + guess_breaks

    
    # initial_guess = guess_alphas + guess_breaks

    # Bounds: enforce breaks in increasing order and alphas positive
    # lower_bounds = [0.01] * 3 + [5, 30] #random ref.
    # upper_bounds = [2.0] * 3 + [40, 90]

    bounds = (
    [1e-5, 0.1, 0.1, 0.1, int(N*0.2), int(N*0.5)],
    [1.0, 3.0, 3.0, 3.0,  int(N*0.5), int(N*0.9)],
    ) #!important, very sensitive

    
    # print("Initial guess length:", len(initial_guess))
    # print("Expected: 1 A +", len(guess_alphas), "alphas +", len(guess_breaks), "breaks")
    # print("time_valid shape:", time_valid.shape)
    # print("msd_valid shape:", msd_valid.shape)


    # fit using curve_fit
    popt, _ = curve_fit(bkn_pow, time_valid, msd_valid, p0=initial_guess, bounds=bounds) # bounds=(lower_bounds, upper_bounds)
    msd_fit = bkn_pow(time_valid, *popt)# evaluating fit


    # plotting original MSD with fits
    plt.figure()
    plt.plot(time_valid * 0.025, msd_valid, label='Original MSD', color='black')
    plt.plot(time_valid * 0.025, msd_fit_single, '--', label=f'Single Power Law (α ≈ {slope_single:.2f})')
    plt.plot(time_valid * 0.025, msd_fit_twostep, '--', label=f'Two-Step Power Law (α₁ ≈ {slope1:.2f}, α₂ ≈ {slope2:.2f})')
    plt.plot(time_valid * 0.025, msd_fit_broken, '--', label=f'Broken Power Law (α₁ ≈ {slope1:.2f}, α₂ ≈ {slope2:.2f})')
    plt.plot(time_valid * 0.025, msd_fit_broken_3, '--', label=f'Three-Segment Power Law\n(α₁ ≈ {s1:.2f}, α₂ ≈ {s2:.2f}, α₃ ≈ {s3:.2f})')
    plt.plot(time_valid * 0.025, msd_fit, '--', label=f'Improved fit Three-Segment Power Law\n(α₁ ≈ {popt[0]:.2f}, α₂ ≈ {popt[1]:.2f}, α₃ ≈ {popt[2]:.2f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('MSD')
    plt.title('MSD Fit Comparison')
    plt.legend()
    plt.show()

    # to print MSD exponent and magnitude
    print(f"[Single Power Law] α = {slope_single:.3f}, A = {10**intercept_single:.3e}")
    print(f"[Two-Step] α₁ = {slope1:.3f}, A₁ = {10**intercept1:.3e}, α₂ = {slope2:.3f}, A₂ = {10**intercept2:.3e}")
    print(f"[Three-Step] α₁ = {s1:.3f}, α₂ = {s2:.3f}, α₃ = {s3:.3f}")  # intercepts optional here
    print(f"[Scipy Fit] α₁ = {popt[1]:.3f}, α₂ = {popt[2]:.3f}, α₃ = {popt[3]:.3f}, A = {popt[0]:.3e}")

    
    #segments, now using msd avg. (mean) - filtered
    # plot for different trajectories
    # plotting a few individual time-averaged MSD trajectories


    # Def. the cutoff for 1 second (in frames)
    max_frames = int(1.0 / 0.025)  # = 40

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

    # filter parameters storage for only double power-law classified tracks
    alpha1_double = []
    alpha2_double = []
    A1_double = []
    A2_double = []
    turning_pts_double = []

    for i, msd in enumerate(msd_sum[:70]): #random 5 trajectories, ->change to see more
        msd_trimmed_unfiltered = msd[:max_frames]
        # Filter valid indices
        valid_mask = ~np.isnan(msd_trimmed_unfiltered) & ~np.isinf(msd_trimmed_unfiltered)

        
        # slice the MSD to include only values within 1 s
        # plt.plot(msd[:max_frames], label=f"Track {i+1}")
        t_unfiltered = np.arange(1, len(msd_trimmed_unfiltered) + 1) #start
        t = t_unfiltered[valid_mask]
        msd_trimmed = msd_trimmed_unfiltered[valid_mask]

        # Optional: skip short or empty tracks
        if len(msd_valid) < 10:
            raise ValueError("Not enough valid data points.")



        # # 2-Segment
        # try:
        #     turning_pt = 20  # adjust as needed
        #     t = np.arange(1, len(msd_trimmed) + 1)  # fixes x-axis mismatch

        #     # getting log values to join 2 segments
        #     log_time = np.log10(t)
        #     log_msd = np.log10(msd_trimmed)

        #     # optional: to avoid some exceptions, i noticed that x array is often empty because of /0 division
        #     # # Safe segment selection
        #     if turning_pt <= 3 or turning_pt >= len(msd_temp) - 3:
        #         raise ValueError("Turning point too close to start or end. Adjust turning_pt.")

        #     # Clean log10 transformation
        #     time_pt = np.arange(1, len(msd_temp) + 1)
        #     msd_temp = np.array(msd_temp)

        #     # Filter out non-positive MSD values
        #     valid = (msd_temp > 0)
        #     time_pt_log = np.log10(time_pt[valid])
        #     msd_temp_log = np.log10(msd_temp[valid])

        #     # Check again
        #     if len(time_pt_log) < turning_pt or turning_pt >= len(msd_temp_log):
        #         raise ValueError("Not enough valid data before or after turning point.")

        #     # fitting both segments
        #     (slope1, intercept1), (slope2, intercept2) = MSD_exp_twostep(msd_trimmed, turning_pt)

        #     # full curve with joined line (piecewise model)
        #     msd_fit_twoseg = np.empty_like(msd_trimmed)
        #     msd_fit_twoseg[:turning_pt] = 10 ** (slope1 * log_time[:turning_pt] + intercept1)
        #     msd_fit_twoseg[turning_pt:] = 10 ** (slope2 * log_time[turning_pt:] + intercept2)

        #     # # plotting
        #     plt.plot(t, msd_trimmed, label="Original MSD", color="black")
        #     plt.plot(t, msd_fit_twoseg, '--', label=f"Two-Step Fit (α₁ ≈ {slope1:.2f}, α₂ ≈ {slope2:.2f})")

        # except Exception as e:
        #     print(f"Skipping 2-seg fit for Track {i+1}: {e}")

        

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
            print(f"Track {i+1} classified as {classification} power law. Alpha_1 is:{alpha1:.2f} and Alpha2 is:{alpha2:.2f}")


            if classification == 'single':
                    # Single power-law fit
                single += 1
                # try:
                #     slope, intercept = single_powerlaw_fit(msd_trimmed)
                #     msd_fit_single = 10**intercept * (t ** slope)
                #     # plt.plot(t, msd_fit_single, '--', label=f"1-seg Fit T{i+1} (α ≈ {slope:.2f})")
                # except Exception as e:
                #     print(f"Skipping 1-seg fit for Track {i+1}: {e}")

                # print(f"Skipping plot for Track {i+1} (classified as single power law)")
                continue  # Skip the rest of the loop for this track
            
            else:
                
                double +=1

                # add to new storage
                alpha1_double.append(alpha1)
                alpha2_double.append(alpha2)
                A1_double.append(A1)
                A2_double.append(A2)
                turning_pts_double.append(break1)

                # Plot
                plt.plot(t, msd_trimmed, label=f"Track {i+1} MSD", color="black")
                plt.plot(t, msd_fit_2seg, '--', label=f"2-Seg Fit v.2 α₁ ≈ {popt[1]:.2f}, α₂ ≈ {popt[2]:.2f}. Turning point is: {break1}")
                

                print(f'Tract #{i+1} MSD. Turning point is: {break1}')

        except Exception as e:
            print(f"Skipping 2-seg fit for Track {i+1}: {e}")

        # # Plot trajectory ; coomented for now to save time debugging van Hove
        # plt.plot(t, msd_trimmed, label=f"Track {i+1}")


        # # ! move this to/from the loop for indvidual/group plot
        # # Plot the average
        # plt.plot(time_valid, msd_valid, 'k--', linewidth=2, label="Average MSD (valid)")
        

        # # Log-log and labels
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel("Lag time (frames)")
        # plt.ylabel("MSD")
        # plt.title("Single and Two-segment Fits on MSD Trajectories")
        # plt.legend(fontsize='small', loc='upper left', ncol=2)
        # plt.tight_layout()
        # plt.show()

    print (f"There are {single} single tracks and {double} double tracks")

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
    plt.show()

    return tracks_filtered


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
path = './'






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


# van hove for x, custom


def van_hove_self_x(tracks, max_lag=50, bins=100, range_max=10.0):
    """
    Computes the self-part of the Van Hove correlation function along x only.

    Parameters:
        tracks (list of arrays): Each element is a 2D array (T x 2) of trajectory coordinates (x, y).
        max_lag (int): Max lag time (Δt) to compute displacements for.
        bins (int): Number of bins for the histogram.
        range_max (float): Maximum displacement (Δx) included in histogram.

    Returns:
        vh_x (ndarray): 2D array (max_lag x bins) of P(Δx, Δt).
        bin_edges (ndarray): Bin edges used for the histograms.
    """
    # def. histogram bin edges from -range_max to +range_max
    bin_edges = np.linspace(-range_max, range_max, bins + 1)
    
    # init. output matrix: each row corresponds to a lag time
    vh_x = np.zeros((max_lag, bins))
    # scale each trcaks
    # loop over all trajectories
    for traj in tracks:
        # loop over lag times from 1 to max_lag
        for lag in range(1, max_lag + 1):
            dx_list = []  # Store Δx values for this lag

            # loop over valid time indices
            for i in range(len(traj) - lag):
                dx = traj[i + lag, 0] - traj[i, 0]  # Δx
                dx_list.append(dx)

            # histogram of Δx at this lag
            hist, _ = np.histogram(dx_list, bins=bin_edges, density=True)
            # instead of density, use raw number-> keeptrack of how many segments=> acumalate raw histogram/ 'total # segments
            vh_x[lag - 1] += hist

    # normalize: divide by number of tracks
    vh_x /= len(tracks)

    return vh_x, bin_edges


def enhanced_van_hove_self_x(tracks, max_lag=50, bins=100, range_max=10.0):
    """
    Computes the self-part of the Van Hove correlation function (only x-component).
    Adds detailed shape and NaN/static checks to justify its filtering strength.

    Parameters:
        tracks (list of arrays): Each element is a 2D array of shape (T x 2) representing a trajectory.
        max_lag (int): Maximum lag time Δt to compute.
        bins (int): Number of histogram bins.
        range_max (float): Range for displacement histogram.

    Returns:
        vh (array): Van Hove matrix (lag time x histogram bin).
        bin_edges (array): Bin edges for histogram.
        summary (dict): Statistics on how many trajectories were skipped and why.
    """
    # histogram bins
    bin_edges = np.linspace(-range_max, range_max, bins + 1)
    vh = np.zeros((max_lag, bins))


    # counters for diagnostics
    total = 0
    short = 0
    nan = 0
    static = 0
    used = 0

    for traj in tracks:
        total += 1

        if traj.shape[0] < 2:
            short += 1
            continue

        x = traj[:, 0]
        if np.isnan(x).all():
            nan += 1
            continue

        if np.allclose(x, x[0]):
            static += 1
            continue

        used += 1


        for lag in range(1, max_lag + 1):
            displacements = [x[i + lag] - x[i] for i in range(len(x) - lag)]

            #scaling factor ξ = exp(mean(log|dx|))
            safe_log_disps = [np.log(abs(dx)) for dx in displacements if dx != 0]
            if not safe_log_disps:
                continue
            xi = np.exp(np.mean(safe_log_disps))

            # to scale displacements
            scaled_dx = [dx / xi for dx in displacements]

            # histogram the scaled displacements
            hist, _ = np.histogram(scaled_dx, bins=bin_edges, density=True)
            vh[lag - 1] += hist
            # avg across differnt pulls, not abevg across lag times; pick few discrete lag times 

        # Normalize by the number of usable trajectories
    if used > 0:
        vh /= used

    summary = {
        "total": total,
        "used": used,
        "short_skipped": short,
        "nan_skipped": nan,
        "static_skipped": static
    }
    # decided not to return summary to easier unpack it later

    # printed instead
    print(f"Custom van Hove filtering summary: Out of {total} tracks, {used} were used. "
          f"There were {short} short_skipped, {nan} nan_skipped, and {static} static_skipped tracks")
    
    return vh, bin_edges


def univ_log_scaled_van_hove_x(max_lag=1, bins=100, range_max=10.0):
    """
    Aggregates all scaled displacements across trajectories and lags into a single Van Hove distribution.
    Scaling per (trajectory, lag) using ξ = exp(mean(log(|dx|))) per the thesis method.

    Returns:
        hist: Final normalized histogram of scaled displacements.
        bin_edges: Histogram bin edges.
    """

    # getting individual tracks
    tracks= CalcMSD(path)
    all_scaled_displacements = []

    for traj in tracks:
        if traj.shape[0] < 2:
            continue
        x = traj[:, 0]
        if np.isnan(x).all() or np.allclose(x, x[0]):
            continue

        for lag in range(1, max_lag + 1): #remove this, choose 1 lag  time
            displacements = x[lag:] - x[:-lag]
            if len(displacements) == 0:
                continue #integrate the hist. within traj but not within lag times/ 

            safe_log = np.log(np.abs(displacements[displacements != 0]))
            if len(safe_log) == 0:
                continue

            xi = np.exp(np.mean(safe_log))
            scaled_dx = displacements / xi
            all_scaled_displacements.extend(scaled_dx)

    # histogram of all scaled displacements
    bin_edges = np.linspace(-range_max, range_max, bins + 1)
    hist, _ = np.histogram(all_scaled_displacements, bins=bin_edges, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # gaussian fit
    mu, sigma = norm.fit(bin_centers, floc=0)
    gauss = norm.pdf(bin_centers, mu, sigma)

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, hist, label="Scaled Van Hove", lw=2)
    plt.plot(bin_centers, gauss, '--', label=f"Gaussian Fit\nμ={mu:.3f}, σ={sigma:.3f}")
    plt.xlabel("Scaled Δx")
    plt.ylabel("P(Δx)")
    plt.title("Universal Van Hove with Log-Scaling (Global Scalling factor)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return hist, bin_edges

# # to plot van hove custom function for a selected lag time

def plot_van_hove(lag=5, bins=100, range_max=1.0, smooth=False, average_lags=3):
    """
    Plots the Van Hove self-part function for a specific lag time.

    Parameters:
        lag (int): Lag time to plot (1-indexed)
        bins (int): Number of bins for histogram
        range_max (float): Max Δx value for binning (histogram width)
        smooth (bool): If True, apply Gaussian smoothing to the curves
        average_lags (int): Number of early lag times to average in the custom function
    """
    # getting individual tracks
    trajectories = CalcMSD(path)

    # custom Van Hove (Δx only)
    vh_custom, bin_edges_custom = enhanced_van_hove_self_x(trajectories, max_lag=max(lag, average_lags), bins=bins, range_max=range_max)

    #to calculate bin centers from edges; 
    centers_custom = 0.5 * (bin_edges_custom[:-1] + bin_edges_custom[1:])
    
    #averaging=> to solve weird artifact around center 
    if average_lags > 1:
        custom_curve = np.mean(vh_custom[:average_lags], axis=0)
        label_custom = f"Custom Van Hove (Δt=1–{average_lags} avg)"
    else:
        custom_curve = vh_custom[lag - 1]
        label_custom = f"Custom Van Hove (Δt={lag})"

    # gaussian fit
    # mu, sigma = norm.fit(centers_custom, floc=0)
    # gauss = norm.pdf(centers_custom, mu, sigma)

    vh, bin_edges = enhanced_van_hove_self_x(...)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_at_lag3 = vh_custom[lag - 1]  

    params = gauss_fit(bin_centers, hist_at_lag3)
    fit_curve = gauss(bin_centers, *params)



    df_tracks = convert_tracks_to_df(trajectories)
    pos_x = df_tracks.set_index(['frame', 'particle'])['x'].unstack()
    vh_tp = tp.motion.vanhove(pos_x, lag, bins=bins, mpp=1.0)

    # snoothing
    if smooth:
        custom_curve = gaussian_filter1d(custom_curve, sigma=1.0)
        vh_tp = gaussian_filter1d(vh_tp, sigma=1.0)

    plt.figure(figsize=(8, 5))
    # plt.plot(centers_custom, custom_curve, label=label_custom, lw=2)
    # plt.plot(centers_custom, gauss, '--', label=f"Gaussian Fit\nμ={mu:.3f}, σ={sigma:.3f}")
    plt.plot(bin_centers, hist_at_lag3, label="Van Hove Δt=3")
    plt.plot(bin_centers, fit_curve, '--', label="Gaussian Fit") #not working
    plt.xlabel("Δx")
    plt.ylabel("P(Δx, Δt)")
    plt.title("Comparison of Van Hove Self-Part Function")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# modified universal log-scaled van Hove function pooling displacements across lags
def pooled_log_scaled_van_hove_x(max_lag=1, bins=100, range_max=10.0):
    tracks = CalcMSD(path)
    all_scaled_displacements = []

    for traj in tracks:
        if traj.shape[0] < 2:
            continue
        x = traj[:, 0]  # x-component
        if np.isnan(x).all() or np.allclose(x, x[0]):
            continue

        for lag in range(1, max_lag + 1):
            displacements = x[lag:] - x[:-lag]
            if len(displacements) == 0:
                continue

            safe_log = np.log(np.abs(displacements[displacements != 0]))
            if len(safe_log) == 0:
                continue

            xi = np.exp(np.mean(safe_log))
            scaled_dx = displacements / xi
            all_scaled_displacements.extend(scaled_dx)

    # histogram of all scaled displacements
    bin_edges = np.linspace(-range_max, range_max, bins + 1)
    hist, _ = np.histogram(all_scaled_displacements, bins=bin_edges, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # paussian fit
    # new
    H, A, mu, sigma = gauss_fit(bin_centers, hist)
    gaussian_curve = gauss(bin_centers, H, A, mu, sigma)


    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, hist, label="Scaled Van Hove", lw=2)
    plt.plot(bin_centers, gaussian_curve, '--', label=f'Gaussian Fit\nμ={mu:.2f}, σ={sigma:.2f}')
    plt.xlabel("Scaled Δx")
    plt.ylabel("P(Δx)")
    plt.title("Universal Van Hove with Log-Scaling (Pooled Displacements)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return hist, bin_edges

# pooled_log_scaled_van_hove_x()

def pooled_log_scaled_van_hove_per_lag(lags_to_plot=[15, 30, 46], bins=100, range_max=10.0):
    tracks = CalcMSD(path)
    bin_edges = np.linspace(-range_max, range_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8, 5))

    for lag in lags_to_plot:
        all_scaled_dx = []

        for traj in tracks:
            if traj.shape[0] < lag + 1:
                continue
            x = traj[:, 0] #x component
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

    plt.xlabel("Scaled Δx")
    plt.ylabel("P(Δx)")
    plt.title("Van Hove (per lag) with Log-Scaling plotted together with Gaussian fits")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

pooled_log_scaled_van_hove_per_lag()
