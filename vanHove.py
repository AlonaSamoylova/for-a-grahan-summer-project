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

global_start = time.time()
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
    plt.savefig("Figure 1: mean_msd_loglog.png", dpi=300)
    # plt.show()

    msd_df = pd.DataFrame({"time_s": np.arange(1, len(msd_mean) + 1) * 0.025, "mean_msd": msd_mean })
    msd_df.to_csv("Table 1: mean_msd_loglog.csv", index=False)

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


    # Basic clean mask
    valid = (msd_mean > 0) & (~np.isnan(msd_mean))
    t_clean = t[valid]
    msd_clean = msd_mean[valid]

    # # Then apply 1s cutoff AFTER cleaning
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
    slope_single, intercept_single = single_powerlaw_fit(msd_clean)
    msd_fit_single = 10**intercept_single * (t_clean ** slope_single)


    log_time = np.log10(t_clean)
    

    # new 2 segment fit:
    # --- 2-segment continuous fit using bkn_pow_2seg ---
    break1 = find_turning_point(msd_clean)  # automatic turning point
    A_guess = np.mean(msd_clean[:5])
    initial_guess = [A_guess, 0.3, 1.0]
    bounds_2seg = ([1e-5, 0.1, 0.1], [10, 3.0, 3.0])

    def fit_wrapper(x, A, alpha1, alpha2):
        return bkn_pow_2seg(x, A, alpha1, alpha2, break1)[0]

    popt_2seg, _ = curve_fit(fit_wrapper, t_clean, msd_clean, p0=initial_guess, bounds=bounds_2seg)
    msd_fit_2seg, A2_2seg = bkn_pow_2seg(t_clean, *popt_2seg, break1)




    # plotting original MSD with fits
    plt.figure()
    plt.plot(t_clean * 0.025, msd_fit_2seg, '--', label=f'2-Seg Fit (α₁ ≈ {popt_2seg[1]:.2f}, α₂ ≈ {popt_2seg[2]:.2f})')

    plt.plot(t_clean * 0.025, msd_clean, label='Original MSD', color='black')
    plt.plot(t_clean * 0.025, msd_fit_single, '--', label=f'Single Power Law (α ≈ {slope_single:.2f})')
    plt.plot(t_clean * 0.025, msd_fit_2seg, '--', label=f'2-Seg Fit (α₁ ≈ {popt_2seg[1]:.2f}, α₂ ≈ {popt_2seg[2]:.2f})')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('MSD')
    plt.title('MSD Fit Comparison')
    plt.legend()
    plt.savefig("Figure4: msd_fit_comparison.png", dpi=300)
    # plt.show()

    # to print MSD exponent and magnitude
    # print(f"[Single Power Law] α = {slope_single:.3f}, A = {10**intercept_single:.3e}")
    # print(f"[2-Seg Continuous Fit] α₁ = {popt_2seg[1]:.3f}, A₁ = {popt_2seg[0]:.3e}, α₂ = {popt_2seg[2]:.3f}, A₂ = {A2_2seg:.3e}, Break = {break1}")


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
    all_data = []

    if len(tracks) >= 10:
        print("Plotting MSD for 10 longest trajectories...")

        # to sort by trajectory length and pick 10 longest
        traj_lengths = [(i, len(traj)) for i, traj in enumerate(tracks)]
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
            time = np.arange(len(msd)) * dt

            # clean MSD: remove NaNs and nonpositive values; just to ensure
            mask = ~np.isnan(msd) & (msd > 0)
            t_clean = time[mask]
            msd_clean = np.array(msd)[mask]

            if len(msd_clean) < 5:
                continue

            # single power-law fit
            slope_single, intercept_single = single_powerlaw_fit(msd_clean)
            msd_fit_single = 10**intercept_single * (t_clean ** slope_single)

            # two-segment power-law fit
            break1 = find_turning_point(msd_clean)
            A_guess = np.mean(msd_clean[:5])
            initial_guess = [A_guess, 0.3, 1.0]
            bounds_2seg = ([1e-5, 0.1, 0.1], [10, 3.0, 3.0])

            def fit_wrapper(x, A, alpha1, alpha2):
                return bkn_pow_2seg(x, A, alpha1, alpha2, break1)[0]

            try:
                popt_2seg, _ = curve_fit(fit_wrapper, t_clean, msd_clean, p0=initial_guess, bounds=bounds_2seg)
                msd_fit_2seg, A2_2seg = bkn_pow_2seg(t_clean, *popt_2seg, break1)
            except Exception as e:
                print(f"Fit failed for trajectory {idx}: {e}")
                continue

            # storing for CSV
            for t, o, s, d in zip(t_clean, msd_clean, msd_fit_single, msd_fit_2seg):
                all_data.append({
                    "trajectory": idx,
                    "time_s": t,
                    "msd_original": o,
                    "msd_fit_single": s,
                    "msd_fit_2seg": d
                })

            # plot
            plt.loglog(t_clean, msd_clean, label=f'Traj {idx}', alpha=0.5)
            plt.loglog(t_clean, msd_fit_single, '--', alpha=0.7)
            plt.loglog(t_clean, msd_fit_2seg, '--', alpha=0.7)

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

        # Optional: skip short or empty tracks
        if len(msd_clean) < 10:
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

            trackmate_single_data = [] #to save data in same format : to create sigle data csv in the same format as original one
            if classification == 'single':
                    # Single power-law fit
                single += 1
                single_group.append(msd_trimmed) #To separate storage
                single_trajs.append(tracks_filtered[i][:, :2])  # keep full x, y trajectory
                # # try:
                #     slope, intercept = single_powerlaw_fit(msd_trimmed)
                #     msd_fit_single = 10**intercept * (t ** slope)
                #     # plt.plot(t, msd_fit_single, '--', label=f"1-seg Fit T{i+1} (α ≈ {slope:.2f})")
                # except Exception as e:
                #     # print(f"Skipping 1-seg fit for Track {i+1}: {e}")
                #     pass

                # # print(f"Skipping plot for Track {i+1} (classified as single power law)")

                 # TrackMate-style storage
                for frame_idx, (x, y) in enumerate(tracks_filtered[i][:, :2]):
                    trackmate_single_data.append({
                        "TRACK_ID": single,
                        "POSITION_X": x,
                        "POSITION_Y": y,
                        "POSITION_T": frame_idx
                    })


                continue  # Skip the rest of the loop for this track
            
            else:
                
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
                

                # print(f'Tract #{i+1} MSD. Turning point is: {break1}')

                # to store data in a single dataframe
                df = pd.DataFrame({
                    "track": i + 1,
                    "frame": t,
                    "time_s": t * 0.025,
                    "msd": msd_trimmed,
                    "fit_2seg": msd_fit_2seg
                })
                all_fit_data.append(df)


        except Exception as e:
            # print(f"Skipping 2-seg fit for Track {i+1}: {e}")
            pass

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

    # saving single traj. data
    df_single_export = pd.DataFrame(trackmate_single_data)
    df_single_export.to_csv("Table 20_single_tracks_exported.csv", index=False)
    print("Saved TrackMate-style single trajectories to 'single_tracks_exported.csv'")

    
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


# van hove for x, custom
tracks_filtered, single_trajs, double_trajs = CalcMSD(path)

def linear_pooled_log_scaled_van_hove_per_lag(tracks, lags_to_plot=[1, 30, 100], bins=100, range_max=10.0): #15,30,46
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

# same as above but with log scale + added cage hopping
def pooled_log_scaled_van_hove_per_lag(tracks, lags_to_plot=[1, 30, 100], bins=100, range_max=10.0):
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
        H, A, mu, sigma = gauss_fit(bin_centers, hist)
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
    plt.savefig("Figure 7: vanhove_log_scaled_fits.png", dpi=300)


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
        threshold = mean_Rg + 2 * std_Rg  #mean + 2×std.

        if np.any(Rg_values > threshold):
            Rg_hoppers.append(traj)
        else:
            Rg_non_hoppers.append(traj)

    return all_data, Rg_hoppers, Rg_non_hoppers



# to save all plots instead of 1 by 1

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

# try change y-axis to log scale (instead of linear) - highlight differences

# overall
data, Rg_hoppers, Rg_non_hoppers = pooled_log_scaled_van_hove_per_lag(tracks_filtered) #vanhove data +cagging
save_van_hove_results(data, csv_filename="Table 7: vanhove_scaled_fits_data.csv", fig_filename="Figure 7: vanhove_scaled_fits.png")

data_single, Rg_hoppers_single, Rg_non_hoppers_single = pooled_log_scaled_van_hove_per_lag(single_trajs)
save_van_hove_results(data_single, csv_filename="Table 12: vanhove_scaled_fits_data_single.csv", fig_filename="Figure 12: vanhove_scaled_fits_single.png")

data_double, Rg_hoppers_double, Rg_non_hoppers_double = pooled_log_scaled_van_hove_per_lag(double_trajs)
save_van_hove_results(data_double, csv_filename="Table 13: vanhove_scaled_fits_data_double.csv", fig_filename="Figure 13: vanhove_scaled_fits_double.png")


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
        for traj_id, rg_traj in enumerate(rg_list):
            for frame, rg_val in enumerate(rg_traj):
                records.append({
                    "trajectory_id": traj_id,
                    "frame": frame,
                    "Rg": rg_val,
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
save_rg_classified_tracks_to_csv(Rg_hoppers=Rg_hoppers_single, Rg_non_hoppers=Rg_non_hoppers_single, output_prefix="single", number=23)
save_rg_classified_tracks_to_csv(Rg_hoppers=Rg_hoppers_double, Rg_non_hoppers=Rg_non_hoppers_double, output_prefix="double", number=25)




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


def plot_and_save_gamma_rg_results(gamma, gamma_v2, Rg_all, Rg_seg_flat, msd_ensemble_mean, msd_mean, lag_times, valid_mask, prefix, N):

    #plot for gamma
    plt.figure()
    plt.plot(lag_times[valid_mask], gamma[valid_mask], label='γ - v1: Rel. Diff')
    plt.axhline(gamma_v2, linestyle='--', color='gray', label=f'γ - v3: Mean Rel. Diff (={gamma_v2:.3f})')
    plt.xlabel('Time lag (s)')
    plt.ylabel('γ (Non-ergodicity parameter)')
    plt.title(f'γ vs Lag Time ({prefix})')
    plt.xscale('log')
    plt.yscale('log')
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

print(f"Total time: {time.time() - global_start:.2f}s")