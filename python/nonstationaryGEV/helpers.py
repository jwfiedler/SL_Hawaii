from imports import *

def remove_files(dirs):
    # remove best.txt and mio.txt if they exist
    if os.path.exists(dirs['run_dir'] / 'best.txt'):
        os.remove(dirs['run_dir'] / 'best.txt')
    if os.path.exists(dirs['run_dir'] / 'mio.txt'):
        os.remove(dirs['run_dir'] / 'mio.txt')


def stepwise(x_inisol, dirs, modelType='GEV_SeasonalMu'):
    N = len(x_inisol)


    cont = 0

    x = np.array([x_inisol]) 
    # x = x_inisol # Start with initial solution
    f, pa = fitness(x[cont], dirs, modelType)  # Compute fitness for initial solution
    f = np.array([f])
    pa = np.array([pa])
    
    better = True
    prob = 0.95
    
    while better:

        dum = np.where(x[cont] == 0)[0]  # Find positions of zeros in the chromosome
        
        if dum.size != 0:
            x_temp = np.tile(x[cont], (len(dum), 1))
            f_temp = np.zeros(len(dum))
            pa_temp = np.zeros(len(dum))
            
            for i in range(len(dum)):
                x_temp[i, dum[i]] = 1
                f_temp[i], pa_temp[i] = fitness(x_temp[i], dirs, modelType)
            best, indi = np.max(f_temp), np.argmax(f_temp)
            # Check if the improvement is significant with chi2 test
            if (best - f[cont]) >= (0.5 * chi2.ppf(prob, df=(pa_temp[indi] - pa[cont]))):
                cont += 1
                x = np.vstack((x, [x_temp[indi]]))
                f = np.vstack((f, [best]))
                pa = np.vstack((pa, [pa_temp[indi]]))
                f[cont], pa[cont] = fitness(x[cont], dirs, modelType)
            else:
                better = False
        else:
            better = False

    return x, f

def fitness(x, dirs, modelType='GEV_SeasonalMu'):

    run_dir = dirs['run_dir']  

    remove_files(dirs)

    
    # Load parameter limits from a file
    aux = np.loadtxt(run_dir / 'limits.txt')
    xmax = np.zeros(len(aux))  # Initialize with NaNs
    xmin = np.zeros(len(aux))
    # Initialize parameter limits based on the chromosome
    xmin[:3] = aux[:3, 0]
    xmax[:3] = aux[:3, 1]
    cont = 3
    

    if x[0] == 1:  # Annual cycle
        xmin[cont:cont+2] = [aux[3, 0], aux[4, 0]]  # should use indices 3 and 4 (i.e., aux(4,1) and aux(5,1) in MATLAB)
        xmax[cont:cont+2] = [aux[3, 1], aux[4, 1]]
        cont += 2

    if x[1] == 1:  # Semiannual cycle
        xmin[cont:cont+2] = [aux[5, 0], aux[6, 0]]  # should use indices 5 and 6 (i.e., aux(6,1) and aux(7,1) in MATLAB)
        xmax[cont:cont+2] = [aux[5, 1], aux[6, 1]]
        cont += 2

    if x[2] == 1:  # Frequency 4w
        xmin[cont:cont+2] = [aux[5, 0], aux[6, 0]]  # should use indices 5 and 6 (same as Semiannual cycle)
        xmax[cont:cont+2] = [aux[5, 1], aux[6, 1]]
        cont += 2

    if modelType == 'GEV_S_T_Cv' or modelType == 'GEV_S_T_Cv_Nodal':
        if x[2] == 1:  # Frequency 4w, for some reason the limits change, so we'll replace them
            cont -= 2
            xmin[cont:cont+2] = [aux[7, 0], aux[8, 0]]  # 
            xmax[cont:cont+2] = [aux[7, 1], aux[8, 1]]
            cont += 2
        
        if x[3] == 1: # Trend in location parameter, beta_LT
            xmin[cont] = aux[9, 0]
            xmax[cont] = aux[9, 1]
            cont += 1
        
        if x[4] == 1: # Covariate in location parameter, beta_CV
            xmin[cont] = aux[10, 0]
            xmax[cont] = aux[10, 1]
            cont += 1

        if x[5] == 1: # Covariate in scale parameter, gamma_CV
            xmin[cont] = aux[11, 0]
            xmax[cont] = aux[11, 1]
            cont += 1
    
    if modelType == 'GEV_S_T_Cv_Nodal':
        if x[6] == 1: # Nodal component in location, beta_N1, beta_N2
            xmin[cont:cont+2] = [aux[12, 0], aux[13, 0]]
            xmax[cont:cont+2] = [aux[12, 1], aux[13, 1]]
            cont += 2


    xmin = xmin[:cont]
    xmax = xmax[:cont]

    # Prepare the initial values and parameter configuration file
    n = len(xmin)
    xini = np.zeros(n) + 0.001
    xini[2] = -0.001  # Specific initialization for third parameter
    
    # Prepare parameters for the external executable
    maxn = -2000 + 1500 * n #number of iterations
    kstop = 3 #number of iterations without improvement allowed
    pcento = 0.001
    ngs = np.ceil(n / 4)
    iseed = 955
    ideflt = 1
    npg = 2 * n + 1
    nps = n + 1
    nspl = 2 * n + 1
    mings = ngs
    iniflg = 1
    iprint = 1
    
    # Write the parameter limits and initial guesses to a file
    
    with open(run_dir / 'scein.dat', 'w') as f:
        # Writing header with specified formatting
        f.write(f"{maxn:5g} {kstop:5g} {pcento:5.3g} {ngs:5g} {iseed:5g} {ideflt:5g}\n")
        f.write(f"{npg:5g} {nps:5g} {nspl:5g} {mings:5g} {iniflg:5g} {iprint:5g}\n")

        # Write chromosome values with better control over formatting
        x_formatted = " ".join(f"{int(value):1.0f}" for value in x[:-1]) + f" {int(x[-1]):1.0f}\n"
        f.write(x_formatted)

        # Writing parameter limits and initial guesses with specified precision
        for i in range(len(xini)):
            f.write(f"{xini[i]:15.8f} {xmin[i]:15.8f} {xmax[i]:15.8f}\n")        

    
    try:
        modelName = 'Model_' + modelType + '.exe'
        subprocess.run([str(modelName)], cwd=str(run_dir), check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {modelName}: {e}")
    
    # Read the output
    with open(run_dir / 'best.txt', 'r') as file:
        bestf = float(file.readline().strip())

    # Retry loop for checking if the 'mio' matrix is singular
    max_retries = 10
    retries = 0
    while retries < max_retries:
        # Run the model
        subprocess.run([modelName], cwd=str(run_dir), check=True, shell=True)
        
        # Try loading the mio matrix
        mio = np.loadtxt(run_dir / 'mio.txt')

        # Check if mio is singular (determinant is 0)
        if np.linalg.det(mio) == 0:
            retries += 1
        else:
            break
    else:
        print(f'Model failed after {max_retries} attempts, mio is still singular')

    
    return bestf, n

def get_monthly_max_time_series(recordID,rsl_hourly):

    ridIndex = np.where(rsl_hourly.record_id == recordID)[0]
    # find the station name that matches the record_id
    station_name = rsl_hourly.station_name[ridIndex].item()
    station_name = rsl_hourly.station_name[ridIndex].item()

    STNDtoMHHW = 0.001*rsl_hourly['MHHW'][ridIndex].values

    sea_level_series = 0.001*rsl_hourly['sea_level'][ridIndex]

    #get only data from 1993 to 2023
    sea_level_series = sea_level_series.sel(time=slice('1993', '2023'))

    # remove nans
    sea_level_series = sea_level_series.dropna('time')

    # Step 1: extract the monthly maxima
    monthly_max = sea_level_series.resample(time='1ME').max()

    # ensure monthly_max is column array
    monthly_max = monthly_max.squeeze()

    # get exact time of the monthly maxima
    t_monthly_max = sea_level_series.resample(time='1ME').map(lambda x: x.time[x.argmax()])

    # To ensure it's in datetime format and to access datetime properties
    t_monthly_max['time'] = pd.to_datetime(t_monthly_max.values)

    # Now extract the day of the year (using fractional days)
    t_yearDay = t_monthly_max['time'].dt.dayofyear + t_monthly_max['time'].dt.hour/24 + t_monthly_max['time'].dt.minute/1440 + t_monthly_max['time'].dt.second/86400

    # get year of t_monthly_max
    t_year = t_monthly_max['time'].dt.year
    year0 = t_year[0].item()

    # convert t_yearDay and monthly_max to float
    t_yearDay = np.array(t_yearDay)
    monthly_max = np.array(monthly_max)
    t_monthly_max = np.array(t_monthly_max)
    t_year = np.array(t_year)

    # get decimal year such that t = year_monthly_max + t_yearDay/366
    t = (t_year-t_year[0]) + t_yearDay/366

    # save t and monthly_max to data frame
    df = pd.DataFrame({'t': t, 'monthly_max': monthly_max, 't_monthly_max': t_monthly_max})

    df = df.dropna()

    return df, STNDtoMHHW, station_name, year0

def get_covariate(t_monthly_max, CI_dir, CIname='PMM'):

    df = pd.read_csv(CI_dir / 'climate_indices_norm.csv', parse_dates=['time'])

    # Set the Date as the index for easier slicing and access
    df.set_index('time', inplace=True)

    # Extract the PMM column
    CI_df = df[CIname]

    # Reindex to include all necessary dates from t_monthly_max for interpolation
    all_dates = CI_df.index.union(t_monthly_max)

    # Reindex the DataFrame with the union of dates
    CI_df = CI_df.reindex(all_dates)

    # Now interpolate 
    CI_interp_df = CI_df.interpolate()

    # retrieve the PMM value for the monthly maxima
    CI_interp = CI_interp_df.loc[t_monthly_max]

    # Save CI_interp as 'covariate' as a 1D numpy array
    covariate = CI_interp.squeeze().to_numpy()

    return covariate

def combine_datasets(model_output_dir):
    # Find all RL_muN.nc files
    file_pattern = str(model_output_dir / '**/RL_muN.nc')
    file_paths = glob.glob(file_pattern, recursive=True)

    # Initialize an empty list to hold each dataset
    datasets = []

    for file_path in file_paths:
        ds = xr.open_dataset(file_path)
        
        # Extract important attributes
        record_id = int(ds.attrs.get('record_id'))    
        station_name = ds.attrs.get('station_name')
        x = ds.attrs.get('x')  # Assuming x is stored as an attribute and has varying lengths
        
        # Ensure 'year' and 'return_level' are coordinates and are preserved
        year = ds['year'] if 'year' in ds else ds.coords.get('year')
        return_level = ds['return_level'] if 'return_level' in ds else ds.coords.get('return_level')
        
        # Expand the dataset with the 'station' dimension (coordinate)
        ds = ds.expand_dims({'record_id': [record_id]})
        
        # Store x as a variable with its own dimension (e.g., 'x_dim')
        ds['x'] = xr.DataArray(x, dims=['x_dim'])
        
        # Keep station_name as an attribute of the dataset
        ds.attrs['station_name'] = station_name
        
        # Add the dataset to the list
        datasets.append(ds)

    # Combine all datasets along the 'station' dimension
    try:
        combined_ds = xr.concat(datasets, dim='record_id', combine_attrs='override')
        print("Combined dataset created successfully.")
    except Exception as e:
        print(f"An error occurred during dataset concatenation: {e}")

    # remove attributes that are not needed
    attributes_to_remove = ['station_name', 'model_parameters', 'model_standard_error', 'model', 'x', 'record_id']
    for attr in attributes_to_remove:
        combined_ds.attrs.pop(attr, None)
        
    return combined_ds


def adjust_w_for_plotting(x, w):
    """
    Adjusts the w array based on the x conditions and returns an array with 14 elements.

    Parameters:
        x (list): A list representing the conditions that influence the adjustment of w.
        w (array-like): An array of values to be adjusted based on x.

    Returns:
        np.ndarray: A 14-element array with adjusted values.
    """
    # Initialize the output array with 14 elements set to zero
    w_s_plot = np.zeros(14)

    # Ensure icromo is 7 elements long
    icromo = np.array(x)
    if len(icromo) != 7:
        # Fill the rest with zeros
        icromo = np.append(icromo, np.zeros(7 - len(icromo)))

    # Always present terms (mu, psi, xi)
    w_s_plot[:3] = w[1:4]
    
    # Initialize the index for the input w
    idx = 4

    # Assign terms based on icromo conditions
    if icromo[0] == 1: # annual
        w_s_plot[3:5] = w[idx:idx+2]
        idx += 2
        
    if icromo[1] == 1: # semiannual
        w_s_plot[5:7] = w[idx:idx+2]
        idx += 2

    if icromo[2] == 1: #triannual
        w_s_plot[7:9] = w[idx:idx+2]
        idx += 2

    if icromo[3] == 1: # long-term trend
        w_s_plot[9] = w[idx]
        idx += 1

    if icromo[4] == 1: # covariate, location
        w_s_plot[10] = w[idx]
        idx += 1

    if icromo[5] == 1: #covariate, scale
        w_s_plot[11] = w[idx]
        idx += 1

    if icromo[6] == 1: #nodal
        w_s_plot[12:14] = w[idx:idx+2]
        idx += 2

    # Return the adjusted array with 14 elements
    return w_s_plot

def Quantilentime(x0, w, x, t00, t11, return_period, T, serieCV):
    """
    Python equivalent of the Menendez MATLAB Quantilentime function.

    Parameters:
    x0 : float
        The variable to find the root of.
    w : np.ndarray
        The current estimates of the parameters.
    x : np.ndarray
        Binary switches that indicate which parameters are active.
    t00 : float
        Start time.
    t11 : float
        End time.
    return_period : float
        Return period.
    ti : np.ndarray
        Time vector for climate index interpolation.
    serieCV : np.ndarray
        Climate index data to be interpolated.

    Returns:
    float
        The value of the function at x_value.
    """
    # ww = adjust_w_for_plotting(x, w)
    # Parameter unpacking
    b0, a0, xi, b1, b2, b3, b4, b5, b6 = w[:9] # basics + seasonal cycle
    bLT, bCI, aCI, bN1, bN2 = w[9:]  # Initialize optional parameters

    dt = 0.001
    ti = np.arange(t00, t11 + dt, dt)

    # Interpolate serieCV at ti points
    serieCV2 = np.interp(ti, T, serieCV)
    # Replace NaNs with zero (np.interp does this by default if outside the bounds)
    serieCV2[np.isnan(serieCV2)] = 0

    km = 12

    # Define mut and other parameters

    # location(t)
    mut = (b0 * np.exp(bLT * ti) +
           b1 * np.cos(2 * np.pi * ti) + b2 * np.sin(2 * np.pi * ti) +
           b3 * np.cos(4 * np.pi * ti) + b4 * np.sin(4 * np.pi * ti) +
           b5 * np.cos(8 * np.pi * ti) + b6 * np.sin(8 * np.pi * ti) +
           bN1 * np.cos((2 * np.pi / 18.61) * ti) + bN2 * np.sin((2 * np.pi / 18.61) * ti) +
           (bCI * serieCV2))
    
    # scale(t)
    psi = a0 + (aCI * serieCV2)

    # shape(t)
    xit = xi

    # factor = 0
    # for i in range(len(ti)):
    #     h = np.maximum(1 + (xit * (x0 - mut[i]) / psi[i]), 0.0001) ** (-1 / xit)
    #     factor += h

    # Vectorized factor calculation
    h = np.maximum(1 + (xit * (x0 - mut) / psi), 0.0001) ** (-1 / xit)
    factor = np.sum(h)    

    prob = 1 - (1 / return_period)

    y = -prob + np.exp(-km * factor * dt)

    return y

def derivative_first_order(k, w, x, t00, t11, return_period, T, covariate):
    h = 0.0001  # Small perturbation
    x0_initial = w[1]  # Initial value for the root-finding function

    wbest = adjust_w_for_plotting(x, w)
    # Step 1: Calculate the baseline value with the current parameters
    baseline_func = lambda x0: Quantilentime(x0, wbest, x, t00, t11, return_period, T, covariate)
    baseline_value = brentq(baseline_func, x0_initial - 1, x0_initial + 1)

    # Perturb the k-th parameter
    perturbed_w = np.copy(w)
    perturbed_w[k] += h
    wbest_perturbed = adjust_w_for_plotting(x, perturbed_w)

    # Step 2: Calculate the function value with the perturbed parameter
    perturbed_func = lambda x0: Quantilentime(x0, wbest_perturbed, x, t00, t11, return_period, T, covariate)
    perturbed_value = brentq(perturbed_func, x0_initial - 1, x0_initial + 1)

    # Step 3: Numerical derivative
    derivative = (perturbed_value - baseline_value) / h
    
    return derivative

def calculate_std(w, x, t00,t11, T,serieCV, mio, r):
    
    # Note here that "mio" stands for "Maximum Information Operator" and is same as the Hessian matrix
    # if mio is not a singular matrix, then the covariance matrix of the parameters is the inverse of the Hessian
    if np.linalg.det(mio) == 0:
        #print('Hessian matrix is singular')
        cov_params = np.eye(len(mio)) # if the Hessian is singular, we'll use the identity matrix as a placeholder
    else:
        cov_params = np.linalg.inv(mio) # the covariance matrix of the parameters is the inverse of the Hessian
    n = len(w)
    Zp1 = np.zeros(n)

    # Here we'll use a 1st order Taylor expansion to estimate the standard error
    for i in range(n):
        Zp1[i] = derivative_first_order(i, w, x, t00, t11, r, T, serieCV)

    # expand to make sure we have the parameters in the right place, rename Zp1 to J for Jacobian
    J = adjust_w_for_plotting(x, Zp1)
    wbest = adjust_w_for_plotting(x, w)

    # remove elements of J where wbest is zero (parameter not included, J should be 0 here as well anyway...)
    J = J[wbest != 0] ## is this correct? I think so...

    # Note here J is a row vector
    # propagate the covariance matrix (e.g. 1x9 * 9x9 * 9x1 = 1x1)
    cov_solution = J @ cov_params @ J.T  #should be a scalar! 

    # calculate the standard error (square root of the scalar variance)
    ic_sqrt = np.sqrt(cov_solution)

    
    return ic_sqrt.item() # return as a scalar

def getTimeDependentReturnValue(T0, serieCV, w, x, ReturnPeriod, mio):
    # ensure input is a numpy array
    T = np.array(T0)
    serieCV = np.array(serieCV)
    years = np.arange(np.floor(T[0]), np.ceil(T[-1]) + 1)
    wbest = adjust_w_for_plotting(x, w)
    YR = np.zeros((len(ReturnPeriod), len(years) - 1))
    ic_sqrt = np.zeros((len(ReturnPeriod), len(years) - 1))
    upper_confidence = np.zeros((len(ReturnPeriod), len(years) - 1))
    lower_confidence = np.zeros((len(ReturnPeriod), len(years) - 1))

    # only run if w[1] is not nan
    if np.isnan(w[1]):
        print('Return value is nan, skipping calculation')
        return years, YR, upper_confidence, lower_confidence
    
    for idx, r in enumerate(ReturnPeriod):
        x0 = w[1]  # initial value of return value in the iteration

        for i in range(len(years) - 1):
            t00 = years[i]
            t11 = years[i + 1]
            # Call the Quantilentime function for each year interval
            
            YR[idx, i] = brentq(Quantilentime, x0 - 2, x0 + 2, args=(wbest, x, t00, t11, r, T, serieCV))    
            
                
            x0 = YR[idx, i]
            # print('calculating return value for year interval:', t00, t11)  
            # Calculate the confidence intervals
            ic_sqrt[idx,i] = calculate_std(w, x, t00,t11, T,serieCV, mio, r)
            # print('ic_sqrt:', ic_sqrt[idx,i])
            upper_confidence[idx, i] = YR[idx,i] + 1.96*ic_sqrt[idx,i]
            lower_confidence[idx, i] = YR[idx,i] - 1.96*ic_sqrt[idx,i]

    return years, YR, upper_confidence, lower_confidence

def save_model_to_netcdf(x,w,mio,modelName,savepath, modelInfo):

    ReturnPeriod = modelInfo['ReturnPeriod']
    years, RL, RL_high, RL_low = getTimeDependentReturnValue(modelInfo['t'], modelInfo['covariate'], w,x, ReturnPeriod, mio)
    
    # save the time-dependent return values to a netcdf file
    df = pd.DataFrame(RL, index=ReturnPeriod, columns=years[:-1]+modelInfo['year0'])
    df.index.name = 'ReturnPeriod'
    df.columns.name = 'Year'

    df_low = pd.DataFrame(RL_low, index=ReturnPeriod, columns=years[:-1]+modelInfo['year0'])
    df_low.index.name = 'ReturnPeriod'
    df_low.columns.name = 'Year'

    df_high = pd.DataFrame(RL_high, index=ReturnPeriod, columns=years[:-1]+modelInfo['year0'])
    df_high.index.name = 'ReturnPeriod'
    df_high.columns.name = 'Year'

    ds = xr.Dataset({'ReturnLevel': (['ReturnPeriod', 'Year'], df),
                     'RL_low': (['ReturnPeriod', 'Year'], df_low),
                     'RL_high': (['ReturnPeriod', 'Year'], df_high)})

    # set attributes
    ds.attrs['description'] = 'Time-dependent return values for the GEV model with seasonality, trend, covariate in location and scale, and nodal cycle in location. The x attribute is the binary array of parameters used in the model, where the first 3 elements are the seasonal parameters, the 4th element is the long-term trend, the 5th element is the covariate in location, the 6th element is the covariate in scale, and the 7th element is the nodal cycle.'
    ds.attrs['station_name'] = modelInfo['station_name']
    ds.attrs['datum'] = 'STND'
    ds.attrs['model_parameters'] = adjust_w_for_plotting(x,w)
    ds.attrs['record_id'] = modelInfo['recordID']
    ds.attrs['units'] = 'm'
    ds.attrs['model'] = modelName
    ds.attrs['x'] = x.tolist()
    if 'covariateName' in modelInfo:
        ds.attrs['covariateName'] = modelInfo['covariateName']
    #if mio is not singular:
    if np.linalg.det(mio) != 0:
        ds.attrs['standard_error'] = np.sqrt(np.linalg.inv(mio).diagonal()).tolist()
    else:
        ds.attrs['standard_error'] = np.full(len(w), np.nan).tolist()

    ds.attrs['covariate'] = modelInfo['covariateName']

    # make year a coordinate
    ds = ds.assign_coords(Year=years[:-1]+modelInfo['year0'])

    # make ReturnPeriod a coordinate
    ds = ds.assign_coords(ReturnPeriod=ReturnPeriod)

    ds.to_netcdf(savepath)
    print('Model saved to netcdf file at: ', savepath)
    return ds


# One dataset to rule them all

def make_directories(rsl_xr, dirs):
    """
    Ensure output and model output directories exist and create station subdirectories
    based on record_id in the given rsl_xr dataset.
    
    Args:
        rsl_xr: xarray dataset containing record_id as a coordinate.
        dirs: Dictionary containing output and model output directory paths.
    """
    # Convert directories to Path objects if they aren't already
    output_dir = Path(dirs['output_dir'])
    model_output_dir = Path(dirs['model_output_dir'])
    run_dir = Path(dirs['run_dir'])
    
    # Ensure output and model output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_run_dir = run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each station in rsl_xr.record_id
    for rid in rsl_xr.record_id:
        ridString = str(rid.values)
        station_dir = model_output_dir / ridString
        station_dir.mkdir(parents=True, exist_ok=True)

    # Cleanup: Remove specific files from the current working directory
    cleanup_files = ['best.txt', 'T.txt', 'CI.txt', 'Y.txt', 'mio.txt', 'scein.dat', 'sceout.dat']
    for file_name in cleanup_files:
        file_path = Path(run_dir / file_name)
        if file_path.exists():
            file_path.unlink()

