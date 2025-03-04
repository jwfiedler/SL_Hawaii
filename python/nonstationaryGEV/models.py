from imports import *
from helpers import *



def prep_model_input_data(rsl_xr,recordID,dirs, CIname):
    CI_dir = dirs['CI_dir']
    run_dir = dirs['run_dir']

    mm, STNDtoMHHW, station_name, year0 = get_monthly_max_time_series(recordID,rsl_xr)
    

    mm['t'].to_csv(run_dir / 'T.txt', header=False, index=False)
    mm['monthly_max'].to_csv(run_dir / 'Y.txt', header=False, index=False)

    if CIname != 'None' and CIname != None and CIname != 'O':
        mm['CI'] = get_covariate(mm['t_monthly_max'], CI_dir, CIname, recordID)
        mm['CI'].to_csv(run_dir / 'CI.txt', header=False, index=False)


    print('Data prepared for model input in ', run_dir)
    return STNDtoMHHW, station_name, year0, mm

def assess_std_error(mio):
    if np.linalg.det(mio) != 0:
        J = np.linalg.inv(mio)
        # if all np.diag(J) are positive, then take the square root
        if np.all(np.diag(J) > 0):
            standard_error = np.sqrt(np.diag(J))
        else:
            standard_error = np.zeros_like(np.diag(J))
            standard_error[np.diag(J) > 0] = np.sqrt(np.diag(J)[np.diag(J) > 0])             
    else:
        print('Determinant of MIO is zero. Cannot compute standard error.')
        standard_error = np.zeros_like(np.diag(mio))
    return standard_error

def run_fitness(x, dirs, modelType, nproc):
    remove_files(dirs)

    fitness(x, dirs, modelType, nproc)
    w = np.loadtxt(dirs['run_dir'] / 'best.txt', dtype=float)
    mio = np.loadtxt(dirs['run_dir'] / 'mio.txt',dtype=float)
    standard_error = assess_std_error(mio)
    return w, mio, standard_error

## Seasonal model
def run_seasonal_model(ridString, dirs,runWithoutModel=False, modelType='GEV_SeasonalMu', nproc=1):
    # Initial chromosome setup
    x_0 = np.array([0,0,0])
    model_output_dir = Path(dirs['model_output_dir'])
    run_dir = Path(dirs['run_dir'])

    if os.path.exists(model_output_dir / ridString / 'seasonal_params.json') and runWithoutModel:
        with open(model_output_dir / ridString / 'seasonal_params.json', 'r') as f:
            output = json.load(f)
            w_s, x_s, mio, standard_error = (np.array(output[key]) for key in ['w', 'x', 'mio', 'standard_error'])
    else:
        x_s, f = stepwise(x_0, dirs, modelType)
        w_s, mio, standard_error = run_fitness(x_s[-1], dirs, modelType, nproc)
        print("w_s is:", np.array2string(w_s, formatter={'float_kind':lambda x: f"{x:.2E}"}))
        x_s = x_s[-1]
    
    output = {'w': w_s.tolist(), 'mio': mio.tolist(), 'standard_error': standard_error.tolist(), 'x': x_s.tolist()}
    savepath = os.path.join(model_output_dir,ridString, 'seasonal_params.json')
    with open(savepath, 'w') as f:
        json.dump(output, f)
    
    return x_s,w_s

## Long-term trend model
def run_long_term_trend_model(x_s, w_s, ridString, dirs, modelInfo, runWithoutModel=False, modelType='GEV_S_T_Cv', nproc=1):
    x_T = np.concatenate((x_s, [1, 0, 0]))  # Long-term Trend

    model_output_dir = Path(dirs['model_output_dir'])
    run_dir = Path(dirs['run_dir'])

    if os.path.exists(model_output_dir / ridString / 'trend_params.json') and runWithoutModel:
        with open(model_output_dir / ridString / 'trend_params.json', 'r') as f:
            output = json.load(f)
            w_T, x_T, mio, standard_error = (np.array(output[key]) for key in ['w', 'x', 'mio', 'standard_error'])
    else:
        w_T, mio, standard_error = run_fitness(x_T, dirs, modelType, nproc)

    aux = np.loadtxt(run_dir / 'limits.txt')
    # check to see if within limits
    wT = w_T[1:]
    for j in range(1, len(wT)):
        if wT[j] == aux[j, 0] or wT[j] == aux[j, 1]:
            raise ValueError(f'Trend Run: Parameter #{j} is at the limit: {wT[j]}')

    # check significance of adding the trend
    diffe = w_T[0] - w_s[0]
    p = 1
    SignifTrend = chi2.cdf(2 * diffe, p)

    print(f'Statistical Significance of Linear Trend: {SignifTrend*100:.2f}%')
    #TODO: check this
    print(f'Estimated Trend on monthly Maxima values is: {w_T[1]*w_T[-1]*1000:.2f} +/- {np.abs(w_T[1]*standard_error[-1]*1000):.2f} mm/year')
    # print(f'x_T is: {x_T}')
    # print(f'x_T is: {x_T}')
    # print(f'w_T is: {w_T}')

    output = {'w': w_T.tolist(), 'mio': mio.tolist(), 'standard_error': standard_error.tolist(), 'x': x_T.tolist()}
    savepath = os.path.join(model_output_dir,ridString, 'trend_params.json')
    with open(savepath, 'w') as f:
        json.dump(output, f)

    savepath = os.path.join(model_output_dir,ridString, 'RL_muT.nc')
    # save_model_to_netcdf(x,w,t,covariate,standard_error,ReturnPeriod,modelName,ridString,savepath, station_name,year0):

    if os.path.exists(savepath) and runWithoutModel==True:
        print('Model already saved to netcdf file.')
    else:
        save_model_to_netcdf(x_T,w_T,mio, modelName=modelType,savepath=savepath, modelInfo=modelInfo)
        #save_model_to_netcdf(x,w,mio,ReturnPeriod,modelName,savepath, modelInfo):
    return x_T, w_T, SignifTrend

## Nodal cycle model
def run_nodal_model(x_T, w_T, x_s, w_s, SignifTrend, ridString, dirs, modelInfo, runWithoutModel=False, modelType='GEV_S_T_Cv_Nodal', nproc=1):
    
    model_output_dir = Path(dirs['model_output_dir'])
    run_dir = Path(dirs['run_dir'])
    remove_files(dirs)

    ## Nodal cycle model
    if SignifTrend>0.95:
        x_N=np.append(x_T , [1]) #Do not include covariate in this model
        wcomp=w_T.copy()
        print('The trend is significant! \nInclude long-term trend and nodal cycle in final model.')
    else:
        x_N=np.append(x_s,[0, 0, 0, 1]) # Do not include covariate in this model
        wcomp=w_s.copy()

    # Check if the results are already saved
    if os.path.exists(model_output_dir / ridString / 'nodal_params.json') and runWithoutModel:
        with open(model_output_dir / ridString / 'nodal_params.json', 'r') as f:
            output = json.load(f)
            w_N, mio, standard_error,x_N = (np.array(output[key]) for key in ['w', 'mio', 'standard_error','x'])
    else:
        print('Running Nodal cycle model...')
        w_N, mio, standard_error = run_fitness(x_N, dirs, modelType, nproc)

    wN = w_N[1:]
    aux = np.loadtxt(run_dir / 'limits.txt')
    for j in range(1, len(wN)):
        if wN[j] == aux[j, 0] or wN[j] == aux[j, 1]:
            raise ValueError(f'Nodal Run: Parameter #{j} is at the limit: {wN[j]}')


    diffe = w_N[0] - wcomp[0]
    p = 2
    SignifN = chi2.cdf(2 * diffe, p)

    print(f'Statistical Significance of adding Nodal cycle: {SignifN*100:.2f}%')

    if SignifN<0.95:
        x_N[-1] = 0 # set the nodal component to zero
        # w_N = wcomp.copy() # use the previous model
        w_N, mio, standard_error = run_fitness(x_N, dirs, modelType, nproc)
        # w_N = np.append(w_N, [0, 0]) # add zeros for the nodal cycle
        print('Nodal cycle is not significant! \nUse previous model without nodal cycle.\n New x_N is: ', x_N)



    # save w_N, mioN, standard_error, x_N as json file
    output = {'w': w_N.tolist(), 'mio': mio.tolist(), 'standard_error': standard_error.tolist(), 'x': x_N.tolist()}
    savepath = os.path.join(model_output_dir,ridString, 'nodal_params.json')
    with open(savepath, 'w') as f:
        json.dump(output, f)

    savepath = os.path.join(model_output_dir,ridString, 'RL_muN.nc')


    if os.path.exists(savepath) and runWithoutModel:
       print('Model exists: ', savepath)
    else:
       save_model_to_netcdf(x_N,w_N, mio, modelName='Model_GEV_S_T_Cv_N.exe',savepath=savepath, modelInfo=modelInfo)

    return x_N, w_N, wcomp, SignifN

## Covariate in location model
def run_covariate_in_location_model(x_N, w_N, wcomp, ridString, SignifN, dirs, modelInfo, runWithoutModel=False, modelType='GEV_S_T_Cv',saveModel=True, nproc=1):
    
    model_output_dir = Path(dirs['model_output_dir'])
    run_dir = Path(dirs['run_dir'])
    remove_files(dirs)
    covariate_params = 'cvte_location_params_'+ modelInfo['covariateName'] + '.json'

       
    if SignifN > 0.95:
        x_cvte1 = x_N.copy()
        if len(x_cvte1) == 7:
            x_cvte1[4:7] = [1,0,1]
        else:
            x_cvte1[3:6] = [1,1,0]
        wcomp = w_N.copy()
    else:
        x_cvte1 = x_N.copy()
        if len(x_cvte1) == 7: #includes nodal cycle
            x_cvte1[4:7] = [1,0,0]
        else: #does not include nodal cycle
            x_cvte1[3:6] = [0,1,0]
        wcomp = wcomp.copy()


    if os.path.exists(model_output_dir / ridString / covariate_params) and runWithoutModel:
        with open(model_output_dir / ridString / covariate_params, 'r') as f:
            output = json.load(f)
            w_cvte1, mio, standard_error = (np.array(output[key]) for key in ['w', 'mio', 'standard_error'])
    else:
        w_cvte1, mio, standard_error = run_fitness(x_cvte1, dirs, modelType, nproc)


    diffe = w_cvte1[0] - wcomp[0]
    p = 2
    SignifCvte1 = chi2.cdf(2 * diffe, p)
    print(f"Statistical Significance of {modelInfo['covariateName']} in location param: {SignifCvte1*100:.2f}%")
    # print(f'x_cvte1 is: {x_cvte1}')

    output = {'w': w_cvte1.tolist(), 'mio': mio.tolist(), 'standard_error': standard_error.tolist(), 'x': x_cvte1.tolist()}
    
    if len(x_cvte1) == 7:
        savepath = os.path.join(model_output_dir,ridString, covariate_params)

        with open(savepath, 'w') as f:
            json.dump(output, f)

    if len(x_cvte1) == 7:
        savepath = os.path.join(model_output_dir,ridString, 'RL_muT_N_cv1.nc')
    else:
        savepath = os.path.join(model_output_dir,ridString, 'RL_muT_cv1.nc')

    if os.path.exists(savepath) and runWithoutModel==True:
        print('Model already saved to netcdf file.')
    elif saveModel and w_cvte1[0] >0:
        save_model_to_netcdf(x_cvte1,w_cvte1,mio, modelName=modelType,savepath=savepath, modelInfo=modelInfo)

    return x_cvte1, w_cvte1, wcomp, SignifCvte1

## Covariate in scale model
def run_covariate_in_scale_model(x_cvte1, w_cvte1, wcomp, ridString, SignifCvte1, dirs, modelInfo, runWithoutModel=False, modelType='GEV_S_T_Cv',saveModel=True, nproc=1):
    
    model_output_dir = Path(dirs['model_output_dir'])
    run_dir = Path(dirs['run_dir'])
    covariate_params = 'cvte_scale_params_'+ modelInfo['covariateName'] + '.json'
    remove_files(dirs)
    
    ## Covariate in scale model
    if SignifCvte1 > 0.95:
        x_cvte2 = x_cvte1.copy()
        x_cvte2[4:6] = [1,1]  
        wcomp = w_cvte1.copy()
    else:
        x_cvte2 = x_cvte1.copy()
        x_cvte2[4:6] = [0,1]
        wcomp = wcomp.copy()

    if os.path.exists(model_output_dir / ridString / covariate_params) and runWithoutModel:
        with open(model_output_dir / ridString / covariate_params, 'r') as f:
            output = json.load(f)
            w_cvte2, mio, standard_error = (np.array(output[key]) for key in ['w', 'mio', 'standard_error'])
    else:
        w_cvte2, mio, standard_error = run_fitness(x_cvte2, dirs, modelType, nproc)


    diffe = w_cvte2[0] - wcomp[0]
    p = 1
    SignifCvte2 = chi2.cdf(2 * diffe, p)




    print(f"Statistical Significance of {modelInfo['covariateName']} in scale param.: {SignifCvte2*100:.2f}%")
    # print(f'x_cvte2 is: {x_cvte2}')

    if SignifCvte2<0.95:
        x_cvte2[5] = 0 # set the covariate in scale component to zero
        print('Covariate in scale is not significant! \nUse previous model without no covariate in scale.\n New x_cvte2 is: ', x_cvte2)
        w_cvte2, mio, standard_error = run_fitness(x_cvte2, dirs, modelType, nproc)


    output = {'w': w_cvte2.tolist(), 'mio': mio.tolist(), 'standard_error': standard_error.tolist(), 'x': x_cvte2.tolist()}
    
    if len(x_cvte2) == 7:
        savepath = os.path.join(model_output_dir,ridString, covariate_params)
        with open(savepath, 'w') as f:
            json.dump(output, f)

    if len(x_cvte2) == 7:
        savepath = os.path.join(model_output_dir,ridString, 'RL_muT_N_cv2.nc')
    else:
        savepath = os.path.join(model_output_dir,ridString, 'RL_muT_cv2.nc')

    # If savepath is not None, save the model to a netcdf file

    if os.path.exists(savepath) and runWithoutModel==True:
        print('Model already saved to netcdf file.')
    elif saveModel and SignifCvte2>0.95:
        save_model_to_netcdf(x_cvte2,w_cvte2,mio,modelName='Model_GEV_S_T_Cv.exe',savepath=savepath, modelInfo=modelInfo)

    return x_cvte2, w_cvte2, wcomp, SignifCvte2

## Best model
def run_best_model(x_cvte2, w_cvte2, wcomp, SignifCvte2, ridString, dirs, modelInfo, runWithoutModel=False, modelType='GEV_S_T_Cv_Nodal', nproc=1):
## FIX THIS, NEEDS TO TAKE IN BEST CVTE INFO ###
    model_output_dir = Path(dirs['model_output_dir'])
    run_dir = Path(dirs['run_dir'])
    remove_files(dirs)

    ## Best model
    if SignifCvte2>0.95:
        x_best = x_cvte2.copy()
        wcomp=w_cvte2.copy()
    else:
        x_best = x_cvte2.copy()
        x_best[5] = 0
        wcomp=wcomp.copy()

    # Check if the results are already saved
    if os.path.exists(model_output_dir / ridString / 'best_params.json') and runWithoutModel:
        with open(model_output_dir / ridString / 'best_params.json', 'r') as f:
            output = json.load(f)
            w_N, mio, standard_error = (np.array(output[key]) for key in ['w', 'mio', 'standard_error'])    
        w_best = wcomp.copy()
    else:
        print('')
        print('*******Running best CI model...*******')
        print(f'x_best is: {x_best}')
        w_best, mio, standard_error = run_fitness(x_best, dirs, modelType, nproc)


    aux = np.loadtxt(run_dir / 'limits.txt')
    wbest = w_best[1:]
    for j in range(1, len(wbest)):
        if wbest[j] == aux[j, 0] or wbest[j] == aux[j, 1]:
            raise ValueError(f'Best Run: Parameter #{j} is at the limit: {wbest[j]}')


    diffe = w_best[0] - wcomp[0]
    p = 1
    SignifB = chi2.cdf(2 * diffe, p)

    print(f'This should be 0% if Best Model is Best: {SignifB*100:.0f}%')

    #if SignifB is not 0:
    if SignifB>0.05: # give it wiggle room (numerical precision??)
        # re-run the model
        w_best, mio, standard_error = run_fitness(x_best, dirs, modelType, nproc)
        diffe = w_best[0] - wcomp[0]
        SignifB = chi2.cdf(2 * diffe, p)
        print(f'This should be 0% if Best Model is Best: {SignifB*100:.0f}%')
        if SignifB>0.05:
            raise ValueError('SignifB is not zero. There might be an issue with the model.')
    

    # if SignifN<0.95:
    #     x_N[-1] = 0 # set the nodal component to zero
    #     w_N = wcomp.copy() # use the previous model
    #     w_N = np.append(w_N, [0, 0]) # add zeros for the nodal cycle
    #     print('Nodal cycle is not significant! \nUse previous model without nodal cycle.\n New x_N is: ', x_N)



    # save w_N, mioN, standard_error, x_N as json file
    output = {'w': w_best.tolist(), 'mio': mio.tolist(), 'standard_error': standard_error.tolist(), 'x': x_best.tolist()}
    
    # add contents of modelInfo to output
    output['modelInfo'] = modelInfo
    output['modelInfo']['covariate'] = output['modelInfo']['covariate'].tolist()
    output['modelInfo']['t'] = output['modelInfo']['t'].tolist()
    output['modelInfo']['monthlyMax'] = output['modelInfo']['monthlyMax'].tolist()
    output['modelInfo']['STNDtoMHHW'] = output['modelInfo']['STNDtoMHHW'][0]

    
    

    savepath = os.path.join(model_output_dir,ridString, 'best_params.json')
    with open(savepath, 'w') as f:
        json.dump(output, f)

    savepath = os.path.join(model_output_dir,ridString, 'RL_best.nc')


    if os.path.exists(savepath) and runWithoutModel:
        print('Model exists: ', savepath)
    else:
        save_model_to_netcdf(x_best,w_best,mio,modelName='Model_GEV_S_T_Cv_N.exe',savepath=savepath, modelInfo=modelInfo)

    return x_best, w_best, wcomp, SignifB


## Run All Models
def run_noClimateIndex_models(rsl_xr,recordID,runWithoutModel,dirs, ReturnPeriod, CIname='None', nproc=1):
    print('')
    print('******Running models without climate index...******')

    ridString = str(recordID)
    model_output_dir = dirs['model_output_dir']
    CI_dir = dirs['CI_dir']
    remove_files(dirs)

      

    STNDtoMHHW, station_name, year0, mm = prep_model_input_data(rsl_xr,recordID,dirs, CIname)

    #make empty CI.txt file
    run_dir = dirs['run_dir']
    # write zeros length of t as CI.txt
    with open(run_dir / 'CI.txt', 'w') as f:
        np.savetxt(f, np.zeros(len(mm['t'])), fmt='%d')
      



    # make dictionary of STNDtoMHHW, station_name, year0, mm,t,monthly_max,covariate
    #make dictionary of stuff that goes into xarray: t,covariate,standard_error,ReturnPeriod,modelName='Model_GEV_S_T_Cv_N.exe',ridString=ridString,savepath=savepath, station_name=station_name,year0=year0)
    modelInfo = {'t': mm['t'], 
                 'monthlyMax': mm['monthly_max'],
                 'covariate': None, 
                 'covariateName': CIname, 
                 'recordID': recordID, 
                 'station_name': station_name, 
                 'year0': year0, 
                 'ReturnPeriod': ReturnPeriod,
                 'STNDtoMHHW': STNDtoMHHW,
                 'datum':'STND'}

    # Run the basic models for seasonal, trend and nodal cycle
    x_s, w_s = run_seasonal_model(ridString, dirs,runWithoutModel=runWithoutModel, modelType='GEV_SeasonalMu', nproc=nproc)
    x_T, w_T, SignifTrend = run_long_term_trend_model(x_s, w_s, ridString, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv', nproc=nproc)
    x_N, w_N, wcomp, SignifN = run_nodal_model(x_T, w_T, x_s, w_s, SignifTrend, ridString, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv_Nodal', nproc=nproc)

    return STNDtoMHHW, station_name, year0, mm, x_s, w_s, x_N, w_N, wcomp, SignifN

def run_CI_models(rsl_xr,recordID,runWithoutModel,dirs, ReturnPeriod, CI_list,x_N, w_N, wcomp, SignifN, nproc=1):
    print('')
    print('****Running models with climate index...****')
    ridString = str(recordID)
    model_output_dir = dirs['model_output_dir']
    CI_dir = dirs['CI_dir']
    remove_files(dirs)

    SignifCvte1 = np.zeros(len(CI_list))
    magCvte1 = np.zeros(len(CI_list))


    # Run the models with covariates, looping through the covariates
    for i,CIname in enumerate(CI_list):
        STNDtoMHHW, station_name, year0, mm = prep_model_input_data(rsl_xr,recordID,dirs, CIname)
        modelInfo = {'t': mm['t'], 
                 'monthlyMax': mm['monthly_max'],
                 'covariate': None, 
                 'covariateName': CIname, 
                 'recordID': recordID, 
                 'station_name': station_name, 
                 'year0': year0, 
                 'ReturnPeriod': ReturnPeriod,
                 'STNDtoMHHW': STNDtoMHHW,
                 'datum':'STND'}
        
        
        
        wcomptemp = wcomp.copy()
        x_cvte1, w_cvte1, wcomptemp, SignifCvte1[i] = run_covariate_in_location_model(x_N, w_N, wcomptemp, ridString, SignifN, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv_Nodal',saveModel=False, nproc=nproc)
        magCvte1[i] = w_cvte1[-1]
    
    # Find the model with the best fit in location
    best_model = np.nanargmax(SignifCvte1)
    print('Best model is ' + CI_list[best_model])
    STNDtoMHHW, station_name, year0, mm = prep_model_input_data(rsl_xr,recordID,dirs, CI_list[best_model])
    modelInfo = {'t': mm['t'], 
                 'covariate': mm['CI'], 
                 'monthlyMax': mm['monthly_max'], 
                 'covariateName': CI_list[best_model], 
                 'recordID': recordID, 
                 'station_name': station_name, 
                 'year0': year0, 
                 'ReturnPeriod': ReturnPeriod,
                 'STNDtoMHHW': STNDtoMHHW,
                 'datum':'STND'}
    
    wcomp_noN = wcomp.copy()
    
    x_cvte1, w_cvte1, wcomp, SignifCvte1_best = run_covariate_in_location_model(x_N, w_N, wcomp, ridString, SignifN, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv_Nodal',saveModel=True, nproc=nproc)  
    x_cvte2, w_cvte2, wcomp, SignifCvte2 = run_covariate_in_scale_model(x_cvte1, w_cvte1, wcomp, ridString,SignifCvte1_best, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv_Nodal',saveModel=True, nproc=nproc)
    run_best_model(x_cvte2, w_cvte2, wcomp, SignifCvte2, ridString, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv_Nodal', nproc=nproc)

    # now run CI models without nodal component
    print('Running models with climate index without nodal component...')
    x_s, w_s = run_seasonal_model(ridString, dirs,runWithoutModel=True, modelType='GEV_SeasonalMu',nproc=nproc)
    x_T, w_T, SignifTrend = run_long_term_trend_model(x_s, w_s, ridString, dirs, modelInfo, runWithoutModel=True, modelType='GEV_S_T_Cv',nproc=nproc)
    x_cvte1, w_cvte1, wcomp, Sig = run_covariate_in_location_model(x_T, w_T, w_T, ridString, SignifTrend, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv',saveModel=True,nproc=nproc)
    run_covariate_in_scale_model(x_cvte1, w_cvte1, wcomp, ridString,Sig, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv',saveModel=True,nproc=nproc)

    print('All models run successfully! Results save to ', model_output_dir)
    return STNDtoMHHW, station_name, year0, mm, magCvte1, SignifCvte1


## Run All Models
def run_all_models(rsl_xr,recordID,runWithoutModel,dirs, ReturnPeriod, CIname='PMM', nproc=1):
    ridString = str(recordID)
    model_output_dir = dirs['model_output_dir']
    CI_dir = dirs['CI_dir']
    remove_files(dirs)
    STNDtoMHHW, station_name, year0, mm = prep_model_input_data(rsl_xr,recordID,dirs, CIname)

    # make dictionary of STNDtoMHHW, station_name, year0, mm,t,monthly_max,covariate
    #make dictionary of stuff that goes into xarray: t,covariate,standard_error,ReturnPeriod,modelName='Model_GEV_S_T_Cv_N.exe',ridString=ridString,savepath=savepath, station_name=station_name,year0=year0)
    modelInfo = {'t': mm['t'], 'covariate': mm['CI'], 'monthlyMax': mm['monthly_max'],'covariateName': CIname, 'recordID': recordID, 'station_name': station_name, 'year0': year0, 'ReturnPeriod': ReturnPeriod}

    # Run the basic models for seasonal, trend and nodal cycle
    x_s, w_s = run_seasonal_model(ridString, dirs,runWithoutModel=True, modelType='GEV_SeasonalMu')
    x_T, w_T, SignifTrend = run_long_term_trend_model(x_s, w_s, ridString, dirs, modelInfo, runWithoutModel=True, modelType='GEV_S_T_Cv', nproc=nproc)
    x_N, w_N, wcomp, SignifN = run_nodal_model(x_T, w_T, x_s, w_s, SignifTrend, ridString, dirs, modelInfo, runWithoutModel=True, modelType='GEV_S_T_Cv_Nodal', nproc=nproc)
    x_cvte1, w_cvte1, wcomp, SignifCvte1 = run_covariate_in_location_model(x_N, w_N, wcomp, ridString, SignifN, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv_Nodal', nproc=nproc)
    x_cvte2, w_cvte2, wcomp, SignifCvte2 = run_covariate_in_scale_model(x_cvte1, w_cvte1, wcomp, ridString,SignifCvte1, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv_Nodal', nproc=nproc)
    run_best_model(x_cvte2, w_cvte2, w_s, wcomp, SignifCvte2, ridString, dirs, modelInfo, runWithoutModel=runWithoutModel, modelType='GEV_S_T_Cv_Nodal', nproc=nproc)

    print('All models run successfully! Results save to ', model_output_dir)
    return STNDtoMHHW, station_name, year0, mm, x_s, w_s, w_cvte1, w_cvte2, w_T