#%%
import pandas as pd
import numpy as np
import scipy.io
from datetime import datetime, timedelta
from scipy.stats import chi2
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import subprocess



#%%

#%%
# Main Functions
def stepwise(x_inisol):
    N = len(x_inisol)


    cont = 0

    x = np.array([x_inisol]) 
    # x = x_inisol # Start with initial solution
    f, pa = fitness_s(x[cont])  # Compute fitness for initial solution
    f = np.array([f])
    pa = np.array([pa])
    
    print(f)
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
                f_temp[i], pa_temp[i] = fitness_s(x_temp[i])
            print(f_temp)
            best, indi = np.max(f_temp), np.argmax(f_temp)
            # Check if the improvement is significant with chi2 test
            if (best - f[cont]) >= (0.5 * chi2.ppf(prob, df=(pa_temp[indi] - pa[cont]))):
                cont += 1
                x = np.vstack((x, [x_temp[indi]]))
                f = np.vstack((f, [best]))
                pa = np.vstack((pa, [pa_temp[indi]]))
                f[cont], pa[cont] = fitness_s(x[cont])
                print(f[cont])
            else:
                better = False
        else:
            better = False

    return x, f
def fitness_s(x):
    # Load parameter limits from a file
    aux = np.loadtxt('limits.txt')
    xmax = np.zeros(len(aux))  # Initialize with NaNs
    xmin = np.zeros(len(aux))
    # Initialize parameter limits based on the chromosome
    xmin[:3] = aux[:3, 0]
    xmax[:3] = aux[:3, 1]
    cont = 3
    
    print('x = '+ str(x))

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
    iprint = 0
    
    with open('scein.dat', 'w') as f:
        # Writing header with specified formatting
        f.write(f"{maxn:5g} {kstop:5g} {pcento:5.3g} {ngs:5g} {iseed:5g} {ideflt:5g}\n")
        f.write(f"{npg:5g} {nps:5g} {nspl:5g} {mings:5g} {iniflg:5g} {iprint:5g}\n")

        # Write chromosome values with better control over formatting
        x_formatted = " ".join(f"{int(value):1.0f}" for value in x[:-1]) + f" {int(x[-1]):1.0f}\n"
        f.write(x_formatted)

        # Writing parameter limits and initial guesses with specified precision
        for i in range(len(xini)):
            f.write(f"{xini[i]:15.8f} {xmin[i]:15.8f} {xmax[i]:15.8f}\n")        

    #print scein.dat
    with open('scein.dat', 'r') as f:
        print(f.read())

    
    # Run the external model executable
    # inputs: scein.dat, ?? what else??
    subprocess.run(["Model_GEV_SeasonalMu.exe"], check=True)
    
    # Read the output
    with open('best.txt', 'r') as file:
        bestf = float(file.readline().strip())
    
    return bestf, n

# Plotting Functions
def plottingExtremeSeasonality(Jd, T0, Y, w, mio):
    dx = 0.001
    t2 = np.arange(0, 1.101, dx)
    
    # Define the mu using the harmonic series
    mu = (w[0] + w[3] * np.cos(2 * np.pi * t2) + w[4] * np.sin(2 * np.pi * t2) +
          w[5] * np.cos(4 * np.pi * t2) + w[6] * np.sin(4 * np.pi * t2) +
          w[7] * np.cos(8 * np.pi * t2) + w[8] * np.sin(8 * np.pi * t2))
    psi = w[1]
    xi = w[2]
    
    # Define the time within the year
    dtime  = pd.to_datetime(Jd-719529, unit='D')
    Jd00 = dtime.dayofyear + dtime.hour/24 + dtime.minute/1440 + dtime.second/86400
    twithinyear = Jd00 / 367
    
    T = T0
    R = 50
    YS = mu - (psi/xi) * (1 - (-np.log(1 - (1/R))) ** (-xi))
    
    
    serieCV = np.ones(len(T))
    
    # Find the root of the Quantilentime function
    def Quantilentime(x):
        t00, t11 = 0.00001, 1.00001  # Define the time bounds if not globally available

        # Variables previously defined as global in MATLAB, handled here locally
        b0, b1, b2, b3, b4, b5, b6 = w[0], w[3], w[4], w[5], w[6], w[7], w[8]
        a0, bLT, bCI, aCI, bN1, bN2 = w[1], 0, 0, 0, 0, 0
        km = 12
        dt = 0.001
        ti = np.arange(t00, t11, dt)

        # Interpolate serieCV at ti points
        serieCV2 = np.interp(ti, T, serieCV)
        # Replace NaNs with zero (np.interp does this by default if outside the bounds)
        serieCV2[np.isnan(serieCV2)] = 0

        # Define mut and other parameters
        mut = (b0 * np.exp(bLT * ti) +
               b1 * np.cos(2 * np.pi * ti) + b2 * np.sin(2 * np.pi * ti) +
               b3 * np.cos(4 * np.pi * ti) + b4 * np.sin(4 * np.pi * ti) +
               b5 * np.cos(8 * np.pi * ti) + b6 * np.sin(8 * np.pi * ti) +
               bN1 * np.cos((2 * np.pi / 18.61) * ti) + bN2 * np.sin((2 * np.pi / 18.61) * ti) +
               (bCI * serieCV2))
        psi = a0 + (aCI * serieCV2)
        xit = xi

        # Calculate factor
        factor = 0
        for i in range(len(ti)):
            h = np.maximum(1 + (xit * (x - mut[i]) / psi[i]), 0.0001) ** (-1 / xit)
            factor += h

        Pr = 1 - (1 / R)

        y = -Pr + np.exp(-km * factor * dt)
        return y
    
    x0 = np.mean(Y)
    # t00=0.00001
    # t11=1.00001
    YY50 = fsolve(Quantilentime,x0)
    print(YY50)
    # Start plotting
    plt.figure()
    sss = plt.subplot(1, 1, 1)
    plt.plot(t2, np.ones(len(t2)) * YY50, 'k', linewidth=2)
    plt.plot(t2, YS, '--k', linewidth=2)
    plt.plot(twithinyear, Y, '+k', markersize=5)
    plt.grid(True)
    plt.axis([0, 1.001, min(Y)-(min(Y)*0.05), max(Y)+(max(Y)*0.1)])
    plt.xticks(np.arange(0.08333/2, 1, 1/6), ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.box(True)
    
    plt.legend(['R=50 years', 'Prob. R_{50} within a year', '4'])
    plt.ylabel('Sea level (m)')
    
    # Uncomment the next line to save the figure
    # plt.savefig('SeasonalExtremeVariations.png', dpi=250)

    plt.show()

#%%
# Load preprocessed monthly max data in (date, sea level (m,MHHW)) format
# read in 'MM_TWL_Kahului_Sigma.mat' file
mat_contents = scipy.io.loadmat('MM_TWL_Kahului_Sigma.mat')
#%%
MM_twl = mat_contents['MM_twl'] # monthly max sea level (in meters MHHW?)
Cvte = mat_contents['Cvte'] #covariate? (what is it?)
#%%
Jd = MM_twl[:,0]  # Julian dates
Y = MM_twl[:,1]    # Sea level data
Cvte = Cvte[:,0]  # Covariate data
#%%
# get the first year of the data
year0 = datetime.fromordinal(int(Jd[0]-366)).year # get the first year of the data
jan1_year0 = datetime(year0, 1, 1)
Jd0 = jan1_year0.toordinal() + 366 # matlab to python adjustment

t = (Jd - Jd0) / 365.25 #decimal year since Jan 1, year0
#%%
# Save processed data
pd.Series(t).to_csv('T.txt', header=False, index=False)
pd.Series(Y).to_csv('Y.txt', header=False, index=False)
pd.Series(Cvte).to_csv('CI.txt', header=False, index=False)
#%%
# Display message
print('Modeling Seasonality in Location parameter')

# Initial chromosome setup
x_0 = np.array([0, 0, 0])

# Run stepwise optimization and get the last results from the optimization
x_s, f = stepwise(x_0)
# x, f = fitness_s(x_s[-1])

# Load best results from text files
w_s = np.loadtxt('best.txt')
mio = np.loadtxt('mio.txt')  # LABEL

# Save results to a .npz (zipped archive of numpy arrays) file
np.savez('Result_Seasonality.npz', w_s=w_s, icromo=x_s[-1])

# Prepare data for plotting
w_s_plot = np.zeros(9)
w_s_plot[:3] = w_s[1:4]

# Adjust w_s_plot based on icromo conditions
icromo = x_s[-1]
if icromo[0] == 1:
    w_s_plot[3:5] = w_s[4:6]
if icromo[1] == 1 and icromo[0] == 0:
    w_s_plot[5:7] = w_s[4:6]
elif icromo[1] == 1 and icromo[0] == 1:
    w_s_plot[5:7] = w_s[6:8]
if icromo[2] == 1:
    if icromo[1] == 1 and icromo[0] == 1:
        w_s_plot[7:9] = w_s[8:10]
    elif icromo[2] == 1 and icromo[1] == 1 and icromo[0] == 0:
        w_s_plot[7:9] = w_s[6:8]
    elif icromo[2] == 1 and icromo[1] == 0 and icromo[0] == 1:
        w_s_plot[7:9] = w_s[6:8]
    elif icromo[2] == 1 and icromo[1] == 0 and icromo[0] == 0:
        w_s_plot[7:9] = w_s[4:6]
# %%
# Make a plot of monthly extremes and 50 year return period with the seasonal model
plottingExtremeSeasonality(Jd,t,Y,w_s_plot,mio)

# %%
