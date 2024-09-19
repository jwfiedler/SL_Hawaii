from helpers import adjust_w_for_plotting

from imports import *

def plotTimeDependentReturnValue_plotly(ridString, STNDtoMHHW, model_output_dir, station_name, year0plot, meanmaxSL, rangemaxSL):
    # Load the dataset
    ds = xr.open_dataset(os.path.join(model_output_dir, ridString, 'RL_muN.nc'))
    dsMHHW = ds - STNDtoMHHW
    dsMHHW.attrs['units'] = 'm, MHHW'

    # Get the first year
    year0 = dsMHHW['Year'][0].item()

    # Generate the Seaborn color palette and convert to hex
    seaborn_palette = sns.color_palette()  # Adjust the palette as needed
    cmap = [mcolors.to_hex(color) for color in seaborn_palette]

    # Create Plotly figure
    fig = go.Figure()

    yearseval = [2008, 2020]


    # Layout configuration
    fig.update_layout(
        title=f'Nonstationary GEV Return Levels: {station_name}',
        xaxis_title='Year',
        yaxis_title='Sea level (m, MHHW)',
        yaxis=dict(showline=True,linecolor='black',mirror=True,tickformat='.1f',range=[meanmaxSL - 0.5 * rangemaxSL, meanmaxSL + 0.75 * rangemaxSL]),
        xaxis=dict(showline=True,linecolor='black',mirror=True,tickformat='.0f',range=[year0plot, 2023]),
        legend_title='Return Period',
        legend=dict(
            x=1.1,  # Position the legend horizontally
            y=1,  # Position the legend at the top
        ),    
        template='plotly_white',
        width = 900,
        height = 600
    )

    # Plot each return period

    for i, rp in enumerate(dsMHHW['ReturnPeriod']):
        fig.add_trace(go.Scatter(
            x=dsMHHW['Year'].values,
            y=dsMHHW['ReturnLevel'][i].values,
            mode='lines',
            line=dict(color=cmap[i], width=2),
            name=f'{rp.values}y',
            hovertemplate='%{x:.0f}: %{y:.3f} m'
        ))
        for year in yearseval:
            yrval = dsMHHW['ReturnLevel'][i].sel(Year=year).values
            std = dsMHHW['RL_high'][i].sel(Year=year).values - yrval
            fig.add_trace(go.Scatter(
                x=[year],
                y=[yrval],
                mode='markers+text',
                marker=dict(color=cmap[i], size=10),
                hovertemplate='%{x:.0f}: %{y:.3f} m',
                textposition='top right',
                name=f'{year}: {yrval:.2f} ± {std:.3f} m'
            ))

    for i in range(2):
        rgba_color = mcolors.to_rgba(cmap[i], alpha=0.2)  # Adjust alpha for transparency

        # Convert the RGBA color to Plotly-friendly format
        fillcolor = f'rgba({rgba_color[0]*255}, {rgba_color[1]*255}, {rgba_color[2]*255}, {rgba_color[3]})'
        fillcolor0 = f'rgba({rgba_color[0]*255}, {rgba_color[1]*255}, {rgba_color[2]*255}, 0)'

        # Add fill between RL_high and RL_low using rgba for transparency
        fig.add_trace(go.Scatter(
            x=dsMHHW['Year'].values,
            y=dsMHHW['RL_high'][i].values,
            mode='lines',  # This can be 'none' if you don't want the line, but usually 'lines' is good
            line=dict(color=fillcolor0),  # Hide the line if desired
            fillcolor=fillcolor,  # Custom RGBA color with transparency
            name='Upper Confidence Interval',
            showlegend=False,
            hoverlabel=dict(
                bgcolor=fillcolor,  # Set background color for the hover box
                font_color=cmap[i]  # Set font color to match the color map
            ),
            hovertemplate='%{x:.0f}: %{y:.3f} m'
        ))

        fig.add_trace(go.Scatter(
            x=dsMHHW['Year'].values,
            y=dsMHHW['RL_low'][i].values,
            mode='lines',  # Again, this can be 'none' if you don't want a line drawn
            fill='tonexty',  # This tells Plotly to fill to the previous trace
            fillcolor=fillcolor,  # Same fillcolor for consistency
            line=dict(color=fillcolor0),  # Hide the line if desired
            showlegend=False,
            name = 'Lower Confidence Interval',
            hoverlabel=dict(
                bgcolor=fillcolor,  # Set background color for the hover box
                font_color=cmap[i]  # Set font color to match the color map
            ),
            hovertemplate='%{x:.0f}: %{y:.3f} m'
        ))





    # Assuming `mm` is defined
    fig.add_trace(go.Scatter(
        x=mm['t'].values + year0,
        y=mm['monthly_max'].values - STNDtoMHHW,
        mode='markers',
        marker=dict(color='black', size=5, symbol='cross'),
        name='Monthly maxima',
        hovertemplate='%{x:.0f}: %{y:.3f} m'
    ))

    # Save the figure
    savename = f'TimeDependentReturnValue_{ridString}.html'


    # Optional: Display the figure
    fig.show()
    #save html
    matrix_dir = Path('../../matrix/plotly')
    fig.write_html(matrix_dir / savename, full_html=True)

def plotExtremeSeasonality(T0, seaLevel, x_s,w_s, ridString, STNDtoMHHW, model_output_dir, station_name, ReturnPeriod=50, SampleRate=12, saveToFile=True):
    dx = 0.001
    t2 = np.arange(0, 1.101, dx)
    w = adjust_w_for_plotting(x_s,w_s)
    
    # Define the mu using the harmonic series
    mu = (w[0] + w[3] * np.cos(2 * np.pi * t2) + w[4] * np.sin(2 * np.pi * t2) +
          w[5] * np.cos(4 * np.pi * t2) + w[6] * np.sin(4 * np.pi * t2) +
          w[7] * np.cos(8 * np.pi * t2) + w[8] * np.sin(8 * np.pi * t2))
    psi = w[1]
    xi = w[2]
    
    # Define the time within the year
    twinthinyear =T0-np.floor(T0)
    
    T = T0
    R = ReturnPeriod
    S = SampleRate

    # get seaLevelS for all ReturnPeriods
    # preallocate seaLevelS with Nans for each ReturnPeriod
    seaLevelS = np.zeros((len(R), len(t2)))
    
    for i in range(len(R)):
        prob = 1 - (1 / (R[i]*S)) ### ALTERED FROM MATLAB CODE!!! ###
        seaLevelS[i,:] = mu - (psi/xi) * (1 - (-np.log(prob)) ** (-xi))
    
    
    serieCV = np.ones(len(T))
    
    # Find the root of the Quantilentime function
    def Quantilentime(x, R):
        t00, t11 = 0.00001, 1.00001  # Define the time bounds if not globally available

        # Variables previously defined as global in MATLAB code, handled here locally
        b0, b1, b2, b3, b4, b5, b6 = w[0], w[3], w[4], w[5], w[6], w[7], w[8]
        a0, bLT, bCI, aCI, bN1, bN2 = w[1], 0, 0, 0, 0, 0
        km = 12
        dt = 0.001
        ti = np.arange(t00, t11, dt)

        # Interpolate serieCV at ti points
        serieCV2 = np.interp(ti, T, serieCV)
        # Replace NaNs with zero (np.interp does this by default if outside the bounds)
        serieCV2[np.isnan(serieCV2)] = 0

        # Define mut (location(t)) and other parameters
        mut = (b0 * np.exp(bLT * ti) +
               b1 * np.cos(2 * np.pi * ti) + b2 * np.sin(2 * np.pi * ti) +
               b3 * np.cos(4 * np.pi * ti) + b4 * np.sin(4 * np.pi * ti) +
               b5 * np.cos(8 * np.pi * ti) + b6 * np.sin(8 * np.pi * ti) +
               bN1 * np.cos((2 * np.pi / 18.61) * ti) + bN2 * np.sin((2 * np.pi / 18.61) * ti) +
               (bCI * serieCV2))

        psi = a0 + (aCI * serieCV2)
        xit = xi

        # Calculate factor, equation 10 in Menendez and Woodworth (2009)
        factor = 0
        for i in range(len(ti)):
            h = np.maximum(1 + (xit * (x - mut[i]) / psi[i]), 0.0001) ** (-1 / xit)
            factor += h

        Pr = 1 - (1 / R)
        
        y = -Pr + np.exp(-km * factor * dt)

        return y
    
    YY = np.zeros(len(R))  # Preallocate for storing results
    
    for i, R in enumerate(ReturnPeriod):
        x0 = w[0]
        YY[i] = brentq(Quantilentime, x0-5, x0+5, args=(R,))  # using bracketing to find the root of Quantilentime function instead of fsolve
    
   # Start plotting
    fig, sss = plt.subplots(figsize=(10, 5))

    cmap = sns.color_palette()

    
    
    for i, R in enumerate(ReturnPeriod):
        sss.plot(t2, seaLevelS[i, :] - STNDtoMHHW, '-.', linewidth=2, color=cmap[i])
        sss.plot(t2, np.ones(len(t2)) * YY[i] - STNDtoMHHW, linewidth=2, color=cmap[i], label=f'R={R} years')
    sss.plot(twinthinyear, seaLevel-STNDtoMHHW, '+k', markersize=5, label='Monthly maxima',alpha=1)
    sss.grid(True,alpha=0.5)

    # arrange yaxis for breathing room
    meanmaxSL = np.nanmean(seaLevel-STNDtoMHHW)
    rangemaxSL = np.nanmax(seaLevel) - np.nanmin(seaLevel) # Range

    plt.axis([0, 1.001, meanmaxSL-rangemaxSL, meanmaxSL+rangemaxSL])
    plt.xticks(np.arange(0.08333/2, 1, 1/6), ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])
    
    # add legend
    fig.legend(loc='lower center', fontsize=11, ncol = 3, bbox_to_anchor=(0.5, -0.1))

    
    plt.ylabel('Sea level (m, MHHW)')
    plt.tight_layout()

    plt.title('Sea level extremes at ' + station_name)

    # save the figure
    if saveToFile:
        savename = 'SeasonalExtremeVariations_'+ ridString +'.png'
        savedir = os.path.join(output_dir, savename)
        plt.savefig(savedir, dpi=250, bbox_inches='tight')

    return fig, cmap


def plotTimeDependentReturnValue(ridString, STNDtoMHHW, model_output_dir, station_name, output_dir, mm, year0plot, saveToFile=True):
    # Load the dataset
    ds = xr.open_dataset(os.path.join(model_output_dir,ridString, 'RL_muN.nc'))
    dsMHHW = ds - STNDtoMHHW
    dsMHHW.attrs['units'] = 'm, MHHW'

    # Get the first year
    year0 = dsMHHW['Year'][0].item()

    # Generate the Seaborn color palette and convert to hex
    seaborn_palette = sns.color_palette()  # Adjust the palette as needed
    cmap = [mcolors.to_hex(color) for color in seaborn_palette]

    # Create the figure
    fig = plt.figure(figsize=(10,5))
    sss = fig.add_subplot(1, 1, 1)

    yearseval = [2008,2020]

    for i, rp in enumerate(dsMHHW['ReturnPeriod']):
        sss.plot(dsMHHW['Year'], dsMHHW['ReturnLevel'][i], color=cmap[i], linewidth=2, label=f'{rp.values}y')
        for year in yearseval:
            yrval = dsMHHW['ReturnLevel'][i].sel(Year = year).values
            std = dsMHHW['RL_high'][i].sel(Year = year).values - yrval
            sss.scatter(year, yrval, color=cmap[i], s=50, label=f'{year}: {yrval:.2f} ± {std:.3f}  m')

    sss.fill_between(dsMHHW['Year'], dsMHHW['RL_low'][0], dsMHHW['RL_high'][0], color=cmap[0], alpha=0.2)
    sss.fill_between(dsMHHW['Year'], dsMHHW['RL_low'][1], dsMHHW['RL_high'][1], color=cmap[1], alpha=0.2)

    # add monthly maxima
    sss.plot(mm['t']+year0, mm['monthly_max']-STNDtoMHHW, '+k', markersize=5, label='Monthly maxima',alpha=1)    

    # add title of station name
    sss.set_title('Nonstationary GEV Return Levels: ' + station_name)

    sss.set_xlabel('Year')
    sss.set_ylabel('Sea level (m, MHHW)')
    sss.grid(True,alpha=0.2)

    sss.legend(title='Return Period', loc='upper left', bbox_to_anchor=(1, 1),fontsize=10)

    # arrange yaxis for breathing room
    meanmaxSL = np.nanmean(mm['monthly_max']-STNDtoMHHW)
    rangemaxSL = np.nanmax(mm['monthly_max']) - np.nanmin(mm['monthly_max']) # Range

    sss.axis([year0plot, 2023, meanmaxSL-0.5*rangemaxSL, meanmaxSL+0.75*rangemaxSL])

    # save the figure
    if saveToFile:
        savename = 'TimeDependentReturnValue_'+ ridString +'.png'
        savedir = os.path.join(output_dir, savename)
        plt.savefig(savedir, dpi=300, bbox_inches='tight')

    return fig


