import numpy as np
# Reload class (e.g., to consider modifications in the code)
import importlib
import mc_lilliefors
importlib.reload(mc_lilliefors)
from mc_lilliefors import McLilliefors
import matplotlib.pyplot as plt
import seis_utils
#from statsmodels.distributions.empirical_distribution import ECDF
# if you need this module you have to install it: pip install dc_stat_think
import dc_stat_think as dcst
import pandas as pd
#=======================================================

def eli_b_value(mags, mt,interval, perc=[2.5, 97.5], n_reps=None):
    """Compute the b-value and optionally its confidence interval."""

    ## Questo script da b=beta...ovvero 1/b...ricordarsi di invertire
    # Extract magnitudes above completeness threshold: m
    m = mags[mags >= mt]

    # Compute b-value: b
    b = ((np.mean(m) - (mt-interval/2)) * np.log(10) )

    # Draw bootstrap replicates
    if n_reps is None:
        return b
    else:
        m_bs_reps = dcst.draw_bs_reps(m, np.mean, size=n_reps)

        # Compute b-value from replicates: b_bs_reps
        b_bs_reps = ((m_bs_reps - (mt-interval/2)) * np.log(10))

        # Compute confidence interval: conf_int
        conf_int = np.percentile(b_bs_reps, perc)
        std_b = (np.log(10) * (1/b)**2 * np.std(m) / np.sqrt(len(m) - 1))


        return b, conf_int,std_b

#=======================================================


def least_squares(bins, log_cum_sum):
    """ Fit a least squares curve
    """
    b,a = np.polyfit(bins, log_cum_sum, 1)
    alpha = np.log(10) * a
    beta = -1.0 * np.log(10) * b
    return a, b, alpha, beta



def distance_point_from_plane_pos_neg(x, y, z, normal, origin):
    d = -normal[0]*origin[0]-normal[1]*origin[1]-normal[2]*origin[2]
    dist = (normal[0]*x+normal[1]*y+normal[2]*z+d)
    dist = dist/np.sqrt(normal[0]**2+normal[1]**2+normal[2]**2)
    return dist

def compute_b_value(catalogue,interval):

    lill = McLilliefors(
    catalogue['Mw'],
    # signif_lev=0.1  # [default: 0.1]
    )

    lill.calc_testdistr_mcutoff(
    n_repeats=50,  # number of iterations for the random noise
    Mstart=0.1,  # lowest magnitude for which to perform the test
    # log=False,  # whether to show anythe progress bar
    )

    ##%%%fig = lill.plot_testdist_expon_mcutoff()
    ##%%%fig.layout.update(
    ##%%%    xaxis_range=(1, 6.0),
    ##%%%    )
##%%%
    ##%%%fig.show(width=fig.layout.width, height=fig.layout.height)
    ##%%%plt.show()
    ####
    Mc = lill.estimate_Mc_expon_test()
   
    print("Mc-Lilliefors: %s\n   --> number of events ≥ Mc-Lilliefors: %d" % (Mc, lill.estimates['n_compl']))
    ####
    #Mc=1.3
    Mc_all=Mc

    magnitudes=catalogue.copy()['Mw']
    #######



    # Magnitude bins
    min_mag=np.min(catalogue['Mw']) # estremi per il plot
    max_mag=np.max(catalogue['Mw'])+interval*2 #estremi per il plot
    min_mag_bin=Mc
    max_mag_bin=np.max(catalogue['Mw'])+interval*2
    print(min_mag_bin,max_mag_bin,min_mag,max_mag)

    # Magnitude bins for plotting - we will re-arrange bins later
    plot_bins_all = np.arange(min_mag, max_mag, interval)
    #bins_all = np.arange(min_mag_bin, max_mag_bin, interval)
    bins_all = plot_bins_all[plot_bins_all>min_mag_bin]
    #print(bins_all)
    #print(plot_bins_all)
    ###########################################################################
    # Generate distribution
    ###########################################################################
    # Generate histogram
    hist = np.histogram(catalogue['Mw'],bins=bins_all)
    hist_plot = np.histogram(catalogue['Mw'],bins=plot_bins_all)
    #print(hist[0])
    #print(hist_plot[0])
    # Reverse array order
    hist = hist[0][::-1]
    bins_all = bins_all[::-1]

    #plot
    hist_plot = hist_plot[0][::-1]
    plot_bins_all = plot_bins_all[::-1]

    # Calculate cumulative sum
    cum_hist = hist.cumsum()
    # Ensure bins have the same length has the cumulative histogram.
    # Remove the upper bound for the highest interval.
    bins_all = bins_all[1:]





    cum_hist_plot_all = hist_plot.cumsum()
    plot_bins_all = plot_bins_all[1:]
    wherecum=np.where(cum_hist>0)
    #log_cum_sum = np.log10(cum_hist[2::])
    log_cum_sum = np.log10(cum_hist[wherecum])


    #####
    # Compute b-value and confidence interval
    #b, conf_int = b_value(mags, mt, perc=[2.5, 97.5], n_reps=10000)
    mags=catalogue['Mw']
    mt=Mc
    
    eventi_above=np.size(np.where(catalogue['Mw']>=mt-interval/2))
    
    eli_b_value_mle, conf_int,std_b = eli_b_value(mags, mt,interval, perc=[2.5, 97.5], n_reps=100000)
    # Generate samples to for theoretical ECDF
    m_theor = np.random.exponential(eli_b_value_mle/np.log(10), size=10000) + mt
    # Report the results
    print("""
    b-value: {0:.5f}
    95% conf int: [{1:.5f}, {2:.5f}]""".format(1/eli_b_value_mle, *conf_int**(-1)))
    eli_b_value_mle=1/eli_b_value_mle

   ####################################################################################
    wherecumplot_all=np.where(cum_hist_plot_all>0)

    #####
    # Generate data to plot maximum likelihood linear curve Eli

    num_eve=len(catalogue['Mw'].loc[np.where(catalogue['Mw']>min_mag_bin)])
    mle_fit = -1.0 * eli_b_value_mle * bins_all + 1.0 * eli_b_value_mle * min_mag_bin + np.log10(num_eve)
    log_mle_fit = []
    for value in mle_fit:
        log_mle_fit.append(np.power(10,value))

    # Compare b-value of 1
    fit_data = -1.0 * bins_all + min_mag_bin + np.log10(num_eve)
    log_fit_data = []
    for value in fit_data:
        log_fit_data.append(np.power(10,value))

    print(log_cum_sum[-1],np.log10(num_eve))
    print('eli_b_value_mle=',eli_b_value_mle)


    ####

    b_all=eli_b_value_mle
    log_mle_fit_all=log_mle_fit.copy()

    num_eq=len(catalogue['Mw'])
    annual_num_eq = num_eq
    ## Get annual rate
    cum_annual_rate = cum_hist

    new_cum_annual_rate_all = []
    for i in cum_annual_rate:
        new_cum_annual_rate_all.append(i+1e-20)

        #####


    # Plotting
    # Plotting
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(plot_bins_all[wherecumplot_all],cum_hist_plot_all[wherecumplot_all],'x',color='gray')
    ax.scatter(bins_all, new_cum_annual_rate_all )
##%%
    ax.plot(bins_all, log_mle_fit, c = 'r',linewidth=2, label = 'b =%.2f'%b_all)
##%%
##%%
##%%
    #ax.plot(bins_all, log_fit_data,'--k', label = 'b=1')
    ax.plot([Mc_all ,Mc_all], [0.1, 10e4], linewidth=1,linestyle='--', color='k', label='Mc =%.2f'%Mc_all)
##%%
    #ax.plot(bins, ls_fit2, c = 'k')
    ax.set_yscale('log')
    ax.legend(fontsize=40)
    #ax.set_ylim(1e-1, max(new_cum_annual_rate_all) * 100.)
    ax.set_ylim(1e-1, 10e4)
    #ax.set_ylim(1e-1, 10e6)
    #ax.set_xlim([min_mag - 1, max_mag + 0.5])
    ax.set_xlim([0,6])
    #ax.set_ylabel('Number of events', fontsize=24)
    #ax.set_xlabel('Magnitude', fontsize=24)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.grid(color='0.95')
    plt.show()
    fig.savefig("b_value_plot.pdf", bbox_inches='tight')
##%%%
    ###

    ##%%%fig, ax = plt.subplots()
    ##%%%catalogue['Mw'].plot.hist(ax=ax, bins=50)#, bottom=1)
    ##%%%plt.show()
##%%%
    #datab=carbonates_shallow_sel.copy()
    datab=catalogue[(catalogue['Mw'] >=Mc_all)]
    datab= datab.reset_index(drop=True)


    year=pd.to_datetime(datab['time']).dt.year
    month=pd.to_datetime(datab['time']).dt.month
    day=pd.to_datetime(datab['time']).dt.day
    hour=pd.to_datetime(datab['time']).dt.hour
    minutes=pd.to_datetime(datab['time']).dt.minute
    seconds=pd.to_datetime(datab['time']).dt.second
    a_t_decYr = np.zeros( len(datab['Mw']))
    for i in range(  len(datab['Mw'])):
        a_t_decYr[i] = seis_utils.dateTime2decYr( [year[i], month[i],day[i],hour[i],minutes[i],seconds[i]])
    #fig = plt.figure(figsize=(10,10))
    fig, ax= plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    #plt.hist(a_t_decYr,density=False, bins=100,color='r')
    binwidth=1/365.
    ax[0].hist(a_t_decYr, bins=np.arange(min(a_t_decYr),max(a_t_decYr)+binwidth,binwidth),color='blue')
    ax[0].set_xlim(2016.620218579235, 2017.6191780821919) # 15 august 20216 - 15 august 2017
    #plt.hist(a_t_decYr, bins=np.arange(min(a_t_decYr),max(a_t_decYr)+binwidth,binwidth),color='r')
    seismic_rate_off=a_t_decYr
    ax[1].scatter(datab['time'],datab['Mw'],color='b',marker='o')
    ax[1].set_xlim(pd.to_datetime('2016-08-1'),pd.to_datetime('2017-08-1'))
    ax[1].set_xlabel('time',fontsize=16)
    ax[1].set_ylabel('Magnitude',fontsize=16)
    ax[0].set_ylabel('Seismicity rate',fontsize=16)
    #ax[0].set(xticklabels=[])
    ax[0].yaxis.set_tick_params(labelsize=16)
    ax[1].xaxis.set_tick_params(labelsize=16,rotation=45)
    ax[1].yaxis.set_tick_params(labelsize=16)
    fig.savefig("seismicity_rate_subset.pdf", bbox_inches='tight')
    return eli_b_value_mle,Mc,eventi_above,*conf_int**(-1),std_b
  #  return eli_b_value_mle,Mc,eventi_above


def plot_histo_selection(catalogue, Mc, colore):
    #datab=carbonates_shallow_sel.copy()
    datab=catalogue[(catalogue['Mw'] >=Mc)]
    datab= datab.reset_index(drop=True)


    year=pd.to_datetime(datab['time']).dt.year
    month=pd.to_datetime(datab['time']).dt.month
    day=pd.to_datetime(datab['time']).dt.day
    hour=pd.to_datetime(datab['time']).dt.hour
    minutes=pd.to_datetime(datab['time']).dt.minute
    seconds=pd.to_datetime(datab['time']).dt.second
    a_t_decYr = np.zeros( len(datab['Mw']))
    for i in range(  len(datab['Mw'])):
        a_t_decYr[i] = seis_utils.dateTime2decYr( [year[i], month[i],day[i],hour[i],minutes[i],seconds[i]])
    #fig = plt.figure(figsize=(10,10))

    fig, ax= plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    binwidth=1/365.

    ax.hist(a_t_decYr, bins=np.arange(min(a_t_decYr),max(a_t_decYr)+binwidth,binwidth),color=colore)

    #plt.hist(a_t_decYr, bins=np.arange(min(a_t_decYr),max(a_t_decYr)+binwidth,binwidth),color='r')

#    ax[1].scatter(datab['time'],datab['Mw'],color=colore,marker='o')
#    ax[1].set_xlabel('time',fontsize=16)
#    ax[1].set_ylabel('Magnitude',fontsize=16)
    ax.set_xlim(2016.620218579235, 2017.6191780821919) # 15 august 20216 - 15 august 2017
    #ax.set_ylabel('Seismicity rate',fontsize=16)
    #ax.set_xlabel('time',fontsize=16)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.xaxis.set_tick_params(labelsize=22,rotation=45)
#    ax[1].yaxis.set_tick_params(labelsize=16)
    fig.savefig("seismicity_rate_subset.pdf", bbox_inches='tight')
