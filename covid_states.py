
import matplotlib.pyplot as plt
import datetime
import numpy as np
import csv
import urllib.request
import io

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


data_colors = ['indigo','blue','forestgreen','red','magenta','brown','grey','orange','cyan','y']
fit_colors  =['indigo','aqua','lightgreen','pink','magneta','burlywood','light_grey','yellow','cyan','yellow']

state_list = ['California','New York','New Jersey','Texas','Illinois','Michigan','Florida','Washington','Louisiana']
num_states = len(state_list)

start_date = '2020-03-01'
end_date = '2020-05-01'
plot_path  = 'C:/Users/Mears/Dropbox/docs/covid/plots/'


date_array = np.arange(np.datetime64(start_date), np.datetime64(end_date))
num_days = date_array.shape[0]

death_array = np.zeros((num_days,num_states),dtype = np.int32)
case_array = np.zeros((num_days,num_states),dtype = np.int32)

url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
with  urllib.request.urlopen(url) as url_open:
    csv_reader = csv.reader(io.TextIOWrapper(url_open, encoding='utf-8'), delimiter=',')
    #line_count = np.zeros((num_states),dtype='int32')
    for row in csv_reader:
        if row[0] == 'date':
            continue
        for state_index,state in enumerate(state_list):
            if row[1] == state:
                dt_days = (np.datetime64(row[0]) - np.datetime64(start_date)).astype('int')
                if (dt_days >= 0) and (dt_days < num_days):
                    print(state,dt_days,int(row[4]))
                    case_array[dt_days,state_index]  = int(row[3])
                    death_array[dt_days,state_index] = int(row[4])

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, title='COVID-19 Stats from NYT', 
                     xlabel='Date', 
                     ylabel=f'Number of cases')

axs=[]
tot_case = np.sum(case_array,axis=1)
ok = tot_case > 1
#a = ax.plot(date_array[ok],tot_case[ok],color='black',label='Total Bay Area')
for state_index,state in enumerate(state_list):
    case_to_plot = case_array[:,state_index]
    ok = case_to_plot > 1
    
    a = ax.plot(date_array[ok],case_to_plot[ok],color=data_colors[state_index],label=state)
    axs.append(a)

ax.set_ylim(0,100000)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.20)
plt.legend()
last_day_of_data = np.datetime_as_string(date_array[ok][-1])
png_file = plot_path + f"states_linear_{last_day_of_data}.png"
fig.savefig(png_file)
axs=[]
fig2 = plt.figure(figsize=(12, 9))
xlim = [datetime.date(year=2020,month=3,day=1),datetime.date(year=2020,month=4,day=15)]
ax2 = fig2.add_subplot(111, title='COVID-19 Stats from NYT, Fits last 7 days', xlabel='Date', 
       ylabel=f"Number of cases",xlim=xlim)



tot_case = np.sum(case_array,axis=1)
ok = tot_case > 1
tot_to_plot = tot_case[ok]
date_to_plot = date_array[ok]
last_date = date_to_plot[-1]
date_to_fit = (date_to_plot[-7:] - np.datetime64(last_date)).astype('float64')
fit=np.polyfit(date_to_fit, np.log(tot_to_plot[-7:]), 1, w=np.sqrt(tot_to_plot[-7:]))
order_of_mag_time = np.log(2)/fit[0]

date_to_plot_for_fit = np.arange(last_date-7,last_date+10) - last_date 
yfit=np.exp(fit[1])*np.exp(fit[0]*date_to_plot_for_fit.astype('float64'))
#a = ax2.semilogy(date_array[ok],tot_case[ok],color='black',label=f'Total Bay Area, doubling time: {order_of_mag_time:.2f} days')
##afit=ax2.semilogy(np.arange(last_date-7,last_date+10),yfit,color='grey',linestyle=':',linewidth=0.5)
#axs.append(a)

for state_index,state in enumerate(state_list):
    case_to_plot = case_array[:,state_index]
    ok = case_to_plot > 1
    date_to_plot = date_array[ok]
    case_to_plot = case_to_plot[ok]
    last_date = date_to_plot[-1]
    date_to_fit = (date_to_plot[-7:] - np.datetime64(last_date)).astype('float64')
    fit=np.polyfit(date_to_fit, np.log(case_to_plot[-7:]), 1, w=np.sqrt(case_to_plot[-7:]))
    order_of_mag_time = np.log(2)/fit[0]
    a = ax2.semilogy(date_to_plot,case_to_plot,color=data_colors[state_index],label=f'{state}, doubling time: {order_of_mag_time:.2f} days')
    axs.append(a)
    date_to_plot_for_fit = np.arange(last_date-7,last_date+10) - last_date 
    yfit=np.exp(fit[1])*np.exp(fit[0]*date_to_plot_for_fit.astype('float64'))   
    afit=ax2.semilogy(np.arange(last_date-7,last_date+10),yfit,color=data_colors[state_index],linestyle=':',linewidth=0.5)
ax2.set_ylim(100,1000000)
plt.legend()
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.20)
plt.grid()

png_file = plot_path + f"states_log_{last_day_of_data}.png"
fig2.savefig(png_file)

fig3 = plt.figure(figsize=(12, 9))
xlim = [datetime.date(year=2020,month=3,day=1),datetime.date(year=2020,month=4,day=15)]
ax3 = fig3.add_subplot(111, title='COVID-19 Stats from NYT, New Cases', xlabel='Date', 
       ylabel=f"Number of new cases",xlim=xlim)
axs=[]
tot_case = np.sum(case_array,axis=1)
ok = tot_case > 1
tot_case_to_plot = tot_case[ok]
date_to_plot= date_array[ok][1:]
new_cases = tot_case_to_plot[1:] - tot_case_to_plot[0:-1]
#a = ax3.plot(date_to_plot,new_cases,color='black',label='Total Bay Area',linewidth=0.5,linestyle=':')

if new_cases.shape[0] > 7:
    new_cases = smooth(new_cases,window_len=7,window='hanning')[3:-3]
#a2 = ax3.plot(date_to_plot,new_cases,color='black',label='Total Bay Area')
#axs.append(a2)

for state_index,state in enumerate(state_list):
    case_to_plot = case_array[:,state_index]
    ok = case_to_plot > 1
    case_to_plot = case_to_plot[ok]
    date_to_plot= date_array[ok][1:]
    new_cases = case_to_plot[1:] - case_to_plot[0:-1]
    a = ax3.plot(date_to_plot,new_cases,color=data_colors[state_index],linewidth=0.5)
    print(state,np.max(new_cases))
    if new_cases.shape[0] > 7:
        new_cases = smooth(new_cases,window_len=7,window='flat')[3:-3]
    a2 = ax3.plot(date_to_plot,new_cases,color=data_colors[state_index],label=state)
    axs.append(a2)

ax3.set_ylim(0,2000)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.20)
plt.legend()
png_file = plot_path + f"states_new_cases_{last_day_of_data}.png"
fig3.savefig(png_file)

plt.show()

