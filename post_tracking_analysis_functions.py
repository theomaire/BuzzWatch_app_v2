

from pyexpat.errors import XML_ERROR_FEATURE_REQUIRES_XML_DTD
import sys
sys.path.append('/Volumes/BBB/Theo_projects/BuzzWatch/buzzwatch_analysis_module/') # ADD PATH OF THE BUZZWATCH PYTHON MODULE
from buzzwatch_data_analysis.experiment_analysis import*
from buzzwatch_data_analysis.misc_functions import *
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.dates import DateFormatter
from scipy import stats
import matplotlib.colors as mcl


###### FUNCTIONS ######
def crop_time(t_i,t_f,df):
    mask = (df.index > t_i) & (df.index < t_f)
    df = df.loc[mask]
    return df.groupby(df.index).mean()

def get_mean_time_int(df,t_i,t_f):
    x = df.between_time(t_i,t_f)
    return x.mean()

def get_std_time_int(df,t_i,t_f):
    x = df.between_time(t_i,t_f)
    return x.std()


def compute_death_rate(df):
    x = np.arange(df.index.size)
    y = np.array(df.values).reshape(-1)
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    #print(r*r)

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, x))

    #ax.plot_date(df.index, mymodel,"--",linewidth=3)
    return slope*(24*60*7) # In number of death week


def run_lin_regression(ax,x,y):
    x = np.array(x).reshape(-1) #np.arange(df.index.size)
    y = np.array(y).reshape(-1)
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    #print(r*r)

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, x))

    ax.plot(x, mymodel,"-",linewidth=2,color="black",label="R^2 ="+ str(np.round(r*r,3)))
    return r*r # In number of death week

def compute_proportion_mosquito_flying(start_date,end_date,number_alive,number_flying,specific_list_of_days):

    number_alive = crop_time(start_date-timedelta(days=1),end_date,number_alive)
    number_flying = crop_time(start_date-timedelta(days=1),end_date,number_flying)
    number_flying = number_flying.interpolate()
    number_alive = number_alive.interpolate()

    #print(number_flying)
    #print(number_alive)



    norm_flying =  number_flying/number_alive


    nb_days = (end_date-start_date).days

    #print(nb_days)
    if len(specific_list_of_days)>0:
        days = [np.int32(i) for i in specific_list_of_days]
    else:
        days = np.arange(nb_days)
        
    for day in days:
        #print(day)

        t_i = start_date+timedelta(days=int(day))
        t_f = start_date+timedelta(days=int(day+1))

        number_alive_day = crop_time(t_i,t_f,number_alive)
        number_flying_day = crop_time(t_i,t_f,number_flying)
        #print(number_alive_day)
        #print(number_flying_day)
        

        if day == 0:
            df_all = pd.DataFrame(index=number_flying_day.index)
            df_all[str(day)] = number_flying_day.values/number_alive_day.values
        else:
 
            df_all[str(day)] = number_flying_day.values/number_alive_day.values

    return df_all,number_flying,number_alive

def week_average(start_date,end_date,df,specific_list_of_days):

    #number_alive = crop_time(start_date-timedelta(days=1),end_date,number_alive)
    df = crop_time(start_date-timedelta(days=1),end_date,df)
    #df = df.interpolate()

    #norm_flying =  number_flying/number_alive


    nb_days = (end_date-start_date).days
    #print(nb_days)

    #print(nb_days)
    if len(specific_list_of_days)>0:
        days = [np.int32(i) for i in specific_list_of_days]
    else:
        days = np.arange(nb_days)
        
    for day in days:
        #print(day)

        t_i = start_date+timedelta(days=int(day))
        t_f = start_date+timedelta(days=int(day+1))

        #number_alive_day = crop_time(t_i,t_f,number_alive)
        number_flying_day = crop_time(t_i,t_f,df)

        #print(number_flying_day)
    #number_flying_day = number_flying_day.resample('1T', label='right').mean()
        #print(number_alive_day)
        #print(number_flying_day)
        

        if day == 0:
            df_all = pd.DataFrame(index=number_flying_day.index)
            df_all[str(day)] = number_flying_day.values
        else:
            df_all[str(day)] = number_flying_day.values

    return df_all

def week_average_save_time(start_date,end_date,df,specific_list_of_days):

    #number_alive = crop_time(start_date-timedelta(days=1),end_date,number_alive)
    df = crop_time(start_date-timedelta(days=1),end_date,df)
    #df = df.interpolate()

    #norm_flying =  number_flying/number_alive


    nb_days = (end_date-start_date).days
    #print(nb_days)

    #print(nb_days)
    if len(specific_list_of_days)>0:
        days = [np.int32(i) for i in specific_list_of_days]
    else:
        days = np.arange(nb_days)
        
    for day in days:
        #print(day)

        if day>10:
            t_i = start_date+timedelta(days=int(day))
            t_f = start_date+timedelta(days=int(day+1))
        else:
            t_i = start_date+timedelta(days=int(day))-timedelta(hours=1)
            t_f = start_date+timedelta(days=int(day+1))-timedelta(hours=1)

        #number_alive_day = crop_time(t_i,t_f,number_alive)
        number_flying_day = crop_time(t_i,t_f,df)

        #print(number_flying_day)
    #number_flying_day = number_flying_day.resample('1T', label='right').mean()
        #print(number_alive_day)
        #print(number_flying_day)
        

        if day == 0:
            df_all = pd.DataFrame(index=number_flying_day.index)
            df_all[str(day)] = number_flying_day.values
        else:
            df_all[str(day)] = number_flying_day.values

    return df_all

# 
def contract_RAB_CAY_KED_COL(df):
    def modif_time_stamp(df,start,end,period):
        if period =="day":
    # Define the original start and end times
            original_start_time = start
            original_end_time = end

            # Define the desired start and end times
            desired_start_time = pd.Timestamp('07:00:00')
            desired_end_time = pd.Timestamp('22:00:00')

            original_duration = original_end_time - original_start_time
            desired_duration = desired_end_time - desired_start_time

            # Calculate the contraction factor
            contraction_factor = desired_duration / original_duration

            # Update the time index using the contraction factor
            df['time_numeric'] = (df.index - original_start_time).total_seconds()
            df['time_numeric'] = df['time_numeric'] * contraction_factor
            df['time_numeric'] = df['time_numeric'] + (pd.Timestamp('07:00:00') - pd.Timestamp('00:00:00')).total_seconds()
            df['date'] = df.index.date

            # Combine the updated time with the original date
            df.index = pd.to_datetime(df['date'].astype(str) + ' ' + pd.to_datetime(df['time_numeric'], unit='s').dt.time.astype(str))
            df.drop(columns=['time_numeric', 'date'], inplace=True)
        else:
            original_start_time = pd.Timestamp('23:00:00')
            original_end_time = pd.Timestamp('05:00:00')

            # Define the desired start and end times
            desired_start_time = pd.Timestamp('22:00:00')
            desired_end_time = pd.Timestamp('07:00:00')

            # Compute the time difference between the original and desired time ranges
            original_duration = (original_end_time + pd.DateOffset(days=1) - original_start_time).total_seconds()
            desired_duration = (desired_end_time + pd.DateOffset(days=1) - desired_start_time).total_seconds()

            # Calculate the expansion factor
            
            expansion_factor = desired_duration / original_duration

            # Update the time index using the expansion factor
            df['time_numeric'] = ((df.index - original_start_time).total_seconds() % original_duration) * expansion_factor
            df['time_numeric'] = df['time_numeric'] + (desired_start_time - original_start_time).total_seconds()

            # Extract the date from original timestamps
            df['date'] = df.index.date

            # Combine the updated time with the original date
            df.index = pd.to_datetime(df['date'].astype(str) + ' ' + pd.to_datetime(df['time_numeric'], unit='s').dt.time.astype(str))
            df.drop(columns=['time_numeric', 'date'], inplace=True)
            
        return df
    
    day_start = 1
    day_end = 13
    start_date = datetime(2023,5,int(day_start),5,0,0)
    end_date = datetime(2023,5,int(day_end),5,0,0)

    for day in np.arange(day_end-day_start):

        # Day contraction
        start_date = datetime(2023,5,int(day_start+day),5,0)
        end_date = datetime(2023,5,int(day_start+day),23,0)
        mask = (df.index >start_date) & (df.index <= end_date)
        df2 = df.loc[mask]
        #df2s.copy(deep=False)
        df3 = df2.copy()
        #df3 = df2.copy()
        df4_1 = modif_time_stamp(df3,start_date,end_date,"day")

        # Night inflation
        start_date = datetime(2023,5,int(day_start+day),23,0)
        end_date = datetime(2023,5,int(day_start+day+1),5,0)
        mask = (df.index >start_date) & (df.index <= end_date)
        df2 = df.loc[mask]
        #df2s.copy(deep=False)
        df3 = df2.copy()
        #df3 = df2.copy()
        df4_2 = modif_time_stamp(df3,start_date,end_date,"night")
        #df4_2.drop(df4_2.tail(1).index,inplace = True)
        start_date = datetime(2023,5,int(day_start+day),22,0)
        end_date = datetime(2023,5,int(day_start+day+1),7,0)
        mask = (df4_2.index >start_date) & (df4_2.index <= end_date)
        df4_2 = df4_2.loc[mask]

        if day==0:
            df_modif = pd.concat([df4_1,df4_2])
        else:
            df_modif = pd.concat([df_modif,df4_1,df4_2])


    return df_modif



def contract_KUM(df):
    def modif_time_stamp(df,start,end,period):
        if period =="day":
    # Define the original start and end times
            original_start_time = start
            original_end_time = end

            # Define the desired start and end times
            desired_start_time = pd.Timestamp('06:00:00')
            desired_end_time = pd.Timestamp('23:00:00')

            # # Compute the time difference be<tween the original and desired time ranges
            # time_difference = desired_end_time - desired_start_time

            # # Calculate the time index offset
            # time_offset = desired_start_time - original_start_time
            # Compute the time difference between the original and desired time ranges
        # Compute the time difference between the original and desired time ranges
            original_duration = original_end_time - original_start_time
            desired_duration = desired_end_time - desired_start_time

            # Calculate the contraction factor
            contraction_factor = desired_duration / original_duration

            # Update the time index using the contraction factor
            df['time_numeric'] = (df.index - original_start_time).total_seconds()
            #print(df['time_numeric'])
            df['time_numeric'] = df['time_numeric'] * contraction_factor
            #print(df['time_numeric'])
            df['time_numeric'] = df['time_numeric'] + (pd.Timestamp('06:00:00') - pd.Timestamp('00:00:00')).total_seconds()
            #print(df['time_numeric'])
            # df.index = pd.to_datetime(df['time_numeric'], unit='s')
            # df.drop(columns='time_numeric', inplace=True)
            # Extract the date from original timestamps
            df['date'] = df.index.date

            # Combine the updated time with the original date
            df.index = pd.to_datetime(df['date'].astype(str) + ' ' + pd.to_datetime(df['time_numeric'], unit='s').dt.time.astype(str))
            
            df.drop(columns=['time_numeric', 'date'], inplace=True)
        else:
            original_start_time = start
            original_end_time = end

            # Define the desired start and end times
            desired_start_time = pd.Timestamp('23:00:00')
            desired_end_time = pd.Timestamp('06:00:00')

            # Compute the time difference between the original and desired time ranges
            original_duration = (original_end_time - original_start_time).total_seconds()
            desired_duration = (desired_end_time + pd.DateOffset(days=1) - desired_start_time).total_seconds()

            # Calculate the expansion factor
            
            expansion_factor = desired_duration / original_duration
            #print(expansion_factor)

            # Update the time index using the expansion factor
            df['time_numeric'] = ((df.index - original_start_time).total_seconds()) * expansion_factor
            #print(df['time_numeric'])
            df['time_numeric'] = df['time_numeric'] + (pd.Timestamp('23:00:00') - pd.Timestamp('00:00:00')).total_seconds()
            #print(df['time_numeric'])
            #print(df['time_numeric'])

            # Extract the date from original timestamps
            df['date'] = df.index.date
            #print(df['date'])

            # Combine the updated time with the original date
            df.index = pd.to_datetime(df['date'].astype(str) + ' ' + pd.to_datetime(df['time_numeric'], unit='s').dt.time.astype(str))
            #df.index = df.index - pd.DateOffset(hours=1)
            df.drop(columns=['time_numeric', 'date'], inplace=True)
        
        return df

    # Contract the vector
    day_start = 1
    day_end = 20

    #start_date = datetime(2023,3,2,6,0,0)
    #end_date = datetime(2023,3,1,6,0,0)

    start_date = datetime(2023,3,int(day_start),5,0,0)
    end_date = datetime(2023,3,int(day_end),5,0,0)

    for day in np.arange(day_end-day_start):

        # Day contraction
        start_date = datetime(2023,3,int(day_start+day),5,0)
        end_date = datetime(2023,3,int(day_start+day),23,0)
        df2 = crop_time(start_date,end_date,df)
        #df2s.copy(deep=False)
        df3 = df2.copy()
        #df3 = df2.copy()
        df4_1 = modif_time_stamp(df3,start_date,end_date,"day")

        # Night inflation
        start_date = datetime(2023,3,int(day_start+day),23,0)
        end_date = datetime(2023,3,int(day_start+day+1),5,0)
        df2 = crop_time(start_date,end_date,df)
        #df2s.copy(deep=False)
        df3 = df2.copy()
        #df3 = df2.copy()
        df4_2 = modif_time_stamp(df3,start_date,end_date,"night")
        #df4_2.drop(df4_2.tail(1).index,inplace = True)
        start_date = datetime(2023,3,int(day_start+day),23,0)
        end_date = datetime(2023,3,int(day_start+day+1),6,0)

        df4_2 = crop_time(start_date,end_date,df4_2)

        if day==0:
            df_modif = pd.concat([df4_1,df4_2])
        else:
            df_modif = pd.concat([df_modif,df4_1,df4_2])
    df_modif = df_modif.shift(-30, freq='T') 

    return df_modif

########### Transform columns days into single time series

def linearize_days(df):


    for i,day in enumerate(df.keys()):
        if i==0:
            df_linear = df[day]
            
        else:
            df[day].index = df[day].index+timedelta(days=i)
            df_linear = pd.concat([df_linear,df[day]])

    return df_linear


#################### Single trajectories analysis ###################





# set_plot_size(20)
# mpl.rcParams['axes.linewidth'] = 1
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['font.family'] = 'Arial'


# ######################### USER INPUT PARAMETERS ###################
# folder_plots = "/Volumes/BBB/Theo_projects/BuzzWatch/ANALYZED/global_aaa_aaf_comparison/"
# folder_processed_data = "/Volumes/BBB/Theo_projects/BuzzWatch/ANALYZED/global_aaa_aaf_comparison/data_all_processed/"

# files_mortality = [f for f in listdir(folder_processed_data ) if isfile(join(folder_processed_data , f)) and f.endswith("number_alive.pkl")]
# files_mortality.sort()

# files_flying = [f for f in listdir(folder_processed_data ) if isfile(join(folder_processed_data , f)) and f.endswith("number_flying.pkl")]
# files_flying.sort()

# files_fraction = [f for f in listdir(folder_processed_data ) if isfile(join(folder_processed_data , f)) and f.endswith("fraction_day_av.pkl")]
# files_fraction.sort()

# #######Initializ data structure ############
# all_data=[]
# listKeys = ["Population_alias",
#             "Lab_gen",
#             "Latitude",
#             "Subspecies",
#             "Generation",
#             "Batch_exp",
#             "Number_alive_mosquitos",
#             "Number_flying_mosquitos",
#             "Fraction_flying_day",
#             "Death_rate"
#             ]

# for i in np.arange(10):
#     all_data.append(dict(zip(listKeys, [None]*len(listKeys))))

# # Load excel file
# df_meta = pd.read_excel("/Volumes/BBB/Theo_projects/BuzzWatch/ANALYZED/global_aaa_aaf_comparison/metadata_all_strains.xlsx")

# colors_pop = ['tab:blue',
#              'tab:orange',
#              'tab:green',
#              'tab:red',
#              'tab:purple',
#              'tab:brown',
#              'tab:pink',
#              'tab:gray',
#              'tab:olive',
#              'tab:cyan']

# for k in np.arange(10):
    

#     with open(folder_processed_data+files_flying[k], 'rb') as f:
#         df= pickle.load(f)
#     all_data[k]["Number_flying_mosquitos"] = df

#     with open(folder_processed_data+files_mortality[k], 'rb') as f:
#         df= pickle.load(f)
#     all_data[k]["Number_alive_mosquitos"] = df

#     with open(folder_processed_data+files_fraction[k], 'rb') as f:
#         df= pickle.load(f)
#     all_data[k]["Fraction_flying_day"] = df

#     all_data[k]["Population_alias"] = df_meta["Population"][k]
#     all_data[k]["Latitude"] = np.float(df_meta["Latitude"][k])
#     all_data[k]["Subspecies"] = df_meta["Subspecies"][k]
#     all_data[k]["Generation"] = np.int32(df_meta["Generation"][k])
#     all_data[k]["Batch_exp"] = df_meta["Batch_exp"][k]

# #####################################


# ########### Plot raw flight data for all
# fig, axes = plt.subplots(5, 2,dpi=200)
# fig.set_figheight(15)
# fig.set_figwidth(20)

# for i,ax in enumerate(axes.reshape(-1)):
#     df = all_data[i]["Number_flying_mosquitos"]



#     t_i= df.index.min()
#     t_f  = t_i+timedelta(days=10)
#     df = crop_time(t_i,t_f,df)
#     #print(df)
#     df = df.interpolate()
#     #df.index = df.index-t_i
#     ax.plot_date(df.index,df.values,label=all_data[i]["Population_alias"],color = colors_pop[i],linestyle="-",marker=None,linewidth=2)

#     ax.legend()

#     hh_mm = mdates.DateFormatter('%m-%d')
#     ax.xaxis.set_major_formatter(hh_mm)
# #ax.legend(loc='best')
#     if i == 0:
#         ax.set_ylabel("Number mosquitos flying")
#         ax.set_xlabel("date")
#     ax.set_ylim([0,5])
# plt.tight_layout()
# plt.savefig(folder_plots+"_number_flying_raw.png",bbox_inches='tight')



# ########### Activity as function of Latitude ###########
# fig, axes = plt.subplots(3,2,dpi=100)
# fig.set_figheight(20)
# fig.set_figwidth(20)

# time_int = [["7:00","10:00"],["10:00","14:00"],["14:00","20:00"],["20:00","22:00"],["00:00","6:00"],["6:00","23:59"]]

# for i,ax in enumerate(axes.reshape(-1)):
#     for k in np.arange(10):

#         df = all_data[k]["Fraction_flying_day"]
#         activity = get_mean_time_int(df.mean(axis=1),time_int[i][0],time_int[i][1])
#         if all_data[k]["Subspecies"] == "formosus":
#             ax.scatter(all_data[k]["Latitude"],activity,s=300,label =all_data[k]["Population_alias"] )
#             #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#         else:
#             ax.scatter(all_data[k]["Latitude"],activity,s=300,marker="v" ,label =all_data[k]["Population_alias"] )
#             #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#     ax.set_title(time_int[i][0]+"_"+time_int[i][1])
#     ax.set_ylim([0,0.06])
#     if i==0:
#         ax.legend(fontsize="20",ncol=2)
#     ax.set_ylabel('Average flight activity')
#     ax.set_xlabel('Latitude')
# plt.tight_layout()
# plt.savefig(folder_plots+"_latitude.png",bbox_inches='tight')
# #plt.show()


# ############ Activity as function aegypti or formosus ###########
# fig, axes = plt.subplots(2,3,dpi=200)
# fig.set_figheight(20)
# fig.set_figwidth(20)
# plt.setp(axes, xticks=[0.5,0.6], xticklabels=["aegypti","formosus"])
# time_int = [["7:00","10:00"],["10:00","14:00"],["14:00","20:00"],["20:00","22:00"],["00:00","6:00"],["6:00","23:59"]]

# for i,ax in enumerate(axes.reshape(-1)):
#     x_formosus = []
#     x_aegypti = []
#     for k in np.arange(10):
#         df = all_data[k]["Fraction_flying_day"]
#         activity = get_mean_time_int(df.mean(axis=1),time_int[i][0],time_int[i][1])
        
#         if all_data[k]["Subspecies"] == "formosus":
#             x_formosus.append(activity)
#             #ax.scatter(all_data[k]["Latitude"],activity,s=100,label = all_data[k]["Population_alias"])
#         else:
#             x_aegypti.append(activity)

#     data = [np.mean(x_aegypti),np.mean(x_formosus)]
#     std_error=[np.std(x_aegypti),np.std(x_formosus)]

#     # Aegypti 
#     ax.errorbar([0.5],np.mean(x_aegypti),np.std(x_aegypti),marker='o', mfc='none',capsize=30,markersize=20,color='tab:orange',facecolor=None,label="A.a aegypti")
#     ax.scatter([0.5 for j in np.arange(5)],x_aegypti,color='tab:orange',s=100)

#     # formosus
#     ax.errorbar([0.6],np.mean(x_formosus),np.std(x_formosus),marker='o', mfc='none',capsize=30,markersize=20,color='tab:blue',facecolor=None,label="A.a formosus")
#     ax.scatter([0.6 for j in np.arange(5)],x_formosus,color='tab:blue',s=100)

#             #ax.scatter(all_data[k]["Latitude"],activity,s=100,marker="v",label = all_data[k]["Population_alias"])
#     ax.set_title(time_int[i][0]+"_"+time_int[i][1])
#     #ax.set_ylim([0,45])
#     if i==0:
#         ax.legend(fontsize="40")
#     ax.set_ylim([0,0.06])
#     ax.set_xlim([0.45,0.65])
#     ax.set_ylabel("Average flight activity")
# plt.tight_layout()
# plt.savefig(folder_plots+"_grouped.png",bbox_inches='tight')
# #plt.show()




# ############ Death rate for all ###########
# fig, axes = plt.subplots(5,2,dpi=100)
# fig.set_figheight(20)
# fig.set_figwidth(20)

# for i,ax in enumerate(axes.reshape(-1)):
#     df = all_data[i]["Number_alive_mosquitos"]
#     death_rate = compute_death_rate(df,ax)
#     ax.plot_date(df.index,df.values,"-",linewidth=3,label=str(np.round(np.max([-death_rate,0]),2))+" mosquito death / week")
#     all_data[i]["Death_rate"] = death_rate
#     date_form = DateFormatter("%m-%d")
#     ax.xaxis.set_major_formatter(date_form)
#     ax.set_ylim([0, 45])
#     ax.set_title(all_data[i]["Population_alias"])
#     ax.legend()
# plt.tight_layout()
# plt.savefig(folder_plots+"_mortality_curves.png",bbox_inches='tight')


# ############ Correlation death rate and activity ###########
# fig, axes = plt.subplots(3,2,dpi=100)
# fig.set_figheight(20)
# fig.set_figwidth(20)

# time_int = [["7:00","10:00"],["10:00","14:00"],["14:00","20:00"],["20:00","22:00"],["00:00","6:00"],["6:00","23:59"]]

# for i,ax in enumerate(axes.reshape(-1)):
#     for k in np.arange(10):

#         df = all_data[k]["Fraction_flying_day"]
#         activity = get_mean_time_int(df.mean(axis=1),time_int[i][0],time_int[i][1])
#         if all_data[k]["Subspecies"] == "formosus":
#             ax.scatter(-all_data[k]["Death_rate"],activity,s=300,label =all_data[k]["Population_alias"] )
#             #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#         else:
#             ax.scatter(-all_data[k]["Death_rate"],activity,s=300,marker="v" ,label =all_data[k]["Population_alias"] )
#             #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#     ax.set_title(time_int[i][0]+"_"+time_int[i][1])
#     ax.set_ylim([0,0.06])
#     if i==0:
#         ax.legend(fontsize="20",ncol=2)
#     ax.set_ylabel('Average flight activity')
#     ax.set_xlabel('Death rate (nb mosquito death/week)')
# plt.tight_layout()
# plt.savefig(folder_plots+"_corr_mortality.png",bbox_inches='tight')


# ############ Week average plots ###########
# fig, axes = plt.subplots(5, 2,dpi=200)
# fig.set_figheight(10)
# fig.set_figwidth(10)

# df_aaf =[]
# df_aaa =[]
# for i,ax in enumerate(axes.reshape(-1)):
#     df = all_data[i]["Fraction_flying_day"]


#     if  i==0:
#         earliest_start_date = df.index.min()
#     else:
#         # Calculate the time difference for each DataFrame
#         df1_time_diff = df.index.min() - earliest_start_date
#             # Shift the index of each DataFrame
#         df.index = df.index - df1_time_diff

#     #df = df.resample('60T', label='right').mean()
#     df = df.interpolate()
#     ax.plot(df.mean(axis=1),label=all_data[i]["Population_alias"],color = colors_pop[i])
#     ax.fill_between(df.index,df.mean(axis=1)-df.std(axis=1),df.mean(axis=1)+df.std(axis=1),alpha=0.5,color = colors_pop[i])

#     if all_data[i]["Subspecies"] == "formosus":
#         df_aaf.append(df.mean(axis=1)) 
#     else:
#         df_aaa.append(df.mean(axis=1)) 


#     ax.legend(fontsize=15)

#     hh_mm = mdates.DateFormatter('%H')
#     ax.xaxis.set_major_formatter(hh_mm)
# #ax.legend(loc='best')
#     if i == 0:
#         ax.set_ylabel("Fraction of mosquitos flying")
#         ax.set_xlabel("hour of the day")
#     ax.set_ylim([0,0.12])
# plt.tight_layout()
# plt.savefig(folder_plots+"_comparision_aaf_aae_all.png",bbox_inches='tight')
# #plt.show()


# ############# Aaa vs aaf plot week average ###################
# fig, axes = plt.subplots(1, 1,dpi=200)
# fig.set_figheight(5)
# fig.set_figwidth(10)

# ax = axes

# for i,df in enumerate(df_aaf):
#     ax.plot_date(df.index,df.values,color="tab:blue",linestyle="-",marker=None,linewidth=1)

# for i,df in enumerate(df_aaa):
#     ax.plot_date(df.index,df.values,color="tab:orange",linestyle="-",marker=None,linewidth=1)

# df_aaf= pd.concat(df_aaf, axis=1)
# df_aaa= pd.concat(df_aaa, axis=1)
# ax.plot_date(df_aaa.mean(axis=1).index,df_aaa.mean(axis=1).values,color="tab:orange",linestyle="-",marker=None,linewidth=4,label="aegypti")
# ax.plot_date(df_aaf.mean(axis=1).index,df_aaf.mean(axis=1).values,color="tab:blue",linestyle="-",marker=None,linewidth=4,label="formosus")
# ax.fill_between(df_aaa.index,df_aaa.mean(axis=1)-df_aaa.std(axis=1),df_aaa.mean(axis=1)+df_aaa.std(axis=1),alpha=0.5,color="tab:orange")
# ax.fill_between(df_aaf.index,df_aaf.mean(axis=1)-df_aaf.std(axis=1),df_aaf.mean(axis=1)+df_aaf.std(axis=1),alpha=0.5,color="tab:blue")


# ax.legend(fontsize=20)
# hh_mm = mdates.DateFormatter('%H')
# ax.xaxis.set_major_formatter(hh_mm)
# ax.set_ylabel("Fraction of mosquitos flying")
# ax.set_xlabel("hour of the day")
# plt.tight_layout()
# plt.savefig(folder_plots+"_average_aaa_vs_aaf.png",bbox_inches='tight')






# ########## All correlations 
# ########### Activity as function of Latitude ###########
# fig, axes = plt.subplots(3,2,dpi=100)
# fig.set_figheight(20)
# fig.set_figwidth(20)

# time_int = [["7:00","10:00"],["10:00","14:00"],["14:00","20:00"],["20:00","22:00"],["00:00","6:00"],["6:00","23:59"]]

# for i,ax in enumerate(axes.reshape(-1)):

#     if i == 0: # Activity and Latitude
#         for k in np.arange(10):
#             df = all_data[k]["Fraction_flying_day"]
#             activity = get_mean_time_int(df.mean(axis=1),time_int[0][0],time_int[0][1])
#             if all_data[k]["Subspecies"] == "formosus":
#                 ax.scatter(all_data[k]["Latitude"],activity,s=300,label =all_data[k]["Population_alias"] )
#                 #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#             else:
#                 ax.scatter(all_data[k]["Latitude"],activity,s=300,marker="v" ,label =all_data[k]["Population_alias"] )
#                 #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#         #ax.set_ylim([0,0.06])
#         ax.legend(fontsize="20",ncol=2)
#         ax.set_ylabel('Average flight activity')
#         ax.set_xlabel('Latitude')

#     if i == 1: # Activity and generation
#         for k in np.arange(10):
#             df = all_data[k]["Fraction_flying_day"]
#             activity = get_mean_time_int(df.mean(axis=1),time_int[0][0],time_int[0][1])
#             if all_data[k]["Subspecies"] == "formosus":
#                 ax.scatter(all_data[k]["Generation"],activity,s=300,label =all_data[k]["Population_alias"] )
#                 #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#             else:
#                 ax.scatter(all_data[k]["Generation"],activity,s=300,marker="v" ,label =all_data[k]["Population_alias"] )
#                 #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#         #ax.set_ylim([0,0.06])
#         #ax.legend(fontsize="20",ncol=2)
#         ax.set_ylabel('Average flight activity')
#         ax.set_xlabel('Death rate(death/week)')

#     if i == 2: # Death rate and generation
#         for k in np.arange(10):
#             df = all_data[k]["Fraction_flying_day"]
#             activity = -all_data[k]["Death_rate"]
#             if all_data[k]["Subspecies"] == "formosus":
#                 ax.scatter(all_data[k]["Generation"],activity,s=300,label =all_data[k]["Population_alias"] )
#                 #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#             else:
#                 ax.scatter(all_data[k]["Generation"],activity,s=300,marker="v" ,label =all_data[k]["Population_alias"] )
#                 #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#         #ax.set_ylim([0,0.06])
#         #ax.legend(fontsize="20",ncol=2)
#         ax.set_ylabel('Death rate')
#         ax.set_xlabel('Generation')

#     if i == 3: # Activity and Latitude
#         for k in np.arange(10):
#             df = all_data[k]["Fraction_flying_day"]
#             activity_morning = get_mean_time_int(df.mean(axis=1),time_int[0][0],time_int[0][1])
#             activity_evening = get_mean_time_int(df.mean(axis=1),time_int[3][0],time_int[3][1])
#             ratio = activity_evening/activity_morning
#             if all_data[k]["Subspecies"] == "formosus":
#                 ax.scatter(all_data[k]["Latitude"],ratio ,s=300,label =all_data[k]["Population_alias"] )
#                 #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#             else:
#                 ax.scatter(all_data[k]["Latitude"],ratio ,s=300,marker="v" ,label =all_data[k]["Population_alias"] )
#                 #ax.annotate(all_data[k]["Population_alias"], (all_data[k]["Latitude"]*1.1, activity*1.1))
#         #ax.set_title(time_int[i][0]+"_"+time_int[i][1])
#         #ax.set_ylim([0,0.06])
#         #ax.legend(fontsize="20",ncol=2)
#         ax.set_ylabel('Bias to evening activity')
#         ax.set_xlabel('Latitude')



# plt.tight_layout()
# plt.savefig(folder_plots+"_overall_corr.png",bbox_inches='tight')

# ax =axes[1]
# # Concatenate the DataFrames into a single DataFrame with new column names
# df_aaf= pd.concat(df_aaf, axis=1)
# df_aaa= pd.concat(df_aaa, axis=1)

# ax.plot(df_aaf.mean(axis=1),label="A.aegypti formosus")
# ax.fill_between(df_aaf.index,df_aaf.mean(axis=1)-df_aaf.std(axis=1),df_aaf.mean(axis=1)+df_aaf.std(axis=1),alpha=0.5)
# ax.plot(df_aaa.mean(axis=1),label="A.aegypti aegypti")
# ax.fill_between(df_aaa.index,df_aaa.mean(axis=1)-df_aaa.std(axis=1),df_aaa.mean(axis=1)+df_aaa.std(axis=1),alpha=0.5)
# ax.set_ylabel("Fraction of mosquitos flying")
# ax.set_xlabel("hour of the day")
# hh_mm = mdates.DateFormatter('%H:%M')
# ax.xaxis.set_major_formatter(hh_mm)
# ax.legend(loc='best')
# ax.legend()







################


                
############ Activity as function of Latitude ###########

# fig, axes = plt.subplots(5, 2,dpi=200)
# fig.set_figheight(10)
# fig.set_figwidth(30)

# for k,ax in enumerate(axes.reshape(-1)):
#     print(all_data[k]["Population_alias"])

#     df = all_data[k]["Number_flying_mosquitos"]

#     ax.plot(df,label = all_data[k]["Population_alias"])
#     ax.legend()

# plt.savefig(folder_plots+"_raw_flight_activities.png",bbox_inches='tight')
# plt.show()
    














# fig, axes = plt.subplots(5, 2,dpi=100)
# fig.set_figheight(20)
# fig.set_figwidth(20)

# for k,ax in enumerate(axes.reshape(-1)):
#     print(all_data[k]["Population_alias"])

#     df = all_data[k]["Number_alive_mosquitos"]

#     ax.plot(df,label = all_data[k]["Population_alias"])
#     ax.set_ylim([0,45])
#     ax.legend()

# plt.savefig(folder_plots+"_number_alive.png",bbox_inches='tight')
# plt.show()

#print(all_data)

############ Get all mortality ########

# for k,exp in enumerate(files_mortality):
#     with open(folder_processed_data+exp+file, 'rb') as f:
#         df= pickle.load(f)





# ############## Week average ##########



# fig, axes = plt.subplots(1, 3,dpi=100)
# fig.set_figheight(5)
# fig.set_figwidth(5)

# for k,ax in enumerate(axes):
#     if k == 0:
#         ratio = ratio_morning
#     elif k == 1:
#         ratio = ratio_evening
#     elif k == 2:
#         ratio = ratio_night

#     for i,name in enumerate(exp_name):
        
#         if name == "Cage01_RAB_1" or name == "Cage04_KED_1" or name == "Cage03_KAK_1" or name == "Cage02_ZIK_1" or name == "Cage01_KUM_1":
#             ax.scatter(1,ratio[i],s=40)
#         else:   
#             ax.scatter(2,ratio[i],s=40)
#     ax.set_xlim([0,3])     
# plt.show()
# #plt.ylim([0,3])






