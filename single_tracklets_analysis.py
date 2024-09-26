

import sys
import math
sys.path.append('/Volumes/BBB/Theo_projects/BuzzWatch/buzzwatch_analysis_module/') # ADD PATH OF THE BUZZWATCH PYTHON MODULE
from buzzwatch_data_analysis.experiment_analysis import*
from buzzwatch_data_analysis.misc_functions import *
from buzzwatch_data_analysis.misc_functions import *



class single_tracklets_analysis:

    # dist_moving_obj = 30
    # dist_resting_obj = 15
    # min_duration_resting_traj = 125

    def __init__(self,experiment_class,f,debug_mode):
        self.video_mp4_name = experiment_class.list_videos_files[f]
        self.folder_analysis = experiment_class.folder_analysis
        self.video_name = os.path.splitext(self.video_mp4_name)[0]
        self.video_path = experiment_class.folder_videos + self.video_mp4_name
        self.background_path = experiment_class.background_path
        self.settings = experiment_class.settings # copy setting file
        self.control_border_points = self.settings["control_border_points"]
        self.cage_border_points = self.settings["cage_border_points"]
        self.sugar_border_points = self.settings["sugar_border_points"]



        tracklets_stats = {
            "position" : value1,
            "velocity" : value2,
            "angle" : value3,
            "angle_velocity" : value3,
            "curvature" : value3,
            "autocorrelation" : ,
            ""


            }
















# #from buzzwatch_data_analysis.single_video_analysis import*
# folder_plots  ="/Volumes/BBB/Theo_projects/BuzzWatch/ANALYZED/single_trajectory_clustering_Sept2023/plots/"
# def concatenate_files(file):
#     try:
#         with open(file, 'rb') as f:
#             mosquito_tracks = pickle.load(f)
#             df = mosquito_tracks.flight_population_activity
#             return df.resample('1T', label='right').mean()
        
#     except Exception as e:
#         print(e)

   
# def plot_flight_trajectory(x_f,y_f,ax,background,z):
#     # Set plot and add background
    
#     im = plt.imread(background)
    #ax.imshow(im,zorder=1,cmap = "gray")


    #x_f = x#uniform_filter1d(x, size=5)
   #y_f = yuniform_filter1d(y, size=5)

    #z = uniform_filter1d(z, size=5)

#     c_f = np.arange(len(x_f))
#     c_f = np.array([c_f[i]/25 for i in np.arange(len(c_f))])

#     points = np.array([x_f, y_f]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)

#     # Create a continuous norm to map from data points to colors
#     norm = plt.Normalize(np.min(c_f), np.max(c_f))
#     lc = LineCollection(segments, cmap='viridis', norm=norm)

#     #print(c_f)

#     # Set the values used for colormapping
#     lc.set_array(c_f)
#     lc.set_linewidth(4)
#     line = ax.add_collection(lc)

#     ax.set_xlim([0 ,im.shape[0]])
#     ax.set_ylim([0 ,im.shape[1]])
#     ax.set_aspect('equal')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     #plt.tight_layout()
#     return ax   

# ############## Function to compute features of trajectories
# def speed_variance(segment):
#     speeds = np.linalg.norm(np.diff(segment, axis=0), axis=1)
#     return np.var(speeds)

# def centroid_displacement(segment):
#     centroid_start = np.mean(segment, axis=0)
#     centroid_end = np.mean(segment[-1], axis=0)
#     return np.linalg.norm(centroid_end - centroid_start)

# def net_displacement(segment):
#     return np.linalg.norm(segment[-1] - segment[0])

# def direction_changes(segment):
#     # Calculate the angle between consecutive segments and count changes
#     angles = [np.arctan2(segment[i + 1][1] - segment[i][1], segment[i + 1][0] - segment[i][0]) for i in range(len(segment) - 1)]
#     angle_diffs = np.diff(angles)
#     positive_changes = np.sum(angle_diffs > 0)
#     negative_changes = np.sum(angle_diffs < 0)
#     total_changes = positive_changes + negative_changes
#     return total_changes

# def curvature(segment):
#     a = np.linalg.norm(segment[:-2] - segment[1:-1], axis=1)
#     b = np.linalg.norm(segment[:-2] - segment[2:], axis=1)
#     c = np.linalg.norm(segment[1:-1] - segment[2:], axis=1)
#     perimeters = a + b + c
#     areas = np.sqrt(np.abs((perimeters/2) * (perimeters/2 - a) * (perimeters/2 - b) * (perimeters/2 - c)))
#     curvatures = (4 * areas) / (perimeters ** 2)
#     return np.sum(curvatures)

# def angle3pt(a, b, c):
#     """Counterclockwise angle in degrees by turning from a to c around b
#         Returns a float between 0.0 and 360.0"""
#     ang = math.degrees(
#         math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
#     return ang + 360 if ang < 0 else ang
 
# def compute_angles(x, y):

#     angle = np.zeros(len(x))
#     for i in np.arange(len(x)-2):
#         a = [x[i],y[i]]
#         b = [x[i+1],y[i+1]]
#         c = [x[i+2],y[i+2]]
#         angle[i+1] = angle3pt(a, b, c)
#     return angle
# ######################### USER INPUT PARAMETERS ###################
# INITIAL_NUMBER_MOSQUITO = 40
# SIZE_OF_CAGE_CM = 15
# SINGLE_MOSQUITO_SIZE_RANGE_MM = [2,5] # [min_length,max_length]

# ######################### USER INPUT VARIABLES ###################
# #CAGE_NAME = "Cage02_RAB"
# #BATCH_NB = 3
# #FOLDER_EXP = "230224_RAB_KUM"

# CAGE_NAME = "Cage01_KPP_1"
# BATCH_NB = 1
# FOLDER_EXP = "230704_KPP_KUM"
# ######################### USER INPUT VARIABLES ###################
# folder_root = "/Volumes/BBB/Theo_projects/BuzzWatch/"
# folder_analysis =folder_root+"ANALYZED/"+FOLDER_EXP+"/"+CAGE_NAME+"/" ####### FOLDER WITH ANALYSIS
# folder_videos =folder_root+"RAW_DATA/"+FOLDER_EXP+"/"+CAGE_NAME+"/batch_"+str(BATCH_NB)+"/" 
# experiment_alias = CAGE_NAME+"_"+str(BATCH_NB)
# settings_file = folder_analysis+"buzzwatch_track_settings.yml"
# ##################################### START ANALYSIS ##############################################################
# run_from_scratch = 1
# ##################################### INITIALIZE EXPERIMENT ANALYSIS ##############################################
# exp_object_path = folder_analysis+"temp_data/"+"temp_data_"+experiment_alias+".pkl"
# if run_from_scratch == 1 or os.path.isfile(exp_object_path)==0:
#     experiment = buzzwatch_experiment_analysis(folder_analysis,folder_videos,experiment_alias,settings_file,debug_mode=1)
# else:
#     with open(exp_object_path, 'rb') as f:
#         #print("Load "+exp_object_path )
#         experiment = pickle.load(f)    


# ###### GET THE BACKGROUND(s) IMAGE(s) ######

# files_tracking = [f for f in listdir(experiment.folder_final) if isfile(join(experiment.folder_final, f)) and f.startswith("forward")]
# files_tracking.sort()
# files_tracking = [experiment.folder_final+"/"+file for file in files_tracking]

# ############################### Extract trajectories ############
# all_traj = []
# for n in [145, 150, 160, 230]:

#     file = files_tracking[n]
#     with open(file, 'rb') as f:
#         mosquito_tracks = pickle.load(f)

#     for k,id in enumerate(mosquito_tracks.objects.keys()):
#         state_v = mosquito_tracks.objects[id]["state"]
#         coord_v = mosquito_tracks.objects[id]["coordinates"]

#         state_v = np.array([1-i for i in state_v])
#         runs = zero_runs(state_v)
#         nb_tracks = len(runs[:,0])

#         x = [a for a,b in  coord_v]
#         y = [b for a,b in  coord_v]

#         smooth_length = 5
#         x = uniform_filter1d(x, size=smooth_length)
#         y = uniform_filter1d(y, size=smooth_length)
#         #z = compute_angles(x, y)


#         if nb_tracks>0:
#             for k in np.arange(nb_tracks):
#                 t_i = runs[k,0]
#                 t_f = runs[k,1]

#                 d_x = np.array(x[t_i+1:t_f]) - np.array(x[t_i:t_f-1])
#                 d_y = np.array(y[t_i+1:t_f]) - np.array(y[t_i:t_f-1])

#                 d_x_2 = [np.square(z) for z in d_x]
#                 d_y_2 = [np.square(z) for z in d_y]

#                 dist = [np.sqrt(d_x_2[i]+d_y_2[i]) for i in np.arange(len(d_x))]



#                 if t_f-t_i>25*5: # If lasts more than 5 secs

#                     if np.mean(dist)>5: # If average speed higher than a threshold

#                         xy = list(zip(x[t_i+1:t_f-1], y[t_i+1:t_f-1]))  # list of points in 2D space
#                         all_traj.append(xy)
#                         print(np.std(dist)/np.mean(dist))

                        
#                         #print(direction_changes(xy))


# print(len(all_traj))


######################### Trajectories 
# n = 145
# file = files_tracking[n]
# print(file)

# with open(file, 'rb') as f:
#     mosquito_tracks = pickle.load(f)


# fig, axes = plt.subplots(5, 5,dpi=400,constrained_layout = True)
# fig.set_figheight(20)
# fig.set_figwidth(20)
# axes  = axes.reshape(-1)
# nb_plotted = 0
# for k,id in enumerate(mosquito_tracks.objects.keys()):
#     state_v = mosquito_tracks.objects[id]["state"]
#     coord_v = mosquito_tracks.objects[id]["coordinates"]

#     state_v = np.array([1-i for i in state_v])
#     runs = zero_runs(state_v)
#     nb_tracks = len(runs[:,0])

#     x = [a for a,b in  coord_v]
#     y = [b for a,b in  coord_v]

#     smooth_length = 10
#     x = uniform_filter1d(x, size=smooth_length)
#     y = uniform_filter1d(y, size=smooth_length)
#     #z = compute_angles(x, y)


#     if nb_tracks>0:
#         for k in np.arange(nb_tracks):
#             t_i = runs[k,0]
#             t_f = runs[k,1]

#             d_x = np.array(x[t_i+1:t_f]) - np.array(x[t_i:t_f-1])
#             d_y = np.array(y[t_i+1:t_f]) - np.array(y[t_i:t_f-1])

#             d_x_2 = [np.square(z) for z in d_x]
#             d_y_2 = [np.square(z) for z in d_y]

#             dist = [np.sqrt(d_x_2[i]+d_y_2[i]) for i in np.arange(len(d_x))]



#             if t_f-t_i>25*5:
#                 #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])

#                 if nb_plotted < len(axes) and np.mean(dist)>5:
#                     #z = np.abs(180-compute_angles(x[t_i+1:t_f], y[t_i+1:t_f]))
#                     try:
#                         #xy = list(zip(x[t_i+10:t_f-10], y[t_i+10:t_f-10]))  # list of points in 2D space
#                         #print(xy)
#                         #curv = Curvature(line=xy)
#                         #curv.calculate_curvature(gap=3)
#                         #print(curv.curvature)
                        
#                         #axes[nb_plotted].plot(curv.curvature)
#                         ax = plot_flight_trajectory(x[t_i+1:t_f],y[t_i+1:t_f],axes[nb_plotted],experiment.background_path,z=1)
#                         nb_plotted += 1
#                     except Exception:
#                         print("aahhaha")
# # set the spacing between subplots
# plt.subplots_adjust(
#                     wspace=0.,
#                     hspace=0.)#plt.tight_layout()
# plt.ioff()
# plt.savefig(folder_plots+"traj_video_n_"+str(n)+'.png',bbox_inches='tight')
# plt.savefig(folder_plots+"traj_video_n_"+str(n)+'.pdf',bbox_inches='tight',format = 'pdf')










################ Curvature ###########

# n = 145
# file = files_tracking[n]
# print(file)

# with open(file, 'rb') as f:
#     mosquito_tracks = pickle.load(f)


# fig, axes = plt.subplots(2, 2,dpi=200)
# fig.set_figheight(10)
# fig.set_figwidth(10)
# axes  = axes.reshape(-1)
# nb_plotted = 0
# for k,id in enumerate(mosquito_tracks.objects.keys()):
#     state_v = mosquito_tracks.objects[id]["state"]
#     coord_v = mosquito_tracks.objects[id]["coordinates"]

#     state_v = np.array([1-i for i in state_v])
#     runs = zero_runs(state_v)
#     nb_tracks = len(runs[:,0])

#     x = [a for a,b in  coord_v]
#     y = [b for a,b in  coord_v]

#     smooth_length = 5
#     x = uniform_filter1d(x, size=smooth_length)
#     y = uniform_filter1d(y, size=smooth_length)
#     #z = compute_angles(x, y)


#     if nb_tracks>0:
#         for k in np.arange(nb_tracks):
#             t_i = runs[k,0]
#             t_f = runs[k,1]

#             d_x = np.array(x[t_i+1:t_f]) - np.array(x[t_i:t_f-1])
#             d_y = np.array(y[t_i+1:t_f]) - np.array(y[t_i:t_f-1])

#             d_x_2 = [np.square(z) for z in d_x]
#             d_y_2 = [np.square(z) for z in d_y]

#             dist = [np.sqrt(d_x_2[i]+d_y_2[i]) for i in np.arange(len(d_x))]



#             if t_f-t_i>25*5:
#                 #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])

#                 if nb_plotted < len(axes) and np.mean(dist)>5:
#                     #z = np.abs(180-compute_angles(x[t_i+1:t_f], y[t_i+1:t_f]))
#                     try:
#                         xy = list(zip(x[t_i+10:t_f-10], y[t_i+10:t_f-10]))  # list of points in 2D space
#                         #print(xy)
#                         curv = Curvature(line=xy)
#                         curv.calculate_curvature(gap=3)
#                         #print(curv.curvature)
                        
#                         axes[nb_plotted].plot(curv.curvature)
#                         #ax = plot_flight_trajectory(x[t_i+1:t_f],y[t_i+1:t_f],axes[nb_plotted],experiment.background_path,z)
#                         nb_plotted += 1
#                     except Exception:
#                         print("aahhaha")
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.ioff()
# plt.savefig(folder_plots+"angle_video_n_"+str(n)+'.png',bbox_inches='tight')