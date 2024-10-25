########################## IMPORT ALL NECESSARY PACKAGES ####################

import numpy as np
# import cv2
# import os
# from os import listdir
# from os.path import isfile, join
# import matplotlib.pyplot as plt
# import time
# import pandas as pd
# #from functions_Tracking_V2 import*
# import sys
# from scipy.spatial import distance as dist
# import pickle
# from resting_obj_tracker import resting_tracker
# from moving_obj_tracker import moving_tracker
# import scipy.stats as scp
# import scipy.sparse.csgraph as graph
# import datetime as dt
# from path_dict import PathDict
from collections import OrderedDict


#######################################################
#######################################################
class mosquito_obj_tracker:
    """
    This class encodes and manages the "tracks" objects, resting or flying.
    """

    def __init__(self,type_object):
        """
        Initialize tracks ensemble.
        """ 
        self.objects = OrderedDict() # Initialize the 
        self.number_objects = 0
        self.type = type_object # Resting, moving or both
        if type_object == "mixed":
            self.time_stamp = []


    def add_new_track(self,track_ID,new_coordinate,new_time_stamp,new_start_boot,new_end_boot,new_start_type,new_end_type):
        """
        Add a track.
        """
        track_dict = dict()
        track_dict["coordinates"] = new_coordinate
        track_dict["time_stamp"] = new_time_stamp
        track_dict["start_type"] = new_start_type
        track_dict["end_type"] = new_end_type
        track_dict["start_boot"] = new_start_boot
        track_dict["end_boot"] = new_end_boot

        self.objects[track_ID] = track_dict
        self.number_objects += 1


    def remove_track(self,track_ID):
        """
        Remove a track
        """
        del self.objects[track_ID]
        self.number_objects += -1

    def remove_points_from_track(self,track_ID,time_inter):
        """
        Remove a chunk of a trajectory.
        """
        if time_inter[1]==-1:
            del self.objects[track_ID]["coordinates"][time_inter[0]:]
            del self.objects[track_ID]["time_stamp"][time_inter[0]:]
        else:
            del self.objects[track_ID]["coordinates"][time_inter[0]:time_inter[1]]
            del self.objects[track_ID]["time_stamp"][time_inter[0]:time_inter[1]]
            #print(self.objects[track_ID]["time_stamp"])


    def add_points_to_track(self,track_ID,coord_to_add,time_stamp_to_add):
        """
        Add coordinates to an already existing track
        """
        if len(coord_to_add)>0:
            self.objects[track_ID]["coordinates"].append(coord_to_add)
            self.objects[track_ID]["time_stamp"].append(time_stamp_to_add)

    def merge_tracks(self,track_ID_1,track_ID_2):
        """
        Merge together tracks if possible
        """

    def add_mosquito_track(self,track_ID):
        """
        Add a mosquito track (computed from at least an entire video, mixing resting and moving tracks).
        """
        track_dict = dict()
        track_dict["coordinates"] = []
        track_dict["state"] = []
        track_dict["start_boot"] = np.nan
        track_dict["end_boot"] = np.nan
        track_dict["start"] = np.nan
        track_dict["end"] = np.nan

        self.objects[track_ID] = track_dict
        self.number_objects += 1



    



#     idx_of_min_back = np.argmin(D_traj, axis=1)




        # for frame_idx in np.arange(self.total_number_frames):
        #     print(frame_idx)

        #     #progress_bar(frame_idx, self.total_number_frames, bar_length=20)
        #     # forward_centroid_t = []
        #     # forward_id_t = []
        #     # #Get the ID and coordinate of foward trackers at t = frame_idx
        #     # for k,rest_id in enumerate(list_of_ID_resting_forward): # For all ID of forward tracking
        #     #     if frame_idx in resting_object_traj["time_ID_"+str(rest_id)]:
        #     #         t_frame = resting_object_traj["time_ID_"+str(rest_id)].index(frame_idx)
        #     #         if t_frame<len(resting_object_traj["ID_"+str(rest_id)]):
        #     #             centroid = resting_object_traj["ID_"+str(rest_id)][t_frame]
        #     #             forward_centroid_t.append(centroid)
        #     #             forward_id_t.append(k)


        #     #Get the ID and coordinate of backward trackers at t = frame_idx
        #     backward_centroid_t = []
        #     backward_id_t = []
        #     for j,rest_id_back in enumerate(list_of_ID_resting_backward): # For all ID of forward tracking
        #         time_v = back_resting_object_traj["time_ID_"+str(rest_id_back)]
        #         pos = back_resting_object_traj["ID_"+str(rest_id_back)]
        #         if n_frames-frame_idx in time_v:
        #             t_frame = time_v.index(n_frames-frame_idx)
        #         #if t_frame<len(back_resting_object_traj["ID_"+str(rest_id_back)]):
        #             centroid = pos[t_frame]
        #             backward_centroid_t.append(centroid)
        #             backward_id_t.append(j)

        # matching_matrix_rest = np.where(matching_matrix_rest>40,1,0)
        # matched_ids = graph.connected_components(matching_matrix_rest)


        # print(matched_ids)

        


# if len(forward_centroid_t)>0 and len(backward_centroid_t)>0:
#     D_traj = dist.cdist(np.array(forward_centroid_t),np.array(backward_centroid_t))
#     idx_of_min_back = np.argmin(D_traj, axis=1)
#     min_vec = np.min(D_traj, axis=1)
#     #print(idx_of_min_back)
#     for i,for_id in enumerate(forward_id_t):
#         back_id = backward_id_t[idx_of_min_back[i]] # Index backward
#         if min_vec[i] < max_dist_rest :
#             matching_matrix_rest[for_id][back_id+numb_id_resting_forward] +=1



        # Find all the landings time
        # t_landings = []
        # coord_landings = []
        # id_landings = []
        # for id_landing in list_of_ID_still:
        #     pos = resting_object_traj["ID_"+str(id_landing)]
        #     time_v = resting_object_traj["time_ID_"+str(id_landing)]
        #     if time_v[0]>2:
        #         #print("ID_"+str(ID)+" landing at t = "+str(time_v[0]))
        #         t_landings.append(time_v[0])
        #         coord_landings.append(pos[0])
        #         id_landings.append(id_landing)
        # print(id_landings)


        # # Initialize the data structures to save results
        # already_tracked_IDs = []
        # not_yet_tracked_IDs = list_of_ID_still
        # remaining_IDs = []
        # trajectories = dict()

        # ##### For each ID, start from the take-off and look for a landing site
        # new_ID = 0

        # for i,current_id in list_of_ID_still:
        #     takeoff_status,time_take_off,coord_take_off = track_taking_off(resting_object_traj,current_ID,moving_object_traj)
        #     print(takeoff_status)




        # while len(not_yet_tracked_IDs)>0:

        #     current_ID = not_yet_tracked_IDs[0]
        #     trajectory = []
        #     time_trajectory = []
        #     continue_tracking = 1
        #     print("start tracking ID "+str(current_ID))
        #     while continue_tracking == 1:
        #         takeoff_status,time_take_off,coord_take_off = track_taking_off(resting_object_traj,current_ID,moving_object_traj)
                #print(takeoff_status)
                # if takeoff_status == "ENDED":
                #     already_tracked_IDs.append(current_ID)
                #     if current_ID in not_yet_tracked_IDs:
                #         not_yet_tracked_IDs.remove(current_ID)
                #     continue_tracking = 0 # Stop tracking that ID
                #     print("ID "+str(current_ID)+" is resting until the end of the video")
                #     # Save the trajectory with the current ID
                #     trajectory,time_trajectory = update_trajectory(resting_object_traj["ID_"+str(current_ID)],resting_object_traj["time_ID_"+str(current_ID)],trajectory,time_trajectory)
                    

            #     elif takeoff_status == "NOT_FOUND":
            #         continue_tracking = 0 # Stop tracking that ID
            #         remaining_IDs.append(current_ID)
            #         if current_ID in not_yet_tracked_IDs:
            #             not_yet_tracked_IDs.remove(current_ID)
            #         print("ID "+str(current_ID)+" could not be taken-off")
            #         trajectory,time_trajectory = update_trajectory(resting_object_traj["ID_"+str(current_ID)][0:-30],resting_object_traj["time_ID_"+str(current_ID)][0:-30],trajectory,time_trajectory)

            #         # Save the trajectory without the current ID

            #     # Mosquito took-off, starting to track the flight trajectory to find landing
            #     elif takeoff_status == "STARTED":
            #         print("ID "+str(current_ID)+" has taken off")
            #         flying_track_status,flying_track,time_flying_track,landing_ID = flying_ID_track(current_ID,resting_object_traj,centroids_moving_sorted,time_take_off,coord_take_off,id_landings,coord_landings,t_landings)
            #         t_stop = resting_object_traj["time_ID_"+str(current_ID)].index(time_take_off)
            #         if flying_track_status == "LANDED":
            #             continue_tracking = 0
            #             print("ID "+str(current_ID)+" has landed and matched with ID : "+str(landing_ID))
            #             # Add the flight trajectory
            #             already_tracked_IDs.append(current_ID)
            #             if current_ID in not_yet_tracked_IDs:
            #                 not_yet_tracked_IDs.remove(current_ID)
                        
                        
            #             trajectory,time_trajectory = update_trajectory(resting_object_traj["ID_"+str(current_ID)][0:t_stop],resting_object_traj["time_ID_"+str(current_ID)][0:t_stop],trajectory,time_trajectory)
            #             trajectory,time_trajectory = update_trajectory(flying_track,time_flying_track,trajectory,time_trajectory)
            #             current_ID = landing_ID

            #         elif flying_track_status == "NOT_LANDED":
            #             print("ID "+str(current_ID)+" has not landed")
            #             continue_tracking = 0 # Stop tracking that ID
            #             remaining_IDs.append(current_ID)
            #             if current_ID in not_yet_tracked_IDs:
            #                 not_yet_tracked_IDs.remove(current_ID)
            #             trajectory,time_trajectory = update_trajectory(resting_object_traj["ID_"+str(current_ID)][0:t_stop],resting_object_traj["time_ID_"+str(current_ID)][0:t_stop],trajectory,time_trajectory)
            #             trajectory,time_trajectory = update_trajectory(flying_track,time_flying_track,trajectory,time_trajectory)
                    

            # # Save the trajectory that was extracted
            # if len(trajectory)>0:
            #     trajectories["ID_"+str(new_ID)] = trajectory
            #     trajectories["time_ID_"+str(new_ID)] = time_trajectory
            #     new_ID += 1

            #     print("Still "+str(len(not_yet_tracked_IDs))+" IDs to be tracked, new ID is "+str(new_ID))




########################## Detect taking-off ##############
    # def track_taking_off(self,resting_object_traj,current_ID,moving_object_traj):

    #     # Parameters
    #     distance_take_off_max = 100
    #     time_window = 60
    #     time_take_off_target = -30 # The time at which the object disappeared

    #     time_take_off = np.nan
    #     coord_take_off = np.nan
        
    #     list_of_ID_still = get_ids_from_dict(resting_object_traj)
    #     list_of_ID_moving = get_ids_from_dict(moving_object_traj)

    #     #### Find the take-off time             
    #     pos = resting_object_traj["ID_"+str(current_ID)]
    #     time_v = resting_object_traj["time_ID_"+str(current_ID)]


    #     ################## Find the time of take-off 
    #     if time_v[-1]<self.total_number_frames-1: ### If tracked stopped before the end of the movie
    #         distance_take_off = 200

    #         for delta in np.arange(time_window):
    #             if len(pos)>time_window:
    #                 coord_still = pos[-int(time_window/2)+time_take_off_target+delta]
    #                 t = time_v[-int(time_window/2)+time_take_off_target+delta]
    #                 if len(centroids_moving_sorted[t])>0:
    #                     D = dist.cdist(np.array(coord_still).reshape(1,-1),np.array(centroids_moving_sorted[t])) ### Maximum distance
    #                     #print(D[0].min(axis=0))
    #                     if D[0].min(axis=0)<distance_take_off:
    #                         distance_take_off = D[0].min(axis=0)
    #                         time_take_off = t
    #                         coord_take_off = coord_still
    #         if distance_take_off < distance_take_off_max:
    #             takeoff_status = "STARTED"
    #         else:
    #             takeoff_status = "NOT_FOUND"
    #             coord_take_off = np.nan
    #             time_take_off = np.nan


    #     else: # Track goes to the end of the movie
    #         takeoff_status = "ENDED"
    #         coord_take_off = np.nan
    #         time_take_off = np.nan

    #     return takeoff_status,time_take_off,coord_take_off





# def extract_trajectories(resting_object_traj,centroids_moving_sorted):
#     nb_ID = int(len(resting_object_traj)/2)







############################## TAKING_OFF TRACKING ############################################




# ############################## TRACK FLIGHT FROM TAKING-OFF TO LANDING ############################################
# def flying_ID_track(current_ID,resting_object_traj,centroids_moving_sorted,time_take_off,coord_take_off,id_landings,coord_landings,t_landings):
#     # Parameters for tracking flying object and finding landingx
#     max_search_distance = 2 # Initial search distance
#     max_search_distance_upper_limit = 80
#     max_time_travel = 20
#     max_landing_distance = 60
#     max_landing_time = 60

#     find_match=0
#     while max_search_distance<max_search_distance_upper_limit and find_match==0:
#         coord_mov = [coord_take_off]
#         t_search = time_take_off

#         distance_flight = 0
#         coord_moving_traj = []
#         while t_search<len(centroids_moving_sorted)-1 and len(coord_mov)>0 : # If vidoe no ended and still objects being tracked

#             ################ Look for landing match among the moving objects
#             for j,pos_land in enumerate(coord_landings):
#                 D_landing = dist.cdist(np.array(pos_land).reshape(1,-1),np.array(coord_mov))
#                 min_distance = np.min(D_landing[0])

#                 if min_distance<max_landing_distance and find_match == 0:
#                     if np.abs(t_search-t_landings[j])<max_landing_time:
#                         min_positions = [i for i, x in enumerate(D_landing[0]) if x == min_distance]
#                         if id_landings[j] == current_ID:
#                             find_match = 0
#                         else:
#                             find_match = 1 # Match was found
#                             print("landing after distance search of : "+str(max_search_distance))
#                             landing_ID = id_landings[j]
#                         #show_trajectory(coord_moving_traj,pos,time_v,pos_land,t_landings[j],video_path,time_take_off,resting_object_traj,ID_landing[j])
            
#             ################ Try to match the moving objects of t-1 and t
#             coord_mov_next = []
#             for k in np.arange(len(centroids_moving_sorted[t_search])):
#                 D_all = [] # List of all distances
#                 for i in np.arange(len(coord_mov)):
#                     D = dist.cdist(np.array(coord_mov[i]).reshape(1,-1),np.array(centroids_moving_sorted[t_search][k]).reshape(1,-1))
#                     D_all.append(D)
#                 if np.min(D_all)<max_search_distance:
#                     coord_mov_next.append(centroids_moving_sorted[t_search][k])
#                     distance_flight += np.min(D_all)
                    
#             ############If no moving objects found, try to find a match in the futur
#             time_travel = 0
#             while len(coord_mov_next)==0 and time_travel<max_time_travel and t_search+time_travel < len(centroids_moving_sorted)-1:
#                 coord_mov_next = []
#                 for k in np.arange(len(centroids_moving_sorted[t_search+time_travel])):
#                     D_all = [] # List of all distances
#                     for i in np.arange(len(coord_mov)):
#                         D = dist.cdist(np.array(coord_mov[i]).reshape(1,-1),np.array(centroids_moving_sorted[t_search+time_travel][k]).reshape(1,-1))
#                         D_all.append(D)
#                     if np.min(D_all)<max_search_distance:
#                         coord_mov_next.append(centroids_moving_sorted[t_search+time_travel][k])
#                         distance_flight += np.min(D_all)
#                 time_travel += 1

#             coord_mov = coord_mov_next
#             if find_match == 0:
#                 coord_moving_traj.append(coord_mov)

#             t_search += 1    
    
#         max_search_distance += 2

#     if find_match == 0:
#         flying_track_status = "NOT_LANDED"
#         print("Has searched for" + str(t_search-time_take_off))
#         #flying_track = np.nan
#         #time_flying_track = np.nan
#         landing_ID = np.nan
#         flying_track,time_flying_track = choose_flight_track(coord_moving_traj,time_take_off)

#     else:
#         flying_track_status = "LANDED"
#         flying_track,time_flying_track = choose_flight_track(coord_moving_traj,time_take_off)

#     return flying_track_status,flying_track,time_flying_track,landing_ID

# ############################## ############################## ############################## ##############################

# def choose_flight_track(coord_moving_traj,time_take_off):
#     #print(coord_moving_traj[-1])
#     len_traj = len(coord_moving_traj)
#     time_flying_track = [time_take_off+t for t in np.arange(len_traj)-1]
#     flying_track = []
#     for t in np.arange(len_traj-1):
#         x = 0
#         y = 0
#         n_centroid = 0
#         #print(coord_moving_traj[t])
#         for centroid in coord_moving_traj[t]:
#             n_centroid += 1
#             x += centroid[0]
#             y += centroid[1]
#         x = x/n_centroid
#         y = y/n_centroid
#         flying_track.append((x,y))
#     return flying_track,time_flying_track

