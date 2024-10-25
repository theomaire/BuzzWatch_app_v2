########################## IMPORT ALL NECESSARY PACKAGES ####################
import numpy as np
import cv2
import os
import pickle
from buzzwatch_data_analysis.resting_obj_tracker import *
from buzzwatch_data_analysis.moving_obj_tracker import *
from buzzwatch_data_analysis.misc_functions import *
from buzzwatch_data_analysis.mosquito_obj_tracker import *
import copy
import yaml
import time
from scipy.spatial import distance as dist
import scipy.sparse.csgraph as graph
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timedelta
import warnings
from logger import MultiLogger

##################################################################################################################################
########################## Class that manages tracking of a single video ##############################################
class single_video_analysis:

    # dist_moving_obj = 30
    # dist_resting_obj = 15
    # min_duration_resting_traj = 125

    def __init__(self,experiment_class,video_name,debug_mode):
        self.video_mp4_name = video_name+".mp4"#experiment_class.list_videos_files[f]
        self.folder_analysis = experiment_class.folder_analysis
        self.video_name = video_name#os.path.splitext(self.video_mp4_name)[0]
        self.video_path = os.path.join(experiment_class.folder_videos , self.video_mp4_name)
        self.background_path = experiment_class.background_path
        self.folder_temp = experiment_class.folder_temp_data
        self.folder_final = experiment_class.folder_final
        self.settings = experiment_class.settings # copy setting file
        self.folder_test_tracking  = experiment_class.folder_test_tracking
        self.control_border_points = self.settings["control_border_points"]
        self.cage_border_points = self.settings["cage_border_points"]
        self.sugar_border_points = self.settings["sugar_border_points"]
        self.mosquito_tracks = None
        self.settings_file = experiment_class.settings_file

        cap = cv2.VideoCapture(self.video_path)
        n_frame = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
        cap.release()
        self.total_number_frames = n_frame

        #objs_resting_tracked = resting_tracker(se)
        #self.maxobjdisappear_resting = objs_resting_tracked.maxDisappeared
        if debug_mode:
            self.video_object_path = self.folder_temp+"/temp_data_"+self.video_name+".pkl"
            with open(self.video_object_path, 'wb') as f:
                pickle.dump(self, f)
        
        print("Starting analysis of "+self.video_name) 

########################## Run video and segment both resting and moving points ####################
    def segment_resting_and_moving_objects(self,step_to_force_analyze,debug_mode):
        """ 
        Segment each frame of video for resting and moving objects
        """
        if hasattr(self, 'resting_objects')==0 or step_to_force_analyze<=1:
            print("Start running segmentation")

            # Parameters
            dist_moving_obj = self.settings["seg_moving"]["dist_moving_obj"] # Can be changed
            dist_resting_obj = self.settings["seg_resting"]["dist_resting_obj"] # Can be changed

            # Load background
            #
            path_back = self.folder_analysis+"/images_mortality/"+self.video_name+".png"
            #path_back = self.background_path
            check_file = os.path.isfile(path_back)
            if check_file: # If the background file exists
                #background = cv2.imread(self.background_path)
                background = cv2.imread(path_back)
                background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                
                # Initialize video variables 
                cap = cv2.VideoCapture(self.video_path)

                moving_objects_forward = [[] for i in range(self.total_number_frames)]
                moving_objects_backward = [[] for i in range(self.total_number_frames)]
                resting_objects = [[] for i in range(self.total_number_frames)]

                ####### Start reading the video 
                frame_idx = 0       
                while frame_idx < self.total_number_frames-1:
                    suc,frame = cap.read()
                    if frame_idx%1000 == 0:
                        if debug_mode:
                            #multi_logger = MultiLogger(None, 'logfile.log')
                            progress_bar(frame_idx, self.total_number_frames, bar_length=20)
                    
                    if suc == True:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame_gray = cv2.absdiff(frame_gray,background)

                        if frame_idx > 1:
                            # Get the moving objects forward
                            moving = self.get_centroids_moving_objects(frame_gray,prev_gray,self.settings["seg_moving"])
                            if len(moving)>0:
                                for centroid_moving in moving:
                                    if self.point_inside_cage(self.cage_border_points,centroid_moving):
                                        if len(moving_objects_forward[frame_idx+1])>0:
                                            D = dist.cdist(np.array(centroid_moving).reshape(1,-1), moving_objects_forward[frame_idx+1])
                                            if D[0].min(axis=0) > dist_moving_obj:            
                                                moving_objects_forward[frame_idx+1].append(centroid_moving)
                                        else:
                                            moving_objects_forward[frame_idx+1].append(centroid_moving)

                            # Get the moving objects backward
                            moving = self.get_centroids_moving_objects(prev_gray,frame_gray,self.settings["seg_moving"])
                            if len(moving)>0:
                                for centroid_moving in moving:
                                    if self.point_inside_cage(self.cage_border_points,centroid_moving):
                                        if len(moving_objects_backward[frame_idx])>0:
                                            D = dist.cdist(np.array(centroid_moving).reshape(1,-1), moving_objects_backward[frame_idx])
                                            if D[0].min(axis=0) > dist_moving_obj :            
                                                moving_objects_backward[frame_idx].append(centroid_moving)
                                        else:
                                            moving_objects_backward[frame_idx].append(centroid_moving)

                            # Get resting objects
                            centroids_still = self.get_centroids_still_objects(frame_gray,self.settings["seg_resting"])
                            if len(centroids_still)>0:
                                for centroid in centroids_still:
                                    if self.point_inside_cage(self.cage_border_points,centroid):
                                        if len(resting_objects[frame_idx])>0:
                                            D = dist.cdist(np.array(centroid).reshape(1,-1), resting_objects[frame_idx])
                                            if D[0].min(axis=0) > dist_resting_obj :            
                                                resting_objects[frame_idx].append(centroid)
                                        else:
                                            resting_objects[frame_idx].append(centroid)        
                            
                        frame_idx += 1
                        prev_gray = frame_gray
                cap.release()
                

                # Save intermediate variables

                self.resting_objects = resting_objects
                self.moving_objects_forw = moving_objects_forward
                moving_objects_backward.reverse()
                self.moving_objects_back = moving_objects_backward

                if debug_mode:
                    with open(self.video_object_path, 'wb') as f:
                        pickle.dump(self, f)
                print("Finished segmentation of video without error")
            else:
                print("background image missing")

        else: #Skip segmentation
            print("Segmentation already done")
            
########################## Check if a point inside the user-defined cage borders
    def point_inside_cage(self,border,coord):
        cage_borders_points = [tuple(border[i]) for i in np.arange(4)]
        
        cage_borders_points = np.array(cage_borders_points, dtype=np.int32)
        cage_borders_points = cage_borders_points.reshape((-1,1,2))
        #print(cage_borders_points)
        #point = np.array(coord, dtype=np.int32)
        #print(point)
        
        return cv2.pointPolygonTest(cage_borders_points, coord, False) >= 0

########################## Segment "resting" object ####################
    def get_centroids_still_objects(self,frame_gray,settings):
        # Parameters resting :
        gray_tresh = settings["gray_tresh"]
        max_elongation_ratio = settings["max_elongation_ratio"]
        max_length = settings["max_length"]
        min_length = settings["min_length"]

        # Conventio
        _, thresh = cv2.threshold(frame_gray, gray_tresh, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        inputCentroids = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # Save the centroid of the bounding rectangle
            if max([w/h,h/w])< max_elongation_ratio and max([h,w])<max_length and min([h,w])>min_length:
                inputCentroids.append((x+w/2, y+h/2))

        return inputCentroids

########################## Segment "moving" objects ####################
    def get_centroids_moving_objects(self,frame_gray,prev_gray,settings):
        # Parameters moving :
        gray_tresh = settings["gray_tresh"]
        max_elongation_ratio = settings["max_elongation_ratio"]
        max_length = settings["max_length"]
        min_length = settings["min_length"]

        diff_gray = cv2.subtract(prev_gray,frame_gray)
        blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur,gray_tresh, 255, cv2.THRESH_BINARY) # Modified the thresh value from 12 to 5
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        inputCentroids = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # Save the centroid of the bounding rectangle
            if max([w/h,h/w])< max_elongation_ratio and max([h,w])<max_length and min([h,w])>min_length:
                inputCentroids.append((x+w/2, y+h/2))
        return inputCentroids

########################## Track resting objects ####################
    def track_resting_obj(self,step_to_force_analyze,debug_mode):
        """ 
        Track resting objects (forward and backward) and update moving_obj
        input : self.resting_objects_forw
        output : self.mosquito_rest_tracks_forw , self.mosquito_moving_objects_forw
        """

        if hasattr(self,'mosquito_rest_tracks_forw')==0 or step_to_force_analyze<=2:

            resting_objects = copy.deepcopy(self.resting_objects)
            for direction in ["forward"]:#["forward","backward"]:
                print("Starting tracking resting objects "+direction)

                # Load parameters
                min_duration_resting_traj = self.settings["track_resting"]["min_duration_resting_traj"]
                dist_moving_obj = self.settings["seg_moving"]["dist_moving_obj"]
                maxDisappeared = self.settings["track_resting"]["maxDisappeared"]

                # Load variables
                
                if direction == "forward":
                    moving_objects = self.moving_objects_forw
                else:
                    resting_objects.reverse()
                    moving_objects = self.moving_objects_back

                # Initialize new data structure for tracking
                objs_resting_tracked = resting_tracker(self.settings["track_resting"])
                mosquito_tracks = mosquito_obj_tracker("resting")


                # Start tracking loop
                frame_idx = 0
                while frame_idx < len(resting_objects):

                    if frame_idx%1000 == 0:
                        if debug_mode:
                            progress_bar(frame_idx, len(resting_objects), bar_length=20)
                        
                    centroids_still = resting_objects[frame_idx]

                    ########## Track resting objects
                    objects_resting,new_IDs,lost_IDs,lost_objects = objs_resting_tracked.update(centroids_still)
                    
                    ########### Initialize new IDs
                    if len(new_IDs)>0: 
                        for new_ID in new_IDs:
                            mosquito_tracks.add_new_track(new_ID,[],[],np.nan,np.nan,np.nan,np.nan)
                            #add_new_track(self,track_ID,new_coordinate,new_time_stamp,new_start_boot,new_end_boot,new_start_type,new_end_type)
                    ######## Save resting object tracked #######
                    if len(objects_resting)>0:
                        for (objectID, centroid) in objects_resting.items():
                            mosquito_tracks.add_points_to_track(objectID,centroid,frame_idx) # Add point to trajectory      
                            #add_points_to_track(self,track_ID,coord_to_add,time_stamp_to_add)       
 
                    ###### Keep or remove finished IDs
                    if len(lost_IDs)>0:
                        for lost_ID in lost_IDs:
                            if len(mosquito_tracks.objects[lost_ID]["time_stamp"]) < maxDisappeared + min_duration_resting_traj: # If the object was there for less that 2 seconds
                                
                                #Save only the coord before dissapearing to moving objects
                                for i in np.arange(len(mosquito_tracks.objects[lost_ID]["time_stamp"])-maxDisappeared): 
                                    t = mosquito_tracks.objects[lost_ID]["time_stamp"][i]
                                    obj_to_add = mosquito_tracks.objects[lost_ID]["coordinates"][i]
                                    if len(moving_objects[t])>0:
                                        D = dist.cdist(np.array(obj_to_add).reshape(1,-1), moving_objects[t])
                                        if D[0].min(axis=0) > dist_moving_obj :
                                                moving_objects[t].append(obj_to_add)
                                    else:
                                        moving_objects[t].append(obj_to_add)

                                # Delete the track
                                mosquito_tracks.remove_track(lost_ID)
                            else:
                                mosquito_tracks.remove_points_from_track(lost_ID,[-maxDisappeared,-1]) # Trim the traj with the dissapeared tail

                    frame_idx += 1

                if direction == "forward":
                    self.mosquito_rest_tracks_forw = mosquito_tracks
                    self.moving_objects = moving_objects
                else:
                    self.mosquito_rest_tracks_back = mosquito_tracks
                    self.moving_objects_back = moving_objects
                # Save

                if debug_mode:
                    with open(self.video_object_path, 'wb') as f:
                        pickle.dump(self, f)
                print("Finished tracking resting objects "+direction)
        else:
            print("Tracking resting objects already done")

########################## Track moving objects ####################
    def track_moving_obj(self,step_to_force_analyze,debug_mode):
        """ 
        Track moving objects (forward and backward) 
        input : self.moving_objects_forw
        output : self.mosquito_mov_tracks_forw
        """

        if hasattr(self,'mosquito_mov_tracks_forw')==0 or step_to_force_analyze<=2:
            for direction in ["forward"]:#["forward","backward"]:

                print("Starting tracking moving objects "+direction)

                # Load parameters
                min_duration_moving_traj = self.settings["track_moving"]["min_duration_moving_traj"]
                maxDisappeared = self.settings["track_moving"]["maxDisappeared"]

                # Load variables
                if direction == "forward":
                    moving_objects = self.moving_objects_forw
                else:
                    moving_objects = self.moving_objects_back

                # Initialize new data structure
                objs_moving_tracked = moving_tracker()
                mosquito_tracks = mosquito_obj_tracker("moving")

                ### Start tracking
                frame_idx = 0
                while frame_idx < len(moving_objects):

                    # Run the tracker
                    centroids_moving = moving_objects[frame_idx]
                    objects_moving,new_IDs,lost_IDs,lost_objects = objs_moving_tracked.update(centroids_moving)
                    
                    if len(new_IDs)>0: ########### Start tracking new disappeared IDs               
                        for new_ID in new_IDs:
                            mosquito_tracks.add_new_track(new_ID,[],[],np.nan,np.nan,np.nan,np.nan)
                            #add_new_track(self,track_ID,new_coordinate,new_time_stamp,new_start_boot,new_end_boot,new_start_type,new_end_type)


                    if len(objects_moving)>0:
                        for (objectID, centroid) in objects_moving.items():            
                            mosquito_tracks.add_points_to_track(objectID,centroid,frame_idx) # Add point to trajectory        
                
                    if len(lost_IDs)>0:
                        for lost_ID in lost_IDs:
                            if len(mosquito_tracks.objects[lost_ID]["time_stamp"]) < maxDisappeared + min_duration_moving_traj: # If the object was there for less that 2 seconds
                                # Delete the track
                                mosquito_tracks.remove_track(lost_ID)
                            else:
                                mosquito_tracks.remove_points_from_track(lost_ID,[-maxDisappeared,-1]) # Trim the traj with the dissapeared tail
                                
                    frame_idx += 1

                if direction == "forward":
                    self.mosquito_mov_tracks_forw = mosquito_tracks
                else:
                    self.mosquito_mov_tracks_back = mosquito_tracks
                # Save 
                if debug_mode:   
                    with open(self.video_object_path, 'wb') as f:
                        pickle.dump(self, f)
                print("Finished tracking moving objects "+direction)

        else:
            print("Tracking moving object already done")

########################## Assemble resting and moving tracks (find taking off and landing match) ####################
    def assemble_resting_and_moving_tracks(self,step_to_force_analyze,time_window_search,max_distance_search,debug_mode):
        """ 
        Track moving objects (forward and backward) 
        input : self.mosquito_rest_tracks_forw self.mosquito_mov_tracks_forw
        output : self.mosquito_tracks_forw
        """

        # #        To remove after        ##
        # settings_file = self.folder_analysis+"buzzwatch_track_settings.yml"
        # if os.path.isfile(settings_file):
        #     with open(settings_file, 'r') as file:
        #         settings = yaml.safe_load(file)
        #         self.settings = settings
        # ##                              ##

        if hasattr(self,'mosquito_tracks_forw')==0 or step_to_force_analyze<=4:
            for direction in ["forward"]:#["forward","backward"]:
                print("Start assembling resting and moving tracks "+direction)

                # Load variables
                if direction == "forward":
                    resting_object_tracks = self.mosquito_rest_tracks_forw
                    moving_object_tracks = self.mosquito_mov_tracks_forw
                else:
                    resting_object_tracks = self.mosquito_rest_tracks_back
                    moving_object_tracks = self.mosquito_mov_tracks_back


                # Get useable indexes of all tracks
                list_of_ID_still = list(resting_object_tracks.objects.keys())
                list_of_ID_moving = list(moving_object_tracks.objects.keys())

                # Initialize distance and time matrices
                landing_matrix_dist = np.full((len(list_of_ID_still),len(list_of_ID_moving)),10000.0)
                takeoff_matrix_dist = np.full((len(list_of_ID_still),len(list_of_ID_moving)),10000.0)
                landing_matrix_time = np.full((len(list_of_ID_still),len(list_of_ID_moving)),0)
                takeoff_matrix_time = np.full((len(list_of_ID_still),len(list_of_ID_moving)),0)

                # Loop over the resting objects
                for i,rest_id in enumerate(list_of_ID_still):
                    if debug_mode:
                        progress_bar(i, len(list_of_ID_still), bar_length=20)

                    # Take-off
                    # If already matched
                    if np.isnan(resting_object_tracks.objects[rest_id]["end_boot"])==0: 
                        t_takeoff_tar = np.nan
                    else:
                        # If resting tracks goes to the end of the video
                        if resting_object_tracks.objects[rest_id]["time_stamp"][-1] == self.total_number_frames-1: # Goes to the end of the video
                            resting_object_tracks.objects[rest_id]["end_boot"] = -1 # Mark as goes to the end
                            t_takeoff_tar = np.nan
                        else:
                            # time of the end of rest_track
                            t_takeoff_tar = resting_object_tracks.objects[rest_id]["time_stamp"][-1]

                    # Landing    
                    # If already matched
                    if np.isnan(resting_object_tracks.objects[rest_id]["start_boot"])==0:
                        t_landing_tar = np.nan
                        #print("resting id "+str(rest_id)+" was already resting at t=0")
                    else:
                        # rest_tracks starts at the start of the video 
                        if resting_object_tracks.objects[rest_id]["time_stamp"][0] < 10:
                            t_landing_tar = np.nan
                            # Mark as done
                            resting_object_tracks.objects[rest_id]["start_boot"] = -1
                        else:
                            # time of the start of rest_track
                            t_landing_tar = resting_object_tracks.objects[rest_id]["time_stamp"][0]

                    # Loop over the moving objects
                    for j,mov_id in enumerate(list_of_ID_moving):

                        # Look for landing match : Look over all time points centered on t_landing_tar
                        # mov_track_end is free
                        if np.isnan(t_landing_tar) == 0 and np.isnan(moving_object_tracks.objects[mov_id]["end_boot"]) == 1 :
                            # If the rest_track and mov_track are well positioned for landing
                            if len(moving_object_tracks.objects[mov_id]["time_stamp"])>0:
                                if moving_object_tracks.objects[mov_id]["time_stamp"][0] < resting_object_tracks.objects[rest_id]["time_stamp"][0]:
                                    already_found = 0
                                    for t_landing in np.arange(t_landing_tar-int(time_window_search/2), t_landing_tar+int(time_window_search/2), 1):
                                        if t_landing in moving_object_tracks.objects[mov_id]["time_stamp"]:
                                            # The potential mov_track end  is before the end of the rest_tracl
                                            if t_landing < resting_object_tracks.objects[rest_id]["time_stamp"][-1]:
                                                t_landing_mov = moving_object_tracks.objects[mov_id]["time_stamp"].index(t_landing) # Relative index
                                                D_landing = dist.cdist(np.array(resting_object_tracks.objects[rest_id]["coordinates"][0]).reshape(1,-1), np.array(moving_object_tracks.objects[mov_id]["coordinates"][t_landing_mov]).reshape(1,-1))
                                                if already_found == 0:
                                                    D_landing_min = D_landing
                                                    t_landing_min = t_landing_mov
                                                    already_found = 1
                                                else:
                                                    if D_landing < D_landing_min:
                                                        D_landing_min = D_landing
                                                        t_landing_min = t_landing_mov

                                    if already_found == 1:
                                        landing_matrix_dist[i][j] = D_landing_min
                                        landing_matrix_time[i][j] = t_landing_min
                      
                        # Look for taking-off match : Look over all time points centered on t_takeoff_tar

                        if np.isnan(t_takeoff_tar) == 0 and np.isnan(moving_object_tracks.objects[mov_id]["start_boot"]) == 1:
                            already_found = 0
                            # If the rest_track and mov_track are well positioned for take-off
                            if len(moving_object_tracks.objects[mov_id]["time_stamp"])>0:
                                if moving_object_tracks.objects[mov_id]["time_stamp"][-1] > resting_object_tracks.objects[rest_id]["time_stamp"][-1]:
                                    for t_takeoff in np.arange(t_takeoff_tar-int(time_window_search/2), t_takeoff_tar+int(time_window_search/2), 1):
                                        if t_takeoff in moving_object_tracks.objects[mov_id]["time_stamp"]:
                                            # The potential mov_track end  is after the start of the rest_tracl
                                            if t_takeoff > resting_object_tracks.objects[rest_id]["time_stamp"][0]:
                                                t_takeoff_mov = moving_object_tracks.objects[mov_id]["time_stamp"].index(t_takeoff) # Relative index
                                                D_takeoff = dist.cdist(np.array(resting_object_tracks.objects[rest_id]["coordinates"][-1]).reshape(1,-1), np.array(moving_object_tracks.objects[mov_id]["coordinates"][t_takeoff_mov]).reshape(1,-1))
                                                if already_found == 0:
                                                    D_takeoff_min = D_takeoff
                                                    t_takeoff_min = t_takeoff_mov
                                                    already_found = 1 
                                                else:
                                                    if D_takeoff < D_takeoff_min:
                                                        D_takeoff_min = D_takeoff
                                                        t_takeoff_min = t_takeoff_mov

                                    if already_found == 1:
                                        takeoff_matrix_dist[i][j] = D_takeoff_min
                                        takeoff_matrix_time[i][j] = t_takeoff_min



                pairing_takeoff = self.find_optimal_matching_resting_moving_from_distance(takeoff_matrix_dist,takeoff_matrix_time)
                pairing_landing = self.find_optimal_matching_resting_moving_from_distance(landing_matrix_dist,landing_matrix_time)

                # Updating the tracks with taking-off
                for (i,j,dist_c,time_c) in pairing_takeoff:
                    if dist_c < max_distance_search:
                        rest_id = list_of_ID_still[i]
                        mov_id = list_of_ID_moving[j]

                        # Mark the tracks as attached
                        resting_object_tracks.objects[rest_id]["end_boot"] = mov_id
                        resting_object_tracks.objects[rest_id]["end_type"] = "moving"
                        moving_object_tracks.objects[mov_id]["start_boot"] = rest_id
                        moving_object_tracks.objects[mov_id]["start_type"] = "resting"

                        # Find the right
                        t_rest = resting_object_tracks.objects[rest_id]["time_stamp"][-1]
                        time_c = 0
                        if moving_object_tracks.objects[mov_id]["time_stamp"][time_c] < t_rest:               
                            while moving_object_tracks.objects[mov_id]["time_stamp"][time_c] < t_rest and time_c < len(moving_object_tracks.objects[mov_id]["time_stamp"])-1:
                                time_c += 1
                                #print(time_c)

                        # Create new moving track from the mov_track before taking-off
                        new_coord = moving_object_tracks.objects[mov_id]["coordinates"][0:time_c]
                        new_time_stamp = moving_object_tracks.objects[mov_id]["time_stamp"][0:time_c]
                        new_id = max(list(moving_object_tracks.objects.keys()))+1 # Create non existing index
                        new_start_boot = np.nan #moving_object_tracks.objects[mov_id]["start_boot"]
                        new_start_type = np.nan #moving_object_tracks.objects[mov_id]["start_type"]
                        new_end_boot = np.nan
                        new_end_type = np.nan
                        #add_new_track(self,track_ID,new_coordinate,new_time_stamp,new_type,new_start_boot,new_end_boot):
                        moving_object_tracks.add_new_track(new_id,new_coord,new_time_stamp,new_start_boot,new_end_boot,new_start_type,new_end_type)

                        # Cut-off the mov_track before taking-off
                        moving_object_tracks.remove_points_from_track(mov_id,[0,time_c])

                # Updating the tracks with landing
                for (i,j,dist_c,time_c) in pairing_landing:
                    if dist_c < max_distance_search:
                        rest_id = list_of_ID_still[i]
                        mov_id = list_of_ID_moving[j]

                        # Mark the tracks as attached
                        resting_object_tracks.objects[rest_id]["start_boot"] = mov_id
                        resting_object_tracks.objects[rest_id]["start_type"] = "moving"
                        moving_object_tracks.objects[mov_id]["end_boot"] = rest_id
                        moving_object_tracks.objects[mov_id]["end_type"] = "resting"

                        # Find the right
                        t_rest = resting_object_tracks.objects[rest_id]["time_stamp"][0]

                        time_c = len(moving_object_tracks.objects[mov_id]["time_stamp"])-1
                        if moving_object_tracks.objects[mov_id]["time_stamp"][time_c] > t_rest:               
                            while moving_object_tracks.objects[mov_id]["time_stamp"][time_c] > t_rest and time_c >0:
                                time_c += -1
                                

                        # Create new moving track from the mov_track before taking-off
                        new_coord = moving_object_tracks.objects[mov_id]["coordinates"][time_c:]
                        new_time_stamp = moving_object_tracks.objects[mov_id]["time_stamp"][time_c:]
                        new_id = max(list(moving_object_tracks.objects.keys()))+1 # Create non existing index
                        new_start_boot = np.nan
                        new_start_type = np.nan
                        new_end_boot = np.nan # moving_object_tracks.objects[mov_id]["end_boot"]
                        new_end_type = np.nan #moving_object_tracks.objects[mov_id]["end_type"]
                        #add_new_track(self,track_ID,new_coordinate,new_time_stamp,new_type,new_start_boot,new_end_boot):
                        moving_object_tracks.add_new_track(new_id,new_coord,new_time_stamp,new_start_boot,new_end_boot,new_start_type,new_end_type)

                        # Cut-off the mov_track after landing
                        moving_object_tracks.remove_points_from_track(mov_id,[time_c,-1])

                # Save the data 
                if direction == "forward":
                    self.mosquito_rest_tracks_forw = resting_object_tracks
                    self.mosquito_mov_tracks_forw = moving_object_tracks
                else:
                    self.mosquito_rest_tracks_back = resting_object_tracks
                    self.mosquito_mov_tracks_back = moving_object_tracks 
                print("Finished assembling resting-moving "+direction)

            #with open(self.video_object_path, 'wb') as f:
            #    pickle.dump(self, f)

########################## Sort matching indexes resting and moving ####################
    def find_optimal_matching_resting_moving_from_distance(self,distance_matrix,time_matrix):
        """
        Track moving objects (forward and backward) 
        input : distance matrix
        output : pairs of resting and moving id to match
        """
        rows = distance_matrix.min(axis=1).argsort()
        cols = distance_matrix.argmin(axis=1)[rows]

        # in order to determine if we need to update, register,
        # or deregister an object we need to keep track of which
        # of the rows and column indexes we have already examined
        usedRows = set()
        usedCols = set()

        rest_id = []
        mov_id = []
        # loop over the combination of the (row, column) index
        # tuples
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or
            # column value before, ignore it
            # val
            if row in usedRows or col in usedCols:
                continue

            usedRows.add(row)
            usedCols.add(col)
            rest_id.append(row)
            mov_id.append(col)

        # compute both the row and column index we have NOT yet
        # examined
        unusedRows = set(range(0, distance_matrix.shape[0])).difference(usedRows)
        unusedCols = set(range(0, distance_matrix.shape[1])).difference(usedCols)

        # return the indexs matched, with distance and time_diff
        return [[rest_id[f],mov_id[f],distance_matrix[rest_id[f]][mov_id[f]],time_matrix[rest_id[f]][mov_id[f]]] for f in np.arange(len(rest_id))]

########################## Assemble together the unmatched moving trajectories ##############
    def assemble_unmatched_moving_tracks(self,step_to_force_analyze,time_window_search,max_distance_search,debug_mode):
        """
        Track moving objects (forward and backward) 
        input : distance matrix
        output : pairs of resting and moving id to match
        """
         #        To remove after        ##
        # settings_file = self.folder_analysis+"buzzwatch_track_settings.yml"
        # if os.path.isfile(settings_file):
        #     with open(settings_file, 'r') as file:
        #         settings = yaml.safe_load(file)
        #         self.settings = settings
        ##                              ##
        if hasattr(self,'mosquito_tracks_forw')==0 or step_to_force_analyze<=4:
            for direction in ["forward"]:#["forward","backward"]:
                print("Start assembling umatched moving tracks together "+direction)
                

                # Load variables
                if direction == "forward":
                    moving_object_tracks = self.mosquito_mov_tracks_forw
                else:
                    moving_object_tracks = self.mosquito_mov_tracks_back

                list_of_ID_moving = list(moving_object_tracks.objects.keys())

                # 
                landing_matrix_dist = np.full((len(list_of_ID_moving),len(list_of_ID_moving)),100000.0)
                landing_matrix_time = np.full((len(list_of_ID_moving),len(list_of_ID_moving)),0)

                # Loop over the moving objects
                for i,mov_id_1 in enumerate(list_of_ID_moving):
                    if debug_mode:
                        progress_bar(i, len(list_of_ID_moving), bar_length=20)

                    # If non empty trajectory (### to remove ###)
                    if len(moving_object_tracks.objects[mov_id_1]["time_stamp"])>0:
                        # If track is already matched
                        if np.isnan(moving_object_tracks.objects[mov_id_1]["end_boot"])==0: # If boot it not matched yet.
                            t_end_tar = np.nan
                        else:
                            # If moving tracks goes to the end of the video 
                            if moving_object_tracks.objects[mov_id_1]["time_stamp"][-1] == self.total_number_frames-1:
                                t_end_tar = np.nan
                                # Mark the track as done
                                moving_object_tracks.objects[mov_id_1]["end_boot"] = -1
                            else:
                                # Get the time of the end of the mov_track_1
                                t_end_tar = moving_object_tracks.objects[mov_id_1]["time_stamp"][-1]

                            # Loop over the moving objects (look for mov_track_2)
                            for j,mov_id_2 in enumerate(list_of_ID_moving):
                                # If non empty trajectory (### to remove ###)
                                if len(moving_object_tracks.objects[mov_id_2]["time_stamp"])>0: 
                                    #If same id or already matched
                                    if mov_id_1 == mov_id_2 or np.isnan(moving_object_tracks.objects[mov_id_2]["start_boot"])==0:
                                        t_start_tar = np.nan
                                    else:
                                        # If moving to match tracks starts at the start of the video
                                        if moving_object_tracks.objects[mov_id_2]["time_stamp"][0] < 10:
                                            t_start_tar = np.nan
                                            # Mark the track as done
                                            moving_object_tracks.objects[mov_id_2]["start_boot"] = -1
                                        else:
                                             # Get the time of the start of the mov_track_2
                                            t_start_tar = moving_object_tracks.objects[mov_id_2]["time_stamp"][0]

                                            # If not too far in time
                                            if np.abs(t_start_tar-t_end_tar) < time_window_search:
                                                # If end of mov_track_2 is after and of mov_track_1
                                                if moving_object_tracks.objects[mov_id_2]["time_stamp"][-1]-t_end_tar>0:
                                                    # If start of mov_track_1 is after and of mov_track_1
                                                    if moving_object_tracks.objects[mov_id_1]["time_stamp"][0] < moving_object_tracks.objects[mov_id_2]["time_stamp"][0]:
                                                        # If start of mov_track_1 is after and of mov_track_1
                                                        if moving_object_tracks.objects[mov_id_1]["time_stamp"][-1] < moving_object_tracks.objects[mov_id_2]["time_stamp"][-1]:
                                                            # If start of mov_track_1 is before end of mov_track_2
                                                            if t_start_tar < t_end_tar:
                                                                # Find the delay in term of mov_track_2 index 
                                                                t_diff = 0
                                                                while moving_object_tracks.objects[mov_id_2]["time_stamp"][t_diff] < t_end_tar:
                                                                    t_diff += 1
                                                                # Save the time delay
                                                                landing_matrix_time[i][j] = t_diff # Shift in the next traj
                                                            else:
                                                                # No time delay
                                                                landing_matrix_time[i][j] = 0
                                                            # Save the euclidian distance end of mov_track_1 and start of mov_track_2
                                                            D_landing = dist.cdist(np.array(moving_object_tracks.objects[mov_id_1]["coordinates"][-1]).reshape(1,-1), np.array(moving_object_tracks.objects[mov_id_2]["coordinates"][0]).reshape(1,-1))
                                                            landing_matrix_dist[i][j] = D_landing

                # Sort the matches based of the distance
                pairing_takeoff = self.find_optimal_matching_resting_moving_from_distance(landing_matrix_dist,landing_matrix_time)

                # Updating the tracks with taking-off
                for (i,j,dist_c,time_c) in pairing_takeoff:
                    if dist_c < max_distance_search:
                        mov_id_1 = list_of_ID_moving[i]
                        mov_id_2 = list_of_ID_moving[j]

                        # Mark the end of mov_track_1 and start of mov_track_2 as attached
                        moving_object_tracks.objects[mov_id_1]["end_boot"] = mov_id_2
                        moving_object_tracks.objects[mov_id_1]["end_type"] = "moving"

                        moving_object_tracks.objects[mov_id_2]["start_boot"] = mov_id_1
                        moving_object_tracks.objects[mov_id_2]["start_type"] = "moving"
                        # Cut-off the mov_track_2 that overlaps with mov_track_1
                        if time_c > 0:
                            moving_object_tracks.remove_points_from_track(mov_id_2,[0,time_c])

            # Display the results
            # Save results 
                if direction == "forward":
                    self.mosquito_mov_tracks_forw = moving_object_tracks
                else:
                    self.mosquito_mov_tracks_back = moving_object_tracks 
                print("Finished assembling moving-moving "+direction)

            #with open(self.video_object_path, 'wb') as f:
            #    pickle.dump(self, f)

########################## Display video with tracking ##############
    def display_video_with_tracking(self,direction,starting_frame,time_btw_frames):

        # Load variables
        if direction == "forward":
            resting_object_tracks = self.mosquito_rest_tracks_forw
            moving_object_tracks = self.mosquito_mov_tracks_forw
            matched_ids = self.matched_ids_forw
        else:
            resting_object_tracks = self.mosquito_rest_tracks_back
            moving_object_tracks = self.mosquito_mov_tracks_back
            matched_ids = self.matched_ids_back

        # Initialiaze video
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        n_frame = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
        f_i = starting_frame

        list_of_ID_still = list(resting_object_tracks.objects.keys())
        list_of_ID_moving = list(moving_object_tracks.objects.keys())

        while True:
            suc,frame = cap.read()
            time.sleep(time_btw_frames)

            if suc == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = frame.copy()

                if f_i > 1:
                    if direction =="forward":
                        frame_idx = f_i
                    else:
                        frame_idx = n_frame-f_i


                    # Show resting tracks
                    n_rest = 0
                    for k,id in enumerate(list_of_ID_still):
                        if frame_idx in resting_object_tracks.objects[id]["time_stamp"]:
                            t_frame = resting_object_tracks.objects[id]["time_stamp"].index(frame_idx)
                            if hasattr(self,"matched_ids_forw"):
                                #text = "i {} s {} e {}".format(id,resting_object_tracks.objects[id]["start_boot"],resting_object_tracks.objects[id]["end_boot"])
                                text = "id {}".format(matched_ids[1][k])
                            else:
                                text = "i {} s {} e {}".format(id,resting_object_tracks.objects[id]["start_boot"],resting_object_tracks.objects[id]["end_boot"])
                            #if matched_ids[1][k] == 6:
                                #print("ID" + str(id))
                                #print(resting_object_tracks.objects[id]["start_boot"])
                                #print(resting_object_tracks.objects[id]["end_boot"])
                            centroid = resting_object_tracks.objects[id]["coordinates"][t_frame]
                            cv2.putText(img, text, (int(centroid[0])-20, int(centroid[1])-20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 2, (0, 0, 255), 1)
                            n_rest += 1
                    # Show moving tracks
                    n_mov = 0
                    for k,id in enumerate(list_of_ID_moving):
                        if frame_idx in moving_object_tracks.objects[id]["time_stamp"]:
                            t_frame = moving_object_tracks.objects[id]["time_stamp"].index(frame_idx)
                            if hasattr(self,"matched_ids_forw"):
                                #text = "i {} s {} e {}".format(id,moving_object_tracks.objects[id]["start_boot"],moving_object_tracks.objects[id]["end_boot"])
                                text = "id {}".format(matched_ids[1][k+len(list_of_ID_still)])
                            else:
                                text = "ID {} take {} land {}".format(id,moving_object_tracks.objects[id]["start_boot"],moving_object_tracks.objects[id]["end_boot"])
                            centroid = moving_object_tracks.objects[id]["coordinates"][t_frame]
                            cv2.putText(img, text, (int(centroid[0])-20, int(centroid[1])-20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 2, (0, 255, 0), 1)
                            n_mov += 1
                    #os.system('clear')
                    print(str(int(n_rest))+" resting and "+str(int(n_mov))+" moving", end="\r")
                cv2.imshow("frame",img)
                f_i += 1

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if f_i>n_frame-2:
                    break
        cap.release()
        cv2.destroyAllWindows()


########################## Clean tracks and return the number of umatched tracks ##############
    def clean_tracks(self,step_to_force_analyze,time_window_search,max_distance_search,debug_mode):
        #print("Cleaning tracks", end="\r")

        if step_to_force_analyze<=4:
            for direction in ["forward"]:#["forward","backward"]:
                # Load variables
                if direction == "forward":
                    resting_object_tracks = self.mosquito_rest_tracks_forw
                    moving_object_tracks = self.mosquito_mov_tracks_forw
                else:
                    resting_object_tracks = self.mosquito_rest_tracks_back
                    moving_object_tracks = self.mosquito_mov_tracks_back

                list_of_ID_resting = list(resting_object_tracks.objects.keys())
                list_of_ID_moving = list(moving_object_tracks.objects.keys())

                min_duration_moving_traj = self.settings["track_moving"]["min_duration_moving_traj"]

                # Check moving objects
                for j,mov_id in enumerate(list_of_ID_moving):
                    #progress_bar(j, len(list_of_ID_moving), bar_length=20)

                    # If not already attached to another tracks
                    if np.isnan(moving_object_tracks.objects[mov_id]["start_boot"]) == 1 and np.isnan(moving_object_tracks.objects[mov_id]["end_boot"]) == 1:
                        if len(moving_object_tracks.objects[mov_id]["time_stamp"]) < time_window_search and self.is_flying(moving_object_tracks.objects[mov_id])==0:
                            moving_object_tracks.remove_track(mov_id)
                        else:
                            if len(moving_object_tracks.objects[mov_id]["time_stamp"]) < min_duration_moving_traj:
                                moving_object_tracks.remove_track(mov_id)


                # Count the percentage of matched tracks
                start_mov = [np.isnan(moving_object_tracks.objects[mov_id]["start_boot"]) for mov_id in moving_object_tracks.objects.keys()]
                perc_matched_mov_start = np.round((1-np.mean(start_mov))*100,3)
                end_mov = [np.isnan(moving_object_tracks.objects[mov_id]["end_boot"]) for mov_id in moving_object_tracks.objects.keys()]
                perc_matched_mov_end = np.round((1-np.mean(end_mov))*100,3)
                start_rest = [np.isnan(resting_object_tracks.objects[mov_id]["start_boot"]) for mov_id in resting_object_tracks.objects.keys()]
                perc_matched_rest_start = np.round((1-np.mean(start_rest))*100,3)
                end_rest = [np.isnan(resting_object_tracks.objects[mov_id]["end_boot"]) for mov_id in resting_object_tracks.objects.keys()]
                perc_matched_rest_end = np.round((1-np.mean(end_rest))*100,3)

                print("Finished leaning tracks")
                print("rest_start_"+str(perc_matched_rest_start)+"%")
                print("rest_end_"+str(perc_matched_rest_end)+"%")
                print("mov_start_"+str(perc_matched_mov_start)+"%")
                print("mov_end_"+str(perc_matched_mov_end)+"%")

                # Save results 
                if direction == "forward":
                    self.mosquito_mov_tracks_forw = moving_object_tracks
                else:
                    self.mosquito_mov_tracks_back = moving_object_tracks 
        if debug_mode:
            with open(self.video_object_path, 'wb') as f:
                pickle.dump(self, f)

            
# ########################## Assemble id from matched tracks ###################
    def assemble_tracks_ids(self,step_to_force_analyze,debug_mode):
         
        if step_to_force_analyze<=5:
            for direction in ["forward"]:#["forward","backward"]:
                print("Assembling IDs")
                # Load variables
                if direction == "forward":
                    resting_object_tracks = self.mosquito_rest_tracks_forw
                    moving_object_tracks = self.mosquito_mov_tracks_forw
                else:
                    resting_object_tracks = self.mosquito_rest_tracks_back
                    moving_object_tracks = self.mosquito_mov_tracks_back

                list_of_ID_resting = list(resting_object_tracks.objects.keys())
                list_of_ID_moving = list(moving_object_tracks.objects.keys())

                matching_matrix = np.zeros((len(list_of_ID_resting)+len(list_of_ID_moving),len(list_of_ID_resting)+len(list_of_ID_moving)))

                # Loop through resting objects
                for i,rest_id in enumerate(list_of_ID_resting):
                    #Check start boot
                    if np.isnan(resting_object_tracks.objects[rest_id]["start_boot"])==0:
                        if resting_object_tracks.objects[rest_id]["start_type"] == "moving":
                            
                            j = list_of_ID_moving.index(resting_object_tracks.objects[rest_id]["start_boot"])+len(list_of_ID_resting)
                            matching_matrix[i][j] = -1
                        elif resting_object_tracks.objects[rest_id]["start_type"] == "resting":
                            
                            j = list_of_ID_resting.index(resting_object_tracks.objects[rest_id]["start_boot"])
                            matching_matrix[i][j] = -1

                    #Check end boot
                    if np.isnan(resting_object_tracks.objects[rest_id]["end_boot"])==0:
                        if resting_object_tracks.objects[rest_id]["end_type"] == "moving":
                            j = list_of_ID_moving.index(resting_object_tracks.objects[rest_id]["end_boot"])+len(list_of_ID_resting)
                            matching_matrix[i][j] = 1
                        elif resting_object_tracks.objects[rest_id]["end_type"] == "resting":
                            
                            j = list_of_ID_resting.index(resting_object_tracks.objects[rest_id]["end_boot"])
                            matching_matrix[i][j] = 1

                # Loop through moving objects
                for i,mov_id in enumerate(list_of_ID_moving):
                    #Check start boot
                    if np.isnan(moving_object_tracks.objects[mov_id]["start_boot"])==0:
                        if moving_object_tracks.objects[mov_id]["start_type"] == "moving":
                            j = list_of_ID_moving.index(moving_object_tracks.objects[mov_id]["start_boot"])+len(list_of_ID_resting)
                            #print(j)
                            matching_matrix[i+len(list_of_ID_resting)][j] = -1
                        elif moving_object_tracks.objects[mov_id]["start_type"] == "resting":
                            j = list_of_ID_resting.index(moving_object_tracks.objects[mov_id]["start_boot"])
                            matching_matrix[i+len(list_of_ID_resting)][j] = -1

                    #Check end boot
                    if np.isnan(moving_object_tracks.objects[mov_id]["end_boot"])==0:
                        if moving_object_tracks.objects[mov_id]["end_type"] == "moving":
                            j = list_of_ID_moving.index(moving_object_tracks.objects[mov_id]["end_boot"])+len(list_of_ID_resting)
                            matching_matrix[i+len(list_of_ID_resting)][j] = 1
                        elif moving_object_tracks.objects[mov_id]["end_type"] == "resting":
                            j = list_of_ID_resting.index(moving_object_tracks.objects[mov_id]["end_boot"])
                            matching_matrix[i+len(list_of_ID_resting)][j] = 1

                matching_matrix_graph = np.where((matching_matrix==-1) | (matching_matrix==1),1,0)
                #print(matching_matrix_graph)
                
                matched_ids = graph.connected_components(matching_matrix_graph)

                print("Total number of tracks: "+str(matched_ids[0]))
                if direction == "forward":
                    self.matched_ids_forw = matched_ids
                else:
                    self.matched_ids_back = matched_ids
            if debug_mode:
                with open(self.video_object_path, 'wb') as f:
                    pickle.dump(self, f)

# ########################## Compute activity and trajectories from the video. ###################
    def extract_complete_trajectories_from_video(self,debug_mode):
        for direction in ["forward"]:#["forward","backward"]:
            print("Finalizing trajectories")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Load variables
                if direction == "forward":
                    resting_object_tracks = self.mosquito_rest_tracks_forw
                    moving_object_tracks = self.mosquito_mov_tracks_forw
                    matched_ids = self.matched_ids_forw
                else:
                    resting_object_tracks = self.mosquito_rest_tracks_back
                    moving_object_tracks = self.mosquito_mov_tracks_back
                    matched_ids = self.matched_ids_back

                list_of_ID_still = list(resting_object_tracks.objects.keys())
                list_of_ID_moving = list(moving_object_tracks.objects.keys())

                # Initialiaze data structure for all trajectories of the video.
                mosquito_tracks = mosquito_obj_tracker("mixed")

                # Get the time_stamp from the video in the right format.
                t_start = self.get_datetime_from_file_name()
                print(t_start)
                mosquito_tracks.time_stamp = [t_start+timedelta(milliseconds=int(40*i)) for i in np.arange(self.total_number_frames)]
                
                
                # Loop over all the Ids. 
                for id in np.arange(int(matched_ids[0])):
                    if debug_mode:
                        progress_bar(id, int(matched_ids[0]), bar_length=20)


                    # Initialize empty NaN track of the lenght of the video.
                    trajectory = [(np.nan,np.nan) for t in np.arange(self.total_number_frames)]
                    state = [np.nan for t in np.arange(self.total_number_frames)]
                    
                    # Loop over all the resting and moving ids of the track
                    list_ids_tracks = np.where(matched_ids[1] == id)
                    #print(list_ids_tracks[0])
                    for sub_id in list_ids_tracks[0]:
                        #If resting track
                        if sub_id < len(list_of_ID_still):
                            for t,frame in enumerate(resting_object_tracks.objects[list_of_ID_still[sub_id]]["time_stamp"]):
                                trajectory[frame] = resting_object_tracks.objects[list_of_ID_still[sub_id]]["coordinates"][t]
                                state[frame] = 0 # Mark as resting object
                        #If moving track
                        else:
                            sub_id = sub_id - len(list_of_ID_still)
                            if self.is_flying(moving_object_tracks.objects[list_of_ID_moving[sub_id]]) == True:
                                for t,frame in enumerate(moving_object_tracks.objects[list_of_ID_moving[sub_id]]["time_stamp"]):
                                    trajectory[frame] = moving_object_tracks.objects[list_of_ID_moving[sub_id]]["coordinates"][t]
                                    state[frame] = 1 # Mark as moving/flying track

                    # Smallest index with nan (beginning of the track)
                    if np.isnan(state).all()==0:
                        # Add a new empty track
                        mosquito_tracks.add_mosquito_track(id)
                        t_start = np.min(np.argwhere(np.isnan(state)==0))
                        t_end = np.max(np.argwhere(np.isnan(state)==0))
                        #print("t_start "+str(t_start)+" t_end"+str(t_end))
                        mosquito_tracks.objects[id]["coordinates"] = trajectory[t_start:t_end]
                        mosquito_tracks.objects[id]["state"] = state[t_start:t_end]
                        mosquito_tracks.objects[id]["start"] = t_start
                        mosquito_tracks.objects[id]["end"] = t_end


                # Save sugar feeding and flight activity
                sugar_all = []
                control_all = []
                fly_all = []
                #print(files_tracking[video_idx])
                for id in mosquito_tracks.objects.keys():

                    # Get main stats
                    sugar = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
                    control = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
                    fly = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]

                    coordinates = mosquito_tracks.objects[id]["coordinates"]
                    fly[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = mosquito_tracks.objects[id]["state"]
                    
                    is_sugar = [self.point_inside_cage(self.sugar_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0  for i,coord in enumerate(coordinates)]
                    is_control = [self.point_inside_cage(self.control_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0  for i,coord in enumerate(coordinates)]

                    sugar[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_sugar
                    control[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_control

                    del sugar[0:5]
                    sugar.pop()

                    del control[0:5]
                    control.pop()

                    del fly[0:5]
                    fly.pop()

                    control_all.append(np.array(control))
                    sugar_all.append(np.array(sugar))
                    fly_all.append(np.array(fly))

                # print(len(np.nansum(control_all,axis=0)))
                # print(len(np.nansum(sugar_all,axis=0)))
                data = {'time': mosquito_tracks.time_stamp[5:-1],
                        'numb_mosquitos_flying': np.nansum(fly_all,axis=0),
                        'numb_mosquitos_sugar': np.nansum(sugar_all,axis=0),
                        'numb_mosquitos_control': np.nansum(control_all,axis=0)}
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                #df.resample('1T', label='right').mean()
                mosquito_tracks.flight_population_activity = df

                #plt.plot_date(mosquito_tracks.flight_population_activity.index,mosquito_tracks.flight_population_activity["numb_mosquitos_flying"],linestyle="-",marker = None)
                #plt.plot_date(mosquito_tracks.flight_population_activity.index,mosquito_tracks.flight_population_activity["numb_mosquitos_control"],linestyle="-",marker = None)
                #plt.plot_date(mosquito_tracks.flight_population_activity.index,mosquito_tracks.flight_population_activity["numb_mosquitos_sugar"],linestyle="-",marker = None)

                #plt.show()

                # Save fligth speed and duration
                create_folder(self.folder_analysis+"/plots_trajectories")
                fig, axes = plt.subplots(3, 3,dpi=200)
                fig.set_figheight(20)
                fig.set_figwidth(20)
                axes  = axes.reshape(-1)

                all_speed = []
                all_start_time = []
                all_duration = []
                nb_plotted = 0
                for k,id in enumerate(mosquito_tracks.objects.keys()):
                    state_v = mosquito_tracks.objects[id]["state"]
                    coord_v = mosquito_tracks.objects[id]["coordinates"]

                    state_v = np.array([1-i for i in state_v])
                    runs = zero_runs(state_v)
                    nb_tracks = len(runs[:,0])

                    x = [a for a,b in  coord_v]
                    y = [b for a,b in  coord_v]

                    if nb_tracks>0:
                        for k in np.arange(nb_tracks):
                            t_i = runs[k,0]
                            t_f = runs[k,1]

                            d_x = np.array(x[t_i+1:t_f]) - np.array(x[t_i:t_f-1])
                            d_y = np.array(y[t_i+1:t_f]) - np.array(y[t_i:t_f-1])

                            d_x_2 = [np.square(z) for z in d_x]
                            d_y_2 = [np.square(z) for z in d_y]

                            dist = [np.sqrt(d_x_2[i]+d_y_2[i]) for i in np.arange(len(d_x))]
                            if t_f-t_i>25*5:
                                #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])
                                all_start_time.append(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]]+timedelta(milliseconds=int(40*t_i)))
                                all_speed.append(np.mean(dist)) # starting time
                                all_duration.append((t_f-t_i)/25) # duration in seconds

                                if nb_plotted < 9 and np.mean(dist)>5:
                                    ax = self.plot_flight_trajectory(x[t_i+1:t_f],y[t_i+1:t_f],axes[nb_plotted])
                                    nb_plotted += 1
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.ioff()
                plt.savefig(self.folder_analysis+"/plots_trajectories/__"+self.video_name+'.png',bbox_inches='tight')
                #print(self.folder_analysis+"plots_trajectories/__"+self.video_name+'.png')
                data = {'time': all_start_time,
                        'average_speed': all_speed,
                        'duration' : all_duration}
                df = pd.DataFrame(data)

                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                mosquito_tracks.flight_trajectories = df



                # Save the mosquito_tracks object in the "final_tracking" dir
                with open(self.folder_final+"/"+direction+"_mosq_tracks_"+self.video_name, 'wb') as f:
                    pickle.dump(mosquito_tracks, f)

                print("Finished finalizing trajectories")



########################## Complete trajectories ################
    def extract_complete_trajectories_from_video_V2(self,debug_mode):
        print("Start Extracting compplete trajectories")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Load variables

            resting_object_tracks = self.mosquito_rest_tracks_forw
            moving_object_tracks = self.mosquito_mov_tracks_forw
            matched_ids = self.matched_ids_forw


            list_of_ID_still = list(resting_object_tracks.objects.keys())
            list_of_ID_moving = list(moving_object_tracks.objects.keys())

            # Initialiaze data structure for all trajectories of the video.
            mosquito_tracks = mosquito_obj_tracker("mixed")

            # Get the time_stamp from the video in the right format.
            t_start = self.get_datetime_from_file_name()
            #print(t_start)
            mosquito_tracks.time_stamp = [t_start+timedelta(milliseconds=int(40*i)) for i in np.arange(self.total_number_frames)]
            
            # Loop over all the Ids. 
            for id in np.arange(int(matched_ids[0])):
                if debug_mode:
                    progress_bar(id, int(matched_ids[0]), bar_length=20)

                # Initialize empty NaN track of the lenght of the video.
                trajectory = [(np.nan,np.nan) for t in np.arange(self.total_number_frames)]
                state = [np.nan for t in np.arange(self.total_number_frames)]
                
                # Loop over all the resting and moving ids of the track
                list_ids_tracks = np.where(matched_ids[1] == id)
                #print(list_ids_tracks[0])
                for sub_id in list_ids_tracks[0]:
                    #If resting track
                    if sub_id < len(list_of_ID_still):
                        for t,frame in enumerate(resting_object_tracks.objects[list_of_ID_still[sub_id]]["time_stamp"]):
                            trajectory[frame] = resting_object_tracks.objects[list_of_ID_still[sub_id]]["coordinates"][t]
                            state[frame] = 0 # Mark as resting object
                    #If moving track
                    else:
                        sub_id = sub_id - len(list_of_ID_still)
                        if self.is_flying(moving_object_tracks.objects[list_of_ID_moving[sub_id]]) == True:
                            for t,frame in enumerate(moving_object_tracks.objects[list_of_ID_moving[sub_id]]["time_stamp"]):
                                trajectory[frame] = moving_object_tracks.objects[list_of_ID_moving[sub_id]]["coordinates"][t]
                                state[frame] = 1 # Mark as moving/flying track

                # Smallest index with nan (beginning of the track)
                if np.isnan(state).all()==0:
                    # Add a new empty track
                    mosquito_tracks.add_mosquito_track(id)
                    t_start = np.min(np.argwhere(np.isnan(state)==0))
                    t_end = np.max(np.argwhere(np.isnan(state)==0))
                    #print("t_start "+str(t_start)+" t_end"+str(t_end))
                    mosquito_tracks.objects[id]["coordinates"] = trajectory[t_start:t_end]
                    mosquito_tracks.objects[id]["state"] = state[t_start:t_end]
                    mosquito_tracks.objects[id]["start"] = t_start
                    mosquito_tracks.objects[id]["end"] = t_end

        self.mosquito_tracks = mosquito_tracks
        print("Completed Extracting compplete trajectories")


############# Extract the poppulation and single mosquito statistics from the mosquito_tracks objects

    def extract_mosquito_population_variables(self):
        print("Start Extracting population variable")
        
        hs_border_points = self.settings["control_border_points"]
        cage_border_points = self.settings["cage_border_points"]
        sugar_border_points = self.settings["sugar_border_points"]
        left_ctrl_border_points = self.settings["square_4_border_points"]
        right_ctrl_border_points = self.settings["square_3_border_points"]

        mosquito_tracks = self.mosquito_tracks

        # Save sugar feeding and flight activity
        sugar_all = []
        hs_all = []
        left_ctrl_all = []
        right_ctrl_all = []

        fly_all = []

        #print(files_tracking[video_idx])
        for id in mosquito_tracks.objects.keys():

            # Get main stats
            sugar = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
            hs = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
            fly = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]

            left_ctrl = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]
            right_ctrl = [np.nan for k in np.arange(len(mosquito_tracks.time_stamp))]

            coordinates = mosquito_tracks.objects[id]["coordinates"]
            fly[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = mosquito_tracks.objects[id]["state"]
            
            #is_sugar = [point_inside_cage(sugar_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0  for i,coord in enumerate(coordinates)]
            #is_control = [point_inside_cage(control_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0  for i,coord in enumerate(coordinates)]
            is_sugar = [self.point_inside_cage(sugar_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0  for i,coord in enumerate(coordinates)]
            is_hs = [self.point_inside_cage(hs_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0 for i,coord in enumerate(coordinates)]

            is_left_ctrl = [self.point_inside_cage(left_ctrl_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0 for i,coord in enumerate(coordinates)]
            is_right_ctrl = [self.point_inside_cage(right_ctrl_border_points,coord) and mosquito_tracks.objects[id]["state"][i]==0 for i,coord in enumerate(coordinates)]

            sugar[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_sugar # Sugar feeder
            hs[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_hs # Host seeking


            left_ctrl[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_left_ctrl 
            right_ctrl[mosquito_tracks.objects[id]["start"]:mosquito_tracks.objects[id]["end"]] = is_right_ctrl 

            del left_ctrl[0:5]
            left_ctrl.pop()

            del right_ctrl[0:5]
            right_ctrl.pop()

            del sugar[0:5]
            sugar.pop()

            del hs[0:5]
            hs.pop()

            del fly[0:5]
            fly.pop()

            hs_all.append(np.array(hs))
            sugar_all.append(np.array(sugar))
            left_ctrl_all.append(np.array(left_ctrl))
            right_ctrl_all.append(np.array(right_ctrl))
            fly_all.append(np.array(fly))


        # print(len(np.nansum(control_all,axis=0)))
        # print(len(np.nansum(sugar_all,axis=0)))
        data = {'time': mosquito_tracks.time_stamp[5:-1],
                'numb_mosquitos_flying': np.nansum(fly_all,axis=0),
                'numb_mosquitos_sugar': np.nansum(sugar_all,axis=0),
                'numb_mosquitos_hs': np.nansum(hs_all,axis=0),
                'numb_mosquitos_left_ctrl': np.nansum(left_ctrl_all,axis=0),
                'numb_mosquitos_right_ctrl': np.nansum(right_ctrl_all,axis=0)
                }
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.resample('1s', label='right').mean()

        mosquito_tracks.population_variables = df

        self.mosquito_tracks = mosquito_tracks
        print("End Extracting population variable")


    def extract_mosquito_individual_variables(self):
        print("Start Extracting individual variable")
        mosquito_tracks = self.mosquito_tracks

        all_start_time = []
        #all_resting_duration = []
        all_flight_duration = []
        all_speed = []
        #print(mosquito_tracks.time_stamp[0])
        for k,id in enumerate(mosquito_tracks.objects.keys()):
            state_v = mosquito_tracks.objects[id]["state"]
            # #coord_v = mosquito_tracks.objects[id]["coordinates"]
            # state_v = np.array([i for i in state_v])
            # runs = zero_runs(state_v)
            # nb_tracks = len(runs[:,0])
            # if nb_tracks>0:
            #     for k in np.arange(nb_tracks):
            #         t_i = runs[k,0]
            #         t_f = runs[k,1]

            #         if t_f-t_i>25*5: # Filter for the not to short resting times
            #             #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])
            #             #all_start_time.append(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]]+timedelta(milliseconds=int(40*t_i)))
            #             all_start_time.append(mosquito_tracks.time_stamp[0])
            #             all_duration.append((t_f-t_i)/25)
            #             #print((t_f-t_i)/25)
            #             #all_duration.append(1)
            #         state_v = mosquito_tracks.objects[id]["state"]

            # # Extract the flight coordinates        
            coord_v = mosquito_tracks.objects[id]["coordinates"]
            state_v = np.array([1-i for i in state_v])
            runs = zero_runs(state_v)
            nb_tracks = len(runs[:,0])
            x = [a for a,b in  coord_v]
            y = [b for a,b in  coord_v]
            if nb_tracks>0:
                for k in np.arange(nb_tracks):
                    t_i = runs[k,0]
                    t_f = runs[k,1]

                    d_x = np.array(x[t_i+1:t_f]) - np.array(x[t_i:t_f-1])
                    d_y = np.array(y[t_i+1:t_f]) - np.array(y[t_i:t_f-1])

                    d_x_2 = [np.square(z) for z in d_x]
                    d_y_2 = [np.square(z) for z in d_y]
                    
                    dist = [np.sqrt(d_x_2[i]+d_y_2[i]) for i in np.arange(len(d_x))]
                    if t_f-t_i>25*2 and np.mean(dist)>5: # Flight trajectory at least 5sec long and not so slow (speed>5)
                        #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])
                        #all_start_time.append(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]]+timedelta(milliseconds=int(40*t_i)))
                        all_start_time.append(mosquito_tracks.time_stamp[0])
                        #all_duration.append((t_f-t_i)/25)
                        all_speed.append(np.mean(dist)) # starting time
                        all_flight_duration.append((t_f-t_i)/25)
                        #print((t_f-t_i)/25)
                        #all_duration.append(1)
        #print(len(all_duration))

        data = {'time': all_start_time,
                #'duration' : all_duration}
                'flight_duration' : all_flight_duration,
                'average_speed' : all_speed}
        df = pd.DataFrame(data)

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        mosquito_tracks.individual_variables = df
        self.mosquito_tracks = mosquito_tracks

        print("End Extracting individual variable")

    def extract_mosquito_resting_variables(self):
        print("Start Extracting resting variable")
        mosquito_tracks = self.mosquito_tracks

        all_start_time = []
        all_resting_duration = []

        #print(mosquito_tracks.time_stamp[0])
        for k,id in enumerate(mosquito_tracks.objects.keys()):
            state_v = mosquito_tracks.objects[id]["state"]
            #coord_v = mosquito_tracks.objects[id]["coordinates"]
            state_v = np.array([i for i in state_v])
            runs = zero_runs(state_v)
            nb_tracks = len(runs[:,0])
            if nb_tracks>0:
                for k in np.arange(nb_tracks):
                    t_i = runs[k,0]
                    t_f = runs[k,1]

                    if t_f-t_i>25*5: # Filter for the not to short resting times
                        #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])
                        #all_start_time.append(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]]+timedelta(milliseconds=int(40*t_i)))
                        all_start_time.append(mosquito_tracks.time_stamp[0])
                        all_resting_duration.append((t_f-t_i)/25)
                        #print((t_f-t_i)/25)
                        #all_duration.append(1)
                   
        data = {'time': all_start_time,
                #'duration' : all_duration}
                'resting_duration' : all_resting_duration}
        df = pd.DataFrame(data)

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        mosquito_tracks.resting_variables = df
        self.mosquito_tracks = mosquito_tracks

        print("End Extracting resting variable")

    def save_tracking_results(self):
        # Save the mosquito_tracks object in the "final_tracking" dir
        self.mosquito_tracks.settings = self.settings
        self.mosquito_tracks.video_name = self.video_name

        with open(f"{self.folder_final}/forward_mosq_tracks_{self.video_name}", 'wb') as f:
            pickle.dump(self.mosquito_tracks, f)

        print("Saving tracking results (mosquito_tracks object)")

# ########################## Plot single coplete tracks (assembled resting+moving) from a video ###################

    #def plot_sample_flight_trajectories_from_video(axes,mosquito_tracks):

    
    def plot_flight_trajectory(self,x,y,ax):
        # Set plot and add background
        back_path = self.video_path = os.path.join(self.folder_analysis ,"images_mortality", self.video_name)+".png"
        im = plt.imread(back_path)
        ax.imshow(im,zorder=1,cmap = "gray")

        x_f = uniform_filter1d(x, size=5)
        y_f = uniform_filter1d(y, size=5)

        c_f = np.arange(len(x_f))
        c_f = np.array([c_f[i]/25 for i in np.arange(len(c_f))])

        points = np.array([x_f, y_f]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(c_f.min(), c_f.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)

        # Set the values used for colormapping
        lc.set_array(c_f)
        lc.set_linewidth(4)
        line = ax.add_collection(lc)

        ax.set_xlim([0 ,im.shape[0]])
        ax.set_ylim([0 ,im.shape[1]])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        return ax

# ########################## Filter out moving_tracks that don't look like flying mosquitos ###################
    def is_flying(self,moving_track):
        coord = moving_track["coordinates"]
        time_s = moving_track["time_stamp"]

        max_vel = 40
        min_vel = 1.0
        min_len = 40

        velocity = []
        for t,centroid in enumerate(coord):
            if t>0:
                velocity.append(dist.cdist(np.array(coord[t-1]).reshape(1,-1),np.array(coord[t]).reshape(1,-1)))

        average_speed = np.mean(velocity)
        unicity = len(np.unique(velocity))
        #print("Av_speed_"+str(average_speed)+" unicity_"+str(unicity))
        #print(average_speed)
        return average_speed < max_vel and average_speed > min_vel and unicity > 10

# ########################## Plot single complete tracks (assembled resting+moving) from a video ###################
    def plot_trajectories_from_video(self,direction):

        plt.rcParams.update({'font.size': 6})
        create_folder(self.folder_analysis+"plots_trajectories")

        # Load the trajectories
        direction = "forward"
        with open(self.folder_final+"/"+direction+"_mosq_tracks_"+self.video_name, 'rb') as f:
            mosquito_tracks = pickle.load(f)  

        
        for id in mosquito_tracks.objects.keys():
            coord =  mosquito_tracks.objects[id]["coordinates"]
            state =  mosquito_tracks.objects[id]["state"]

            if len(coord)>10:
                # Set plot and add background
                fig, axes = plt.subplots(3, 1,dpi=200)
                fig.set_figheight(16)
                fig.set_figwidth(8)


            
                # Plot the resting points
                ax = axes[0]
                plt.tight_layout()
                im = plt.imread(self.background_path)
                ax.imshow(im,zorder=1,cmap = "gray")

                resting_coord_idx =[]
                for i,s in enumerate(state):
                    if s==0:
                        resting_coord_idx.append(i)
                x = [coord[k][0] for k in resting_coord_idx]
                y = [coord[k][1] for k in resting_coord_idx]
                ax.scatter(x,y,s=50,color="tab:red")
                ax.set_xlim([0 ,im.shape[0]])
                ax.set_ylim([0 ,im.shape[1]])
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])

                
                ax = axes[1]
                plt.tight_layout()
                im = plt.imread(self.background_path)
                ax.imshow(im,zorder=1,cmap = "gray")
                x = [coord[k][0] for k in np.arange(len(coord))]
                y = [coord[k][1] for k in np.arange(len(coord))]

                x_f = uniform_filter1d(x, size=5)
                y_f = uniform_filter1d(y, size=5)

                c_f = np.arange(len(x_f))
                c_f = np.array([c_f[i]/25 for i in np.arange(len(c_f))])

                points = np.array([x_f, y_f]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Create a continuous norm to map from data points to colors
                norm = plt.Normalize(c_f.min(), c_f.max())
                lc = LineCollection(segments, cmap='viridis', norm=norm)

                # Set the values used for colormapping
                lc.set_array(c_f)
                lc.set_linewidth(1)
                line = ax.add_collection(lc)
                cbar = fig.colorbar(line, ax=ax,fraction=0.046, pad=0.04,orientation="horizontal")

                ax.set_xlim([0 ,im.shape[0]])
                ax.set_ylim([0 ,im.shape[1]])
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])

                ax  =axes[2]
                ax.plot(mosquito_tracks.objects[id]["state"])

                plt.ioff()
                plt.savefig(self.folder_analysis+"plots_trajectories/__"+self.video_name+"_plot_ID_"+str(id)+'.png',bbox_inches='tight')



                        
    def get_datetime_from_file_name(self):
        try:
            print(self.video_name)
            if "_raspberrypi_" in self.video_name:
                s = self.video_name.find('_raspberrypi_')
                l = 13
                #print(s)
            elif "_mosquipi4_" in self.video_name:
                s = self.video_name.find('_mosquipi4_')
                l = 11
            elif "_mosquipi3_" in self.video_name:
                s = self.video_name.find('_mosquipi3_')
                l = 11

            elif "_mosquipi1_" in self.video_name:
                s = self.video_name.find('_mosquipi1_')
                l = 11

            elif "_moscam" in self.video_name:
                s = self.video_name.find('_moscam')
                l = 11
                #print(s)
            #print(s)
            YY = int(self.video_name[s-6:s-4])
            #print(YY)
            MM = int(self.video_name[s-4:s-2])
            #print(MM)
            DD = int(self.video_name[s-2:s])
            #print(DD)
            HH = int(self.video_name[s+l:s+l+2])
            #print(HH)
            MI = int(self.video_name[s+l+2:s+l+4])
            #print(MI)
            SS = int(self.video_name[s+l+4:s+l+6])
            print(SS)
            VV = int(self.video_name[s+l+8:s+l+10])*1204

            t = datetime(2000+YY,MM,DD,HH,MI,SS)
            #print(t)
            t = t + timedelta(seconds=VV)
        except Exception as e:
            #print(e)
            return print("Incorrect name file, cannot find date")

        return t


    def extract_flight_metrics_around_resting(self):
        # Assuming the mosquito_tracks object has already been computed and exists.
        assert hasattr(self, 'mosquito_tracks'), "mosquito_tracks object not found. Run the extraction pipeline first."
        
        borders = {
            'sugar': self.settings["sugar_border_points"],
            'hs': self.settings["control_border_points"],
            'left_ctrl': self.settings["square_4_border_points"],
            'right_ctrl': self.settings["square_3_border_points"]
        }
        #print(borders)

        metrics = []

        for id in self.mosquito_tracks.objects.keys():
            resting_times = []
            coords = self.mosquito_tracks.objects[id]["coordinates"]
            state = self.mosquito_tracks.objects[id]["state"]

            # Analyze the coordinates and states to check for resting points in each zone
            for zone, border in borders.items():
                is_in_zone = [self.point_inside_cage(border, coord) and state[i] == 0 for i, coord in enumerate(coords)]
                resting_indices = np.where(is_in_zone)[0]
                
                if len(resting_indices) == 0:
                    continue

                # Extracting landing metrics (if any)
                landing_duration, landing_speed = np.nan, np.nan
                if resting_indices[0] > 0:
                    # Find the end of the last flight segment before resting
                    landing_end_idx = resting_indices[0] - 1
                    
                    landing_start_idx = landing_end_idx
                    if landing_start_idx >= 0:
                        while landing_start_idx > 0 and state[landing_start_idx] == 1:
                            landing_start_idx -= 1
                        if state[landing_start_idx] == 0:
                            landing_start_idx += 1
                        # Debug prints
                        #print(f"(ID: {id}) Landing: Start={landing_start_idx}, End={landing_end_idx}")     
                        landing_coords = coords[landing_start_idx:landing_end_idx + 1]
                        if len(landing_coords) > 1:  # To prevent issues with 1D arrays
                            landing_dists = np.linalg.norm(np.diff(landing_coords, axis=0), axis=1)
                            landing_duration = len(landing_coords) / 25  # Assuming 25 FPS
                            landing_speed = np.mean(landing_dists) if landing_duration > 0 else np.nan
                        else:
                            landing_duration, landing_speed = np.nan, np.nan
                        #print("Landing Metrics - Duration:", landing_duration, " Speed:", landing_speed)

                # Extracting takeoff metrics (if any)
                takeoff_duration, takeoff_speed = np.nan, np.nan
                if resting_indices[-1] < len(state) - 1:
                    # Find the start of the first flight segment after resting
                    takeoff_start_idx = resting_indices[-1] + 1

                    takeoff_end_idx = takeoff_start_idx

                    if takeoff_end_idx < len(state):
                        while takeoff_end_idx < len(state) and state[takeoff_end_idx] == 1:
                            takeoff_end_idx += 1
                        takeoff_coords = coords[takeoff_start_idx:takeoff_end_idx]
                        if len(takeoff_coords) > 1:  # To prevent issues with 1D arrays
                            takeoff_dists = np.linalg.norm(np.diff(takeoff_coords, axis=0), axis=1)
                            takeoff_duration = len(takeoff_coords) / 25  # Assuming 25 FPS
                            takeoff_speed = np.mean(takeoff_dists) if takeoff_duration > 0 else np.nan
                        else:
                            takeoff_duration, takeoff_speed = np.nan, np.nan
                        #print("Takeoff Metrics - Duration:", takeoff_duration, " Speed:", takeoff_speed)

                # Resting duration
                resting_duration = len(resting_indices) / 25  # Assuming 25 FPS

                metrics.append([zone, id, [landing_duration, landing_speed], resting_duration, [takeoff_duration, takeoff_speed]])
                #print(f"ID: {id}, Zone: {zone}, Metrics: {metrics[-1]}")

        metrics_df = pd.DataFrame(metrics, columns=['zone', 'mosquito_id', 'landing', 'resting_time', 'takeoff'])
        self.mosquito_tracks.flight_metrics_around_resting = metrics_df
        print("Flight metrics around resting points have been extracted and saved.")

# ########################## Assemble backward and forward tracks ###################
#    def assemble_forward_backward_tracks(self,step_to_force_analyze):








#     def assemble_forward_backward_tracks(self):

#         #Parameters
#         max_dist_rest = 3
        
#         # Load necessary data forward
#         direction = "forward"
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             moving_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/matching_matrix_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             matching_matrix = pickle.load(f)

#         # Load necessary data backward
#         direction = "backward"
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             back_resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             back_moving_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/matching_matrix_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             back_matching_matrix = pickle.load(f)

#         # Get the list of IDs resting
#         list_of_ID_resting_forward = get_ids_from_dict(resting_object_traj)
#         list_of_ID_resting_backward = get_ids_from_dict(back_resting_object_traj)

#         n_frames = self.total_number_frames

#         numb_id_resting_forward = len(list_of_ID_resting_forward)
#         numb_id_resting_backward = len(list_of_ID_resting_backward)

#         #print(numb_id_resting_forward)
#         #print(numb_id_resting_backward)

#         # Square zeros matrix to store link between IDs (resting and moving) and (moving moving)
#         matching_matrix_rest = np.zeros((numb_id_resting_forward+numb_id_resting_backward,numb_id_resting_forward+numb_id_resting_backward))

#         for i,rest_id_forw in enumerate(list_of_ID_resting_forward): # For all ID of forward tracking
#             progress_bar(i, numb_id_resting_forward , bar_length=20)
#             time_forw = resting_object_traj["time_ID_"+str(rest_id_forw)]
#             pos_forw = resting_object_traj["ID_"+str(rest_id_forw)]

#             for j,rest_id_back in enumerate(list_of_ID_resting_backward):
#                 time_back = back_resting_object_traj["time_ID_"+str(rest_id_back)]
#                 pos_back = back_resting_object_traj["ID_"+str(rest_id_back)]

#                 time_back.reverse()
#                 pos_back.reverse()

#                 time_back = [n_frames-k for k in time_back]

#                 start_f = time_forw[0]
#                 end_f = time_forw[-1]

#                 start_b = time_back[0]
#                 end_b = time_back[-1]

#                 start = np.min([start_f,start_b])
#                 end = np.max([end_f,end_b])

#                 if end-start < (end_f-start_f)+(end_b-start_b):
#                     start_search = np.max([start_f,start_b])
#                     end_search = np.min([end_f,end_b])

#                     start_time_back = time_back.index(start_search)
#                     start_time_forw = time_forw.index(start_search)

#                     for s,t in enumerate(np.arange(start_search,end_search,1)):
#                         D_traj = dist.cdist(np.array(pos_forw[start_time_forw+s]).reshape(1,-1),np.array(pos_back[start_time_back+s]).reshape(1,-1))
#                         if D_traj[0] < max_dist_rest:
#                             matching_matrix_rest[i][j+numb_id_resting_forward] +=1

#         with open(self.folder_intermediate+"/matching_matrix_rest_"+self.video_name+".pkl", 'wb') as f:
#             pickle.dump(matching_matrix_rest,f)

#         matching_matrix_rest = np.where(matching_matrix_rest>40,1,0)
#         matched_ids = graph.connected_components(matching_matrix_rest)
#         print(matched_ids)



########################## Assemble resting and moving obj tracks ##############
#     def assemble_resting_moving_tracks(self,direction):
#         print("Start assembling resting and moving tracks "+direction)
#         # Parameters
#         max_distance_takeoff = 60
#         time_window_takeoff = 30
#         max_distance_landing = 60
#         time_window_landing = 30
#         max_distance_moving_cross = 20

#         # Load necessary data
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             moving_object_traj = pickle.load(f)

#         # Get the list of IDs resting
#         list_of_ID_still = get_ids_from_dict(resting_object_traj)
#         list_of_ID_moving = get_ids_from_dict(moving_object_traj)

#         numb_id_resting = len(list_of_ID_still)
#         numb_id_moving = len(list_of_ID_moving)

#         # Square zeros matrix to store link between IDs (resting and moving) and (moving moving)
#         matching_matrix = np.zeros((numb_id_resting+numb_id_moving,numb_id_resting+numb_id_moving))

#         # Find all the intersection between resting and moving tracks
#         for i,rest_id in enumerate(list_of_ID_still):
#             progress_bar(i, numb_id_resting, bar_length=20)
#             pos_v = resting_object_traj["ID_"+str(rest_id)]
#             time_v = resting_object_traj["time_ID_"+str(rest_id)]

#             if time_v[-1] == self.total_number_frames-1:
#                 print("resting id "+str(rest_id)+" does not move until end of vid")
#             else:
#                 t_landing_tar = time_v[0]
#                 t_takeoff_tar = time_v[-self.maxobjdisappear_resting]
#                 for j,mov_id in enumerate(list_of_ID_moving):

#                     mov_pos = moving_object_traj["ID_"+str(mov_id)]
#                     mov_time = moving_object_traj["time_ID_"+str(mov_id)]

#                     # Look for landing
#                     for t_landing in np.arange(t_landing_tar-int(time_window_landing/2), t_landing_tar+int(time_window_landing/2), 1):
#                         if t_landing in mov_time:
#                             t_landing_mov = mov_time.index(t_landing)
#                             D_landing = dist.cdist(np.array(pos_v[0]).reshape(1,-1), np.array(mov_pos[t_landing_mov]).reshape(1,-1))
#                             if D_landing[0].min(axis=0) < max_distance_landing:
#                                 #print("resting id "+str(rest_id)+" lands with :"+str(mov_id))
#                                 matching_matrix[i][j+numb_id_resting] += 1 # Save the link for landing

#                     # Look for take-off match
#                     for t_takeoff in np.arange(t_takeoff_tar-int(time_window_takeoff/2), t_takeoff_tar+int(time_window_takeoff/2), 1):
#                         if t_takeoff in mov_time:
#                             t_takeoff_mov = mov_time.index(t_takeoff)
#                             D_takeoff = dist.cdist(np.array(pos_v[-self.maxobjdisappear_resting]).reshape(1,-1), np.array(mov_pos[t_takeoff_mov]).reshape(1,-1))
#                             if D_takeoff[0].min(axis=0) < max_distance_takeoff:
#                                 #print("resting id "+str(rest_id)+" takes-off with :"+str(mov_id))
#                                 matching_matrix[i][j+numb_id_resting] += 1 

#         # Find all the intersection between moving tracks
#         # for j_1,mov_id_1 in enumerate(list_of_ID_moving):
#         #     progress_bar(j_1, numb_id_moving, bar_length=20)
#         #     mov_pos_1 = moving_object_traj["ID_"+str(mov_id_1)]
#         #     mov_time_1 = moving_object_traj["time_ID_"+str(mov_id_1)]
#         #     for k,t in enumerate(mov_time_1):
#         #         for j_2,mov_id_2 in enumerate(list_of_ID_moving):
#         #             mov_pos_2 = moving_object_traj["ID_"+str(mov_id_2)]
#         #             mov_time_2 = moving_object_traj["time_ID_"+str(mov_id_2)]
#         #             if np.abs(j_2-j_1)>0:
#         #                 if t in mov_time_2:
#         #                     t_mov_2 = mov_time_2.index(t)
#         #                     D_mov = dist.cdist(np.array(mov_pos_1[k]).reshape(1,-1), np.array(mov_pos_2[t_mov_2]).reshape(1,-1))
#         #                     if D_mov[0].min(axis=0) < max_distance_moving_cross:
#         #                         matching_matrix[j_1+numb_id_resting][j_2+numb_id_resting] += 1
#         #                         print("moving id "+str(mov_id_1)+" cross with moving id :"+str(mov_id_2))

#         #matching_matrix = np.where(matching_matrix>0,1,0)
#         #matched_ids = graph.connected_components(matching_matrix)
#         with open(self.folder_intermediate+"/matching_matrix_"+direction+"_"+self.video_name+".pkl", 'wb') as f:
#             pickle.dump(matching_matrix, f)
#         #print(matched_ids)
# ########################## Assemble resting and moving obj tracks ##############
#     def assemble_resting_moving_tracks_V2(self,direction):
#         print("Start assembling resting and moving tracks "+direction)
#         # Parameters
#         max_distance_takeoff = 60
#         time_window_takeoff = 30
#         max_distance_landing = 60
#         time_window_landing = 30
#         max_distance_moving_cross = 20

#         # Load necessary data
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             moving_object_traj = pickle.load(f)

#         # Get the list of IDs resting
#         list_of_ID_still = get_ids_from_dict(resting_object_traj)
#         list_of_ID_moving = get_ids_from_dict(moving_object_traj)

#         numb_id_resting = len(list_of_ID_still)
#         numb_id_moving = len(list_of_ID_moving)

#         rest_ids_boots = np.empty((len(list_of_ID_still),2,))
#         rest_ids_boots[:] = np.nan
#         mov_ids_boots = np.empty((len(list_of_ID_moving),2,))
#         mov_ids_boots[:] = np.nan

#         # Find all the intersection between resting and moving tracks
#         for i,rest_id in enumerate(list_of_ID_still):
#             progress_bar(i, numb_id_resting, bar_length=20)
#             pos_v = resting_object_traj["ID_"+str(rest_id)]
#             time_v = resting_object_traj["time_ID_"+str(rest_id)]

#             if time_v[-1] == self.total_number_frames-1:
#                 print("resting id "+str(rest_id)+" does not move until end of vid")
#             else: # If track disappears before the end of the video
#                 t_landing_tar = time_v[0]
#                 t_takeoff_tar = time_v[-self.maxobjdisappear_resting]

#                 # Initialize 
#                 min_dist_takeoff = max_distance_takeoff
#                 min_dist_landing = max_distance_landing

#                 mov_id_takeoff_tar = np.nan
#                 mov_id_landing_tar = np.nan

#                 j_mov_takeoff = np.nan
#                 j_mov_landing = np.nan

#                 # Loop over all the moving IDs
#                 for j,mov_id in enumerate(list_of_ID_moving):

#                     mov_pos = moving_object_traj["ID_"+str(mov_id)]
#                     mov_time = moving_object_traj["time_ID_"+str(mov_id)]

#                     # Look for landing
#                     if np.isnan(rest_ids_boots[i][0]) == 1 and np.isnan(mov_ids_boots[j][1]) == 1: # if ids not already matched 
#                         for t_landing in np.arange(t_landing_tar-int(time_window_landing/2), t_landing_tar+int(time_window_landing/2), 1):
#                             if t_landing in mov_time:
#                                 t_landing_mov = mov_time.index(t_landing)
#                                 D_landing = dist.cdist(np.array(pos_v[0]).reshape(1,-1), np.array(mov_pos[t_landing_mov]).reshape(1,-1))
#                                 if D_landing[0].min(axis=0) < max_distance_landing:
#                                     #print("resting id "+str(rest_id)+" lands with :"+str(mov_id))
#                                     #matching_matrix[i][j+numb_id_resting] += 1 # Save the link for landing
#                                     if D_landing[0].min(axis=0)<min_dist_landing: #and np.abs(t_landing-t_landing_tar)<min_time_diff_landing: # If better landing match
#                                         min_dist_landing = D_landing[0].min(axis=0)
#                                         #min_time_diff_landing = np.abs(t_landing-t_landing_tar)
#                                         mov_id_landing_tar = mov_id
#                                         j_mov_landing = j

#                     # Look for take-off match
#                     if np.isnan(rest_ids_boots[i][1]) == 1 and np.isnan(mov_ids_boots[j][0]) == 1: # if ids not already matched 
#                         for t_takeoff in np.arange(t_takeoff_tar-int(time_window_takeoff/2), t_takeoff_tar+int(time_window_takeoff/2), 1):
#                             if t_takeoff in mov_time:
#                                 t_takeoff_mov = mov_time.index(t_takeoff)
#                                 D_takeoff = dist.cdist(np.array(pos_v[-self.maxobjdisappear_resting]).reshape(1,-1), np.array(mov_pos[t_takeoff_mov]).reshape(1,-1))
#                                 if D_takeoff[0].min(axis=0) < max_distance_takeoff:
#                                         if D_takeoff[0].min(axis=0)<min_dist_takeoff: #and np.abs(t_takeoff-t_takeoff_tar)<min_time_diff_takeoff: # If better landing match
#                                             min_dist_takeoff = D_takeoff[0].min(axis=0)
#                                             #min_time_diff_takeoff = np.abs(t_takeoff-t_takeoff_tar)
#                                             mov_id_takeoff_tar = mov_id
#                                             j_mov_takeoff = j
#                                 #print("resting id "+str(rest_id)+" takes-off with :"+str(mov_id))
#                                 #matching_matrix[i][j+numb_id_resting] += 1 


#                 if np.isnan(mov_id_takeoff_tar)==0: # If a taking-off match was found
#                     #print("resting id "+str(rest_id)+" takesoff with :"+str(mov_id_takeoff_tar))
#                     #print(min_dist_takeoff)
#                     rest_ids_boots[i][1] = mov_id_takeoff_tar
#                     mov_ids_boots[j_mov_takeoff][1] = rest_id

#                 if np.isnan(mov_id_landing_tar)==0: # If a landing match was found
#                     #print("resting id "+str(rest_id)+" lands with :"+str(mov_id_landing_tar))
#                     #print(min_dist_landing)
#                     rest_ids_boots[i][0] = mov_id_landing_tar
#                     mov_ids_boots[j_mov_landing][1] = rest_id
#         print(rest_ids_boots)
#         print(mov_ids_boots)
#         # with open(self.folder_intermediate+"/matching_matrix_"+direction+"_"+self.video_name+".pkl", 'wb') as f:
#             #     pickle.dump(matching_matrix, f)
#         #print(matched_ids)
#         # 

# ########################## Assemble resting and moving obj tracks ##############
#     def assemble_resting_moving_tracks_V3(self,direction):
#         print("Start assembling resting and moving tracks "+direction)

#         # Load necessary data
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             moving_object_traj = pickle.load(f)

#         # Parameters
#         limit_max_distance_search = 100
#         time_window_search = 30

#         # Get the list of IDs resting
#         list_of_ID_still = get_ids_from_dict(resting_object_traj)
#         list_of_ID_moving = get_ids_from_dict(moving_object_traj)        # 

#         rest_ids_boots = np.empty((len(list_of_ID_still),3,))
#         rest_ids_boots[:] = np.nan
#         for k in np.arange(len(list_of_ID_still)):
#             rest_ids_boots[k][0] = list_of_ID_still[k]
        
#         mov_ids_boots = np.empty((len(list_of_ID_moving),3,))
#         mov_ids_boots[:] = np.nan
#         for k in np.arange(len(list_of_ID_moving)):
#             mov_ids_boots[k][0] = list_of_ID_moving[k]
        
#         # Initialize
#         updated_moving_traj = moving_object_traj
#         updated_resting_traj = resting_object_traj
#         current_resting_traj = resting_object_traj
#         current_moving_traj = moving_object_traj

#         max_distance_search = 2 # Initial search distance
#         while max_distance_search < limit_max_distance_search:

#             # Loop over all the resting IDs
#             for i,rest_id in enumerate(list_of_ID_still):
#                 print(i)
#                 updated_moving_traj,updated_resting_traj,rest_ids_boots,mov_ids_boots = self.find_match_with_moving(i,rest_id,current_resting_traj,current_moving_traj,time_window_search,max_distance_search,rest_ids_boots,mov_ids_boots,updated_resting_traj,updated_moving_traj)
            
#             # Udpate the traj for the next round
#             current_moving_traj = updated_moving_traj
#             current_resting_traj = updated_resting_traj

#             max_distance_search += 2
#             print(rest_ids_boots)

# #####################################

    
# #####################################
#     def find_match_with_moving(self,i,rest_id,current_resting_traj,current_moving_traj,time_window_search,max_distance_search,rest_ids_boots,mov_ids_boots,updated_resting_traj,updated_moving_traj):
#         list_of_ID_moving = get_ids_from_dict(current_moving_traj) 
        
#         pos_v = current_resting_traj["ID_"+str(rest_id)]
#         time_v = current_resting_traj["time_ID_"+str(rest_id)]

#         if time_v[-1] == self.total_number_frames-1:
#             print("resting id "+str(rest_id)+" does not move until end of vid")
#         else: # If track disappears before the end of the video
#             t_landing_tar = time_v[0]
#             t_takeoff_tar = time_v[-self.maxobjdisappear_resting]

#             # Initialize 
#             min_time_diff_takeoff = int(time_window_search/2)
#             min_time_diff_landing = int(time_window_search/2)

#             mov_id_takeoff_tar = np.nan
#             mov_id_landing_tar = np.nan

#             j_mov_takeoff = np.nan
#             j_mov_landing = np.nan

#             # Loop over all the moving IDs
#             for j,mov_id in enumerate(list_of_ID_moving):

#                 mov_pos = current_moving_traj["ID_"+str(mov_id)]
#                 mov_time = current_moving_traj["time_ID_"+str(mov_id)]

#                 # Look for landing
#                 print(np.size(mov_ids_boots))
#                 if np.isnan(rest_ids_boots[i][1]) == 1 and np.isnan(mov_ids_boots[j][2]) == 1: # if ids not already matched 
#                     for t_landing in np.arange(t_landing_tar-int(time_window_search/2), t_landing_tar+int(time_window_search/2), 1):
#                         if t_landing in mov_time:
#                             t_landing_mov = mov_time.index(t_landing)
#                             D_landing = dist.cdist(np.array(pos_v[0]).reshape(1,-1), np.array(mov_pos[t_landing_mov]).reshape(1,-1))
#                             if D_landing[0].min(axis=0) < max_distance_search:
#                                 #print("resting id "+str(rest_id)+" lands with :"+str(mov_id))
#                                 #matching_matrix[i][j+numb_id_resting] += 1 # Save the link for landing
#                                 if np.abs(t_landing-t_landing_tar)<min_time_diff_landing: # If better landing match
#                                     #min_dist_landing = D_landing[0].min(axis=0)
#                                     min_time_diff_landing = np.abs(t_landing-t_landing_tar)
#                                     mov_id_landing_tar = mov_id
#                                     j_mov_landing = j
#                                     t_landing_opt = t_landing_mov

#                 # Look for take-off match
#                 if np.isnan(rest_ids_boots[i][2]) == 1 and np.isnan(mov_ids_boots[j][1]) == 1: # if ids not already matched 
#                     for t_takeoff in np.arange(t_takeoff_tar-int(time_window_search/2), t_takeoff_tar+int(time_window_search/2), 1):
#                         if t_takeoff in mov_time:
#                             t_takeoff_mov = mov_time.index(t_takeoff)
#                             D_takeoff = dist.cdist(np.array(pos_v[-self.maxobjdisappear_resting]).reshape(1,-1), np.array(mov_pos[t_takeoff_mov]).reshape(1,-1))
#                             if D_takeoff[0].min(axis=0) < max_distance_search:
#                                     if np.abs(t_takeoff-t_takeoff_tar)<min_time_diff_takeoff: # If better landing match
#                                         #min_dist_takeoff = D_takeoff[0].min(axis=0)
#                                         min_time_diff_takeoff = np.abs(t_takeoff-t_takeoff_tar)
#                                         mov_id_takeoff_tar = mov_id
#                                         j_mov_takeoff = j
#                                         t_takeoff_opt = t_takeoff_mov

#                 # Update the trajectories
                
#                 if np.isnan(mov_id_landing_tar)==0: # If landing match found
                    
#                     rest_ids_boots[i][1] = mov_id_landing_tar # Mark the ID as landed


#                     mov_pos = current_moving_traj["ID_"+str(mov_id_landing_tar)]
#                     mov_time = current_moving_traj["time_ID_"+str(mov_id_landing_tar)]

#                     # Remove the part of the flying traj after landing
#                     updated_moving_traj["ID_"+str(mov_id_landing_tar)] = mov_pos[0:t_landing_opt]
#                     updated_moving_traj["time_ID_"+str(mov_id_landing_tar)] = mov_time[0:t_landing_opt]
                    
#                     # Transfer this deleted traj as a new flying trajectory
#                     list_of_ID_moving = get_ids_from_dict(updated_moving_traj)  
#                     new_mov_id = np.max(list_of_ID_moving)+1 # define an ID that does not exist

#                     updated_moving_traj["ID_"+str(new_mov_id)] = mov_pos[t_landing_opt:-1]
#                     updated_moving_traj["time_ID_"+str(new_mov_id)] = mov_time[t_landing_opt:-1]

#                     index_vec = [mov_ids_boots[i][0] for i in np.arange(len(mov_ids_boots))]
#                     index_of_moving_id =  index_vec.index(index_vec == mov_id_landing_tar)
#                     np.append(mov_ids_boots,[[new_mov_id,np.nan,mov_ids_boots[index_of_moving_id][2]]],axis=0)

#                 if np.isnan(mov_id_takeoff_tar)==0: # If taking-off match found

#                     rest_ids_boots[i][2] = mov_id_takeoff_tar # Mark the ID as taken-off
#                     updated_resting_traj["ID_"+str(rest_id)] = updated_resting_traj["ID_"+str(rest_id)][0:-self.maxobjdisappear_resting]
#                     updated_resting_traj["time_ID_"+str(rest_id)] = updated_resting_traj["time_ID_"+str(rest_id)][0:-self.maxobjdisappear_resting]

#                     # Remove the part of the flying traj before taking-off
#                     mov_pos = current_moving_traj["ID_"+str(mov_id_takeoff_tar)]
#                     mov_time = current_moving_traj["time_ID_"+str(mov_id_takeoff_tar)]

#                     # Remove the part of the flying traj before taking off
#                     #print(updated_moving_traj.keys())
#                     updated_moving_traj["ID_"+str(mov_id_takeoff_tar)] = mov_pos[t_takeoff_opt:-1]
#                     updated_moving_traj["time_ID_"+str(mov_id_takeoff_tar)] = mov_time[t_takeoff_opt:-1]
                    
#                     # Transfer this deleted traj as a new flying trajectory
#                     list_of_ID_moving = get_ids_from_dict(updated_moving_traj)  
#                     new_mov_id = np.max(list_of_ID_moving)+1 # define an ID that does not exist

#                     updated_moving_traj["ID_"+str(new_mov_id)] = mov_pos[0:t_takeoff_opt]
#                     updated_moving_traj["time_ID_"+str(new_mov_id)] = mov_time[0:t_takeoff_opt]

#                     index_vec = [mov_ids_boots[i][0] for i in np.arange(len(mov_ids_boots))]
#                     index_of_moving_id =  index_vec.index(mov_id_takeoff_tar)
#                     #print(index_of_moving_id)
#                     #print(mov_ids_boots[-1])
#                     np.append(mov_ids_boots,[[new_mov_id,mov_ids_boots[index_of_moving_id][1],np.nan]],axis=0)


#         return updated_moving_traj,updated_resting_traj,rest_ids_boots,mov_ids_boots
# ########################## Display video with assembled tracks ###################
#     def display_assembled_resting_moving_tracks(self,direction,starting_frame,time_btw_frames):

#         # Load necessary data
#         with open(self.folder_intermediate+"/resting_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             resting_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/moving_obj_tracks_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             moving_object_traj = pickle.load(f)
#         with open(self.folder_intermediate+"/matching_matrix_"+direction+"_"+self.video_name+".pkl", 'rb') as f:
#             matching_matrix = pickle.load(f)


#         matching_matrix = np.where(matching_matrix>0,1,0)
#         matched_ids = graph.connected_components(matching_matrix)

#         # Get the list of IDs resting
#         list_of_ID_still = get_ids_from_dict(resting_object_traj)
#         list_of_ID_moving = get_ids_from_dict(moving_object_traj)

#         numb_id_resting = len(list_of_ID_still)
#         numb_id_moving = len(list_of_ID_moving)


#         # Initialiaze video
#         cap = cv2.VideoCapture(self.video_path)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
#         n_frame = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
#         f_i = starting_frame

#         moving_objects_ID = list(moving_object_traj.keys())
#         resting_objects_ID = list(resting_object_traj.keys())

#         #print(list_of_ID_still)
#         #print(resting_objects_ID)

#         while True:
#             suc,frame = cap.read()
#             time.sleep(time_btw_frames)

#             if suc == True:
#                 frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 img = frame.copy()

#                 if f_i > 1:
#                     if direction =="forward":
#                         frame_idx = f_i
#                     else:
#                         frame_idx = n_frame-f_i


#                     # Show resting tracks
#                     for k,id in enumerate(resting_objects_ID):
#                         parse_ID = id.partition("ID_")[2]
#                         if "time_ID_"+str(parse_ID) in resting_objects_ID:
#                             if frame_idx in resting_object_traj["time_ID_"+str(parse_ID)]:
#                                 t_frame = resting_object_traj["time_ID_"+str(parse_ID)].index(frame_idx)
#                                 if t_frame < len(resting_object_traj["ID_"+str(parse_ID)]):
#                                     index_still = list_of_ID_still.index(int(parse_ID))
#                                     text = "ID {}".format(matched_ids[1][index_still])
#                                     centroid = resting_object_traj["ID_"+str(parse_ID)][t_frame]
#                                     cv2.putText(img, text, (int(centroid[0])-20, int(centroid[1])-20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#                                     cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 2, (0, 0, 255), 1)

#                     # Show moving tracks
#                     for k,id in enumerate(moving_objects_ID):
#                         parse_ID = id.partition("ID_")[2]
#                         if "time_ID_"+str(parse_ID) in moving_objects_ID:
#                             if frame_idx+1 in moving_object_traj["time_ID_"+str(parse_ID)]:
#                                 t_frame = moving_object_traj["time_ID_"+str(parse_ID)].index(frame_idx+1)
#                                 if t_frame < len(moving_object_traj["ID_"+str(parse_ID)]):
#                                     index_mov = list_of_ID_moving.index(int(parse_ID))
#                                     text = "ID {}".format(matched_ids[1][index_mov+numb_id_resting])
#                                     centroid = moving_object_traj["ID_"+str(parse_ID)][t_frame]
#                                     cv2.putText(img, text, (int(centroid[0])-20, int(centroid[1])-20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#                                     cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 4, (0, 255, 0), 1)

#                 cv2.imshow("frame",img)
#                 f_i += 1

#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break

#                 if f_i>n_frame-2:
#                     break
#         cap.release()
#         cv2.destroyAllWindows()





# ########################## Check moving object traj looks like a mosquito flight (maybe remove?) ####################
#     def looks_like_flight(self,moving_object_single_traj):

#         max_vel = 40
#         min_vel = 2
#         min_len = 40

#         velocity = []
#         for t,centroid in enumerate(moving_object_single_traj):
#             if t>0:
#                 velocity.append(dist.cdist(np.array(moving_object_single_traj[t-1]).reshape(1,-1),np.array(moving_object_single_traj[t]).reshape(1,-1)))

#         average_speed = np.mean(velocity)
#         unicity = len(np.unique(velocity))
#         #print(average_speed)
#         return average_speed < max_vel and average_speed > min_vel
