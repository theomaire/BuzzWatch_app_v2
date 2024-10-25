########################## IMPORT ALL NECESSARY PACKAGES ####################

# import numpy as np
# import cv2
import os
import yaml
from os import listdir
from os.path import isfile, join
from buzzwatch_data_analysis.misc_functions import *
from buzzwatch_data_analysis.single_video_analysis import *
import pickle
import sys
import numpy as np
from logger import MultiLogger
# Add the custom constructor to PyYAML Loader

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


##################################################################################################################################
########################## Class that manages all videos from a given experiment ####################
class buzzwatch_experiment_analysis:
    # Class variables

    def __init__(self,folder_analysis,folder_videos,experiment_alias,settings,settings_file,log_func=None,debug_mode=False):

        self.experiment_alias = experiment_alias
        self.folder_videos = folder_videos
        self.folder_analysis = folder_analysis
        #print(self.folder_analysis)
        self.background_path = folder_analysis+"/background_"+experiment_alias+".png"

        try:
            files_video = [f for f in listdir(folder_videos) if isfile(join(folder_videos, f)) and f.endswith(".mp4")]
            files_video.sort()
            
            self.list_video_name = [os.path.splitext(video_mp4)[0] for video_mp4 in files_video]
            self.list_videos_files = files_video
        except Exception:
            print("Warning no videos files in "+ folder_videos)
        self.folder_test_tracking = folder_analysis+"/test_tracking"
        create_folder(self.folder_test_tracking)
        self.folder_final = folder_analysis+"/final_tracking_data"
        create_folder(self.folder_final)
        self.folder_temp_data = folder_analysis+"/temp_data"
        create_folder(self.folder_temp_data)



        if os.path.isfile(settings_file):
            with open(settings_file, 'r') as file:
                settings = yaml.safe_load(file)
                self.settings = settings
        self.settings_file = settings_file
        self.control_border_points = self.settings["control_border_points"]
        self.cage_border_points = self.settings["cage_border_points"]
        self.sugar_border_points = self.settings["sugar_border_points"]
        if debug_mode:
            # Save the experiment_object
            exp_object_path = self.folder_temp_data+"/temp_data_"+self.experiment_alias+".pkl"
            with open(exp_object_path, 'wb') as f:
                pickle.dump(self,f)

        self.log = log_func  # Store the logging function
########################## Compute background or not ####################
    def add_background(self,force_to_redo):
        if force_to_redo == 1 or os.path.exists(self.background_path)==0:
            background= self.get_background()
            cv2.imwrite(self.background_path, background)

########################## Background from median frame of all videos ####################
    def get_background(self):
        print("Extracting the background image")
        images_to_av = []
        for k,video_file in enumerate(self.list_videos_files):
            progress_bar(k, len(self.list_videos_files), bar_length=20)
            if video_file.endswith('.mp4'):
                #start = video_file.find('Cage')
                video_path = self.folder_videos+video_file#[start::]
                print(video_path)
                cap = cv2.VideoCapture(video_path)
                suc,frame = cap.read()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                images_to_av.append(frame_gray)
                cap.release()
        median_frame = np.median(images_to_av, axis=0).astype(np.uint8)
        print("Background image saved")
        return median_frame

########################## Background from median frame of all videos ####################
    def run_video_analysis_test(self,video_idx,step_to_force_analyze):
        """ 
        This function analyzes a videos for testing the tracking performance
        before running everything on a cluster or workstation
        """
        video_name = os.path.splitext(self.list_videos_files[video_idx])[0]
        video_object_path = self.folder_temp_data+"/temp_data_"+video_name+".pkl"
        path_back = self.folder_analysis+"images_mortality/"+video_name+".png"
        #path_back = self.background_path
        
        if os.path.isfile(path_back): # If the video has a background image made already
        
            if os.path.isfile(video_object_path)==0 or step_to_force_analyze==0: # if video_object was not already created
                video_tracked = single_video_analysis(self,video_idx,debug_mode=1) # Initialize a video analysis object.
            else:
                print("Load video_obj of : "+video_name)
                with open(video_object_path, 'rb') as f:
                    video_tracked = pickle.load(f)

    ########       
            video_tracked.segment_resting_and_moving_objects(step_to_force_analyze,debug_mode=1) # Step 1
            video_tracked.track_resting_obj(step_to_force_analyze,debug_mode=1) #                  Step 2
            video_tracked.track_moving_obj(step_to_force_analyze,debug_mode=1) #                   Step 3
    ######## Loop over increasing values of time_window_search and max_distance #    Step 4
            max_distance_search = self.settings["assembly"]["max_distance_search"]
            time_window_search = self.settings["assembly"]["time_window_search"]
            MAX_NB_ASSEMBLING = 5
            for trial in np.arange(MAX_NB_ASSEMBLING):
                print("Tracking step :"+str(trial))
                video_tracked.clean_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=1)
                try:
                    video_tracked.assemble_resting_and_moving_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=1)
                except Exception:
                    print("No resting tracks to match")
                try:
                    video_tracked.assemble_unmatched_moving_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=1)
                except Exception:
                    print("No moving tracks to match")

                if trial == MAX_NB_ASSEMBLING-1: # Finished last round
                    video_tracked.clean_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=1)

                time_window_search = time_window_search*1.1
                max_distance_search = max_distance_search*1.2

    # ############
    # # #         # Assemble forward backward tracks #                                     Step 5
            video_tracked.assemble_tracks_ids(step_to_force_analyze,debug_mode=1)
            video_tracked.extract_complete_trajectories_from_video(debug_mode=1)

            video_tracked.display_video_with_tracking("forward",starting_frame=15000,time_btw_frames=0.005)

        else:
            print("background image missing")
        


########################### Compute activity and trajectories from the video. ###################
    def run_video_analysis_all(self,video_idx):
        """ 
        This function analyzes all videos from a given folder
        """
        # Logging files
        create_folder(self.folder_analysis+"log_analysis")
        video_name = os.path.splitext(self.list_videos_files[video_idx])[0]

        path_back = self.folder_analysis+"images_mortality/"+video_name+".png"
        #path_back = self.background_path
        if os.path.isfile(path_back): # If the video has a background image made already

            if os.path.isfile(self.folder_final+"/forward_mosq_tracks_"+video_name)==False:
                log_file = open(self.folder_analysis+"log_analysis/"+video_name+".log","w")
                old_stdout = sys.stdout
                sys.stdout = log_file

                step_to_force_analyze = 0
                MAX_NB_ASSEMBLING = 5
                video_tracked = single_video_analysis(self,video_idx,debug_mode=0) # Initialize a video analysis object.
        ########
                video_tracked.segment_resting_and_moving_objects(step_to_force_analyze,debug_mode=0) # Step 1
                video_tracked.track_resting_obj(step_to_force_analyze,debug_mode=0) #                  Step 2
                video_tracked.track_moving_obj(step_to_force_analyze,debug_mode=0) #                   Step 3
        ######## Loop over increasing values of time_window_search and max_distance #    Step 4
                max_distance_search = self.settings["assembly"]["max_distance_search"]
                time_window_search = self.settings["assembly"]["time_window_search"]
                for trial in np.arange(MAX_NB_ASSEMBLING):
                    print("Tracking step :"+str(trial))
                    video_tracked.clean_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=0)
                    try:
                        video_tracked.assemble_resting_and_moving_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=0)
                    except Exception:
                        print("No resting tracks to match")
                    try:
                        video_tracked.assemble_unmatched_moving_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=0)
                    except Exception:
                        print("No moving tracks to match")
                    
                    if trial == MAX_NB_ASSEMBLING-1: # Finished last round
                        video_tracked.clean_tracks(step_to_force_analyze,time_window_search,max_distance_search,debug_mode=0)

                    time_window_search = time_window_search*1.1
                    max_distance_search = max_distance_search*1.2

        # ############
        # # #         # Assemble forward backward tracks #                                     Step 5
                video_tracked.assemble_tracks_ids(step_to_force_analyze,debug_mode=0)
                try:
                    video_tracked.extract_complete_trajectories_from_video(debug_mode=0)
                except Exception as error:
                # handle the exception
                    print("An exception occurred:", error)
                

                print("Finished analyzing "+video_name)

                sys.stdout = old_stdout
                log_file.close()
        else:
            print("background image missing")            


    def run_single_video_zone_traj_analysis(self, video_name,debug_mode):
            
        video_analysis = single_video_analysis(self, video_name, debug_mode=False)
        forward_tracks_path = os.path.join(self.folder_final, f"forward_mosq_tracks_{video_name}")

        with open(forward_tracks_path, 'rb') as f:
            video_analysis.mosquito_tracks = pickle.load(f)
    # Perform the analysis to extract flight metrics around resting points
        video_analysis.extract_flight_metrics_around_resting()
        video_analysis.save_tracking_results()
            



########################### Compute activity and trajectories from the video. ###################
    def run_single_video_analysis(self, video_name,debug_mode):
        """ 
        This function analyzes all videos from a given folder
        """
        # Logging files
        #create_folder(self.folder_analysis + "/log_analysis")

        path_back = os.path.join(self.folder_analysis, "images_mortality", f"{video_name}.png")
        #print(path_back)

        if os.path.isfile(path_back):  # If the video has a background image made already

            forward_tracks_path = os.path.join(self.folder_final, f"forward_mosq_tracks_{video_name}")
            if not os.path.isfile(forward_tracks_path):
                log_file_path = os.path.join(self.folder_analysis, "log_analysis", f"{video_name}.log")
                logger = MultiLogger(self.log, log_file_path)

                old_stdout = sys.stdout
                sys.stdout = logger

                step_to_force_analyze = 0
                MAX_NB_ASSEMBLING = 5
                video_tracked = single_video_analysis(self, video_name, debug_mode)  # Initialize a video analysis object.

                # Steps for video analysis
                video_tracked.segment_resting_and_moving_objects(step_to_force_analyze, debug_mode)  # Step 1
                video_tracked.track_resting_obj(step_to_force_analyze, debug_mode)  # Step 2
                video_tracked.track_moving_obj(step_to_force_analyze, debug_mode)  # Step 3

                # Loop over increasing values of time_window_search and max_distance # Step 4
                max_distance_search = self.settings["assembly"]["max_distance_search"]
                time_window_search = self.settings["assembly"]["time_window_search"]

                for trial in np.arange(MAX_NB_ASSEMBLING):
                    print(f"Tracking step: {trial}")
                    video_tracked.clean_tracks(step_to_force_analyze, time_window_search, max_distance_search, debug_mode)

                    try:
                        video_tracked.assemble_resting_and_moving_tracks(step_to_force_analyze, time_window_search, max_distance_search, debug_mode)
                    except Exception:
                        print("No resting tracks to match")

                    try:
                        video_tracked.assemble_unmatched_moving_tracks(step_to_force_analyze, time_window_search, max_distance_search, debug_mode)
                    except Exception:
                        print("No moving tracks to match")

                    if trial == MAX_NB_ASSEMBLING - 1:  # Finished last round
                        video_tracked.clean_tracks(step_to_force_analyze, time_window_search, max_distance_search, debug_mode)

                    time_window_search *= 1.1
                    max_distance_search *= 1.2

                # Assemble forward backward tracks # Step 5
                video_tracked.assemble_tracks_ids(step_to_force_analyze, debug_mode)

                try:
                    video_tracked.extract_complete_trajectories_from_video_V2(debug_mode)
                except Exception as error:
                    print("An exception occurred completing tracks:", error)
                try:
                    video_tracked.extract_mosquito_population_variables()
                except Exception as error:
                    print("An exception occurred in computing population var (fraction flying etc):", error)
                try:
                    video_tracked.extract_mosquito_individual_variables()
                except Exception as error:
                    print("An exception occurred in computing individual variables (flight speed etc):", error)

                try:
                    video_tracked.extract_mosquito_resting_variables()
                except Exception as error:
                    print("An exception occurred in extracting resting variables results:", error)
                
                try:
                    video_tracked.save_tracking_results()
                except Exception as error:
                    print("An exception occurred in saving results:", error)

                    #video_tracked.extract_complete_trajectories_from_video(debug_mode=0)


                #print(f"Finished analyzing {video_name}")

                # Reset stdout
                sys.stdout = old_stdout

        else:
            print("background image missing")

    def display_video_final_tracking(self,video_name,starting_frame,time_btw_frames):

        #video_name = os.path.splitext(self.list_videos_files[video_idx])[0] # Load video_name
        video_tracked = single_video_analysis(self,video_name,debug_mode=0) # Initialize a video analysis object.

        with open(self.folder_final+"/forward_mosq_tracks_"+video_name, 'rb') as f:
            mosquito_tracks = pickle.load(f)

        # Initialiaze video
        cap = cv2.VideoCapture(video_tracked.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        n_frame = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
        f_i = starting_frame

        while True:
            suc,frame = cap.read()
            time.sleep(time_btw_frames)

            if suc == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = frame.copy()

                if f_i > 1:
                    frame_idx = f_i

                    for k,id in enumerate(mosquito_tracks.objects.keys()):
                        t_start = mosquito_tracks.objects[id]["start"]
                        t_end = mosquito_tracks.objects[id]["end"]

                        if frame_idx > t_start and frame_idx < t_end:
                            t_relative = frame_idx - t_start+2

                            try:
                                centroid = mosquito_tracks.objects[id]["coordinates"][t_relative]
                                text = "id {}".format(id)
                                if mosquito_tracks.objects[id]["state"][t_relative]==0:
                                    color = (0, 0, 255)
                                else:
                                    color = (255, 0, 0)
                                cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 2, color, 1)
                                cv2.putText(img, text, (int(centroid[0])-20, int(centroid[1])-20 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                            except Exception:
                                print("error display")


                cv2.imshow("frame",img)
                f_i += 1

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if f_i>n_frame-2:
                    break
        cap.release()
        cv2.destroyAllWindows()



########################### Compute activity and trajectories from the video. ###################
    def concatenate_flight_activity_(self):
        print("assembling flight activities of all videos")

        files_tracking = [f for f in listdir(self.folder_final) if isfile(join(self.folder_final, f)) and f.startswith("forward")]
        files_tracking.sort()

        for i,file_name in enumerate(files_tracking):
            if i <10000:
                progress_bar(i, len(files_tracking), bar_length=20)
                #print(file_name)
                with open(self.folder_final+"/"+file_name, 'rb') as f:
                    mosquito_tracks = pickle.load(f)

                    if 'fly' in locals():
                        fly = pd.concat([fly,mosquito_tracks.nb_mosquitos_flying])
                    else:
                        fly = mosquito_tracks.nb_mosquitos_flying

        self.global_flight_activity = fly
        exp_object_path = self.folder_temp_data+"/temp_data_"+self.experiment_alias+".pkl"
        with open(exp_object_path, 'wb') as f:
            pickle.dump(self,f)

########################### Compute activity and trajectories from the video. ###################
    def plot_flight_activity_(self):       
        
        fly = self.global_flight_activity 

        fig, ax = plt.subplots(1, 1,dpi=100)
        fig.set_figheight(5)
        fig.set_figwidth(30)
        
        ax.plot(fly.rolling(25*1200,min_periods=1).mean())
        ax.set_ylim([0,0.2])

        plt.ioff()
        plt.savefig(self.folder_analysis+"plots_trajectories/__"+self.video_name+"_plot_ID_"+str(id)+'.png',bbox_inches='tight')

        #plt.show()

########################### Define the precise angle of the image ###################
    def user_input_draw_borders_cage(self, force_to_redo):
        if force_to_redo == 1:
            images_mortality_folder = os.path.join(self.folder_analysis, "images_mortality")
            mortality_images = [f for f in os.listdir(images_mortality_folder) if f.endswith('.png')]
            mortality_images.sort()
            if not mortality_images:
                print("No images found in images_mortality folder.")
                return

            background_image_path = os.path.join(images_mortality_folder, mortality_images[0])

            points = draw_parallelogram(background_image_path)
            self.settings["cage_border_points"] = points

            if os.path.isfile(self.settings_file):
                with open(self.settings_file, 'w') as file:
                    yaml.dump(self.settings, file)


    def user_input_draw_sugar_feeding(self, force_to_redo):
        if force_to_redo == 1:
            images_mortality_folder = os.path.join(self.folder_analysis, "images_mortality")
            mortality_images = [f for f in os.listdir(images_mortality_folder) if f.endswith('.png')]
            mortality_images.sort()
            if not mortality_images:
                print("No images found in images_mortality folder.")
                return

            background_image_path = os.path.join(images_mortality_folder, mortality_images[0])

            points = draw_parallelogram(background_image_path)
            self.settings["sugar_border_points"] = points

            if os.path.isfile(self.settings_file):
                with open(self.settings_file, 'w') as file:
                    yaml.dump(self.settings, file)


    def user_input_draw_control_squares(self, force_to_redo):
        if force_to_redo == 1:
            # Select an image from the images_mortality folder
            images_mortality_folder = os.path.join(self.folder_analysis, "images_mortality")
            mortality_images = [f for f in os.listdir(images_mortality_folder) if f.endswith('.png')]
            mortality_images.sort()
            if not mortality_images:
                print("No images found in images_mortality folder.")
                return

            # Use the first image as the background
            background_image_path = os.path.join(images_mortality_folder, mortality_images[0])
            
            points = draw_parallelogram(background_image_path)
            # Save the points to YAML file
            self.settings["control_border_points"] = points

            if os.path.isfile(self.settings_file):
                # Save a hard copy of the updated settings
                with open(self.settings_file, 'w') as file:
                    yaml.dump(self.settings, file)

    def user_input_draw_control_squares_3(self, force_to_redo):
        if force_to_redo == 1:
            # Select an image from the images_mortality folder
            images_mortality_folder = os.path.join(self.folder_analysis, "images_mortality")
            mortality_images = [f for f in os.listdir(images_mortality_folder) if f.endswith('.png')]
            mortality_images.sort()
            if not mortality_images:
                print("No images found in images_mortality folder.")
                return

            # Use the first image as the background
            background_image_path = os.path.join(images_mortality_folder, mortality_images[0])
            
            points = draw_parallelogram(background_image_path)
            # Save the points to YAML file
            self.settings["square_3_border_points"] = points

            if os.path.isfile(self.settings_file):
                # Save a hard copy of the updated settings
                with open(self.settings_file, 'w') as file:
                    yaml.dump(self.settings, file)

    def user_input_draw_control_squares_4(self, force_to_redo):
        if force_to_redo == 1:
            # Select an image from the images_mortality folder
            images_mortality_folder = os.path.join(self.folder_analysis, "images_mortality")
            mortality_images = [f for f in os.listdir(images_mortality_folder) if f.endswith('.png')]
            mortality_images.sort()
            if not mortality_images:
                print("No images found in images_mortality folder.")
                return

            # Use the first image as the background
            background_image_path = os.path.join(images_mortality_folder, mortality_images[0])
            
            points = draw_parallelogram(background_image_path)
            # Save the points to YAML file
            self.settings["square_4_border_points"] = points

            if os.path.isfile(self.settings_file):
                # Save a hard copy of the updated settings
                with open(self.settings_file, 'w') as file:
                    yaml.dump(self.settings, file)

                
########################### Define the precise angle of the image ###################
    def plot_all_borders(self):

        with open(self.settings_file, 'r') as file:
                settings = yaml.safe_load(file)
                self.settings = settings

        images_mortality_folder = os.path.join(self.folder_analysis, "images_mortality")
        mortality_images = [f for f in os.listdir(images_mortality_folder) if f.endswith('.png')]
        mortality_images.sort()
        if not mortality_images:
            print("No images found in images_mortality folder.")
            return

        # Use the first image as the background
        background_image_path = os.path.join(images_mortality_folder, mortality_images[0])

        def plot_square(color,points,image):
            cv2.line(image, points[0], points[1], color, 2)
            cv2.line(image, points[1], points[2], color, 2)
            cv2.line(image, points[2], points[3], color, 2)
            cv2.line(image, points[3], points[0], color, 2)

        image = cv2.imread(background_image_path)
        a = self.settings["cage_border_points"]
        cage_borders_points = [tuple(a[i]) for i in np.arange(4)]
        plot_square((0,255,0),cage_borders_points,image)

        try :
            a = self.settings["sugar_border_points"]
            sugar_border_points = [tuple(a[i]) for i in np.arange(4)]
            plot_square((255,0,0),sugar_border_points,image)
        except Exception:
            print("no coordinates for sugar feeder area")

        try:
            a = self.settings["control_border_points"]
            control_border_points = [tuple(a[i]) for i in np.arange(4)]
            plot_square((0,0,255),control_border_points,image)
        except Exception:
            print("no coordinates for control feeder area")

        try:
            a = self.settings["square_3_border_points"]
            control_border_points = [tuple(a[i]) for i in np.arange(4)]
            plot_square((0,0,255),control_border_points,image)
        except Exception:
            print("no coordinates for square_3")

        try:
            a = self.settings["square_4_border_points"]
            control_border_points = [tuple(a[i]) for i in np.arange(4)]
            plot_square((0,0,255),control_border_points,image)
        except Exception:
            print("no coordinates for square_4")



            # Display the final image with the parallelogram
        cv2.imwrite(self.folder_analysis+"/background_with_borders.png",image)
        



########################### Define the precise angle of the image ###################
    def extract_images(self,BATCH_NB):
        print("Extracting one image per video")
        create_folder(self.folder_analysis + "individual_images_"+str(BATCH_NB)+"/")
        for k, video_file in enumerate(self.list_videos_files):
            progress_bar(k, len(self.list_videos_files), bar_length=20)
            if video_file.endswith('.mp4'):
                # Save each frame as a PNG file only if it doesn't exist already
                frame_filename = video_file.replace('.mp4', '.png')
                output_path = os.path.join(self.folder_analysis, "individual_images_"+str(BATCH_NB), frame_filename)
                if not os.path.exists(output_path):
                    #start = video_file.find('Cage')
                    video_path = self.folder_videos + video_file  # [start::]
                    cap = cv2.VideoCapture(video_path)
                    suc, frame = cap.read()
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cap.release()


                    
                    
                    cv2.imwrite(output_path, frame_gray)

        return
    
    def extract_images_v2(self,force_to_rerun):
        print("Extracting one image per video")
        create_folder(self.folder_analysis + "/individual_images/")
        for k, video_file in enumerate(self.list_videos_files):
            progress_bar(k, len(self.list_videos_files), bar_length=20)
            if video_file.endswith('.mp4') and video_file.startswith('Cage'):
                # Save each frame as a PNG file only if it doesn't exist already
                frame_filename = video_file.replace('.mp4', '.png')
                output_path = os.path.join(self.folder_analysis, "individual_images", frame_filename)
                print(output_path)
                if not os.path.exists(output_path) or force_to_rerun==1 :
                    #start = video_file.find('Cage')
                    video_path = self.folder_videos + "/"+video_file  # [start::]
                    print(video_path)
                    cap = cv2.VideoCapture(video_path)
                    suc, frame = cap.read()
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cap.release()

                    cv2.imwrite(output_path, frame_gray)

        return  # No need to return images_to_av if not used elsewhere
    

    def extract_average_background(self,force_to_rerun):
                # Average images
        create_folder(self.folder_analysis + "/images_mortality/")

        

        logger = MultiLogger(self.log, None)
        old_stdout = sys.stdout
        sys.stdout = logger

        size_window = 50 ##### Can change
        print("Computing the median frame of 100-frame moving window over time")
        force_to_redo = 1

        image_folder = self.folder_analysis + "/individual_images/"
        all_images = [cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE)
                    for filename in sorted(os.listdir(image_folder)) if filename.endswith('.png')]
        
        all_names = [filename[:-4] for filename in sorted(os.listdir(image_folder)) if filename.endswith('.png')]

        for s, image in enumerate(all_images):
            if  size_window < s < len(all_images) - size_window:
                video_name = all_names[s]  # You might need to adjust how you get the video name
                progress_bar(s, len(all_images), bar_length= size_window)

                if os.path.isfile(self.folder_analysis + "/images_mortality/" + video_name + ".png") == False or force_to_rerun == 1:
                    median_frame = np.median(all_images[s -  size_window:s +  size_window], axis=0).astype(np.uint8)
                    cv2.imwrite(self.folder_analysis + "/images_mortality/" + video_name + ".png", median_frame)
            elif len(all_images) -  size_window <= s < len(all_images):
                video_name = all_names[s]
                if os.path.isfile(self.folder_analysis + "/images_mortality/" + video_name + ".png") == False or force_to_rerun == 1:
                    median_frame = np.median(all_images[-  size_window*2:-1], axis=0).astype(np.uint8)
                    cv2.imwrite(self.folder_analysis + "/images_mortality/" + video_name + ".png", median_frame)

            elif 0 <= s <=  size_window:
                video_name = all_names[s]
                if os.path.isfile(self.folder_analysis + "/images_mortality/" + video_name + ".png") == False or force_to_rerun == 1:
                    median_frame = np.median(all_images[0: size_window*2], axis=0).astype(np.uint8)
                    cv2.imwrite(self.folder_analysis + "/images_mortality/" + video_name + ".png", median_frame)

        sys.stdout = old_stdout

    # def extract_images(self):
    #     print("Extracting one image per video")

    #     images_to_av = []
    #     for k,video_file in enumerate(self.list_videos_files):
    #         progress_bar(k, len(self.list_videos_files), bar_length=20)
    #         if video_file.endswith('.mp4'):
    #             #start = video_file.find('Cage')
    #             video_path = self.folder_videos+video_file#[start::]
    #             cap = cv2.VideoCapture(video_path)
    #             suc,frame = cap.read()
    #             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             images_to_av.append(frame_gray)
    #             cap.release()

    #     return images_to_av
    

#################### Extract
    def extract_number_longterm_resting_objects(self,background_image,show_movie,draw_plot):  

        def get_datetime_from_video_name(video_name):
            try:
                s = video_name.find('_raspberrypi_')
                #print(s)
                YY = int(video_name[s-6:s-4])
                MM = int(video_name[s-4:s-2])
                DD = int(video_name[s-2:s])
                HH = int(video_name[s+13:s+15])
                MI = int(video_name[s+15:s+17])
                SS = int(video_name[s+17:s+19])
                VV = int(video_name[s+21:s+23])*1204

                t = datetime(2000+YY,MM,DD,HH,MI,SS)
                t = t + timedelta(seconds=VV)
            except Exception as e:
                #print(e)
                return print("Incorrect name file, cannot find date")

            return t
        
        print("Counting the number of dead mosquitos")
        # Initialize variable
        number_of_dead = []
        time_vid = []

        background = cv2.imread(background_image)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        folder_images = self.folder_analysis+"images_mortality/"

        files_images = [f for f in listdir(folder_images) if isfile(join(folder_images, f)) and f.startswith("Cage") and f.endswith(".png")]
        files_images.sort()
        video_tracked = single_video_analysis(self,0,debug_mode=0)

        with open(self.folder_analysis+"all_names", 'rb') as f:
            all_names = pickle.load(f)

        for s, image_name in enumerate(files_images):    
            progress_bar(s, len(files_images), bar_length=20)
            image = cv2.imread(folder_images+image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            img = image.copy()

            frame_gray = cv2.subtract(background,image)
            frame_gray[frame_gray<0]=0
            centroids_still = video_tracked.get_centroids_still_objects(frame_gray,self.settings["seg_resting"])         
            nb_dead = 0   
            for centroid in centroids_still:
                if video_tracked.point_inside_cage(self.cage_border_points,centroid):
                    cv2.circle(img, (int(centroid[0]), int(centroid[1]) ), 10, (0, 0, 255), 1)
                    nb_dead +=1
            number_of_dead.append(nb_dead)
            #print(get_datetime_from_video_name(all_names[s]))
            
            time_vid.append(get_datetime_from_video_name(all_names[s]))

            if show_movie:
                cv2.imshow("Dead buddy",img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        cv2.destroyAllWindows()

            
        data = {'time': time_vid,
        'dead_count': number_of_dead}
        df = pd.DataFrame(data)

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        if draw_plot:
            plt.plot(df)
            plt.savefig(self.folder_analysis+"mortality.png",bbox_inches='tight')
            plt.show()
        with open(self.folder_analysis+self.experiment_alias+"_count_dead_mosquito", 'wb') as f:
            pickle.dump(df,f)


    def plot_sample_flight_trajectories_from_video(self,axes,mosquito_tracks):
        video_name = mosquito_tracks.video_name
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
                    if t_f-t_i>25*10:
                        #print(mosquito_tracks.time_stamp[mosquito_tracks.objects[id]["start"]])

                        if nb_plotted < 18 and np.mean(dist)>5:
                            axes[nb_plotted] = self.plot_flight_trajectory(x[t_i+1:t_f],y[t_i+1:t_f],axes[nb_plotted],video_name)
                            nb_plotted += 1
                            
        return axes

    
    def plot_flight_trajectory(self,x,y,ax,video_name):
        # Set plot and add background
        back_path = self.video_path = os.path.join(self.folder_analysis ,"images_mortality", video_name)+".png"
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
        lc.set_linewidth(1)
        line = ax.add_collection(lc)

        ax.set_xlim([0 ,im.shape[0]])
        ax.set_ylim([0 ,im.shape[1]])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.tight_layout()
        return ax