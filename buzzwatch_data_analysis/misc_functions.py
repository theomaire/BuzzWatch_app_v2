import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from logger import MultiLogger
########################## FUNCTIONS OUTSIDES OF THE CLASS BUT VERY USEFUL ####################


def set_plot_size(BIGGER_SIZE):
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

def create_folder(folder_path):
    try:
    # creating a folder named data
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # if not created then raise error
    except OSError:
            print ('Error: Creating directory '+folder_path)

def get_datetime_from_file_name(video_name):
    YY = int(video_name[7+cor:9+cor])
    MM = int(video_name[9+cor:11+cor])
    DD = int(video_name[11+cor:13+cor])
    HH = int(video_name[26+cor:28+cor])
    MI = int(video_name[28+cor:30+cor])
    SS = int(video_name[30+cor:32+cor])
    VV = int(video_name[34+cor:36+cor])*1200
    t = datetime(2000+YY,MM,DD,HH,MI,SS)
    t = t + timedelta(seconds=VV)
    return t

def progress_bar(current, total, bar_length=20):
#     fraction = current / total

#     arrow = int(fraction * bar_length - 1) * '-' + '>'
#     padding = int(bar_length - len(arrow)) * ' '

#     ending = '\n' if current == total else '\r'

#     print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    ending = '\n' if current == total else '\r'
    #ending = '\r'
    message = f'Progress: [{arrow}{padding}] {int(fraction*100)}%'
    #self.log_func(message.strip(), end='')  # Avoid adding a newline
    print('')
    print(message.strip())  # Print to stdout to also handle in console

#def progress_bar(logger, current, total, bar_length=20):
#    logger.progress(current, total, bar_length)


def get_ids_from_dict(resting_object_traj):
    list_of_columns_names = list(resting_object_traj.keys())
    list_of_ID_still = []
    for col_str in list_of_columns_names:
        if col_str.startswith("ID"):
            str_ID = col_str.partition("_")[2]
            list_of_ID_still.append(int(str_ID))
    return list_of_ID_still


def draw_parallelogram(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Create a copy of the image to draw the parallelogram on
    image_copy = image.copy()
    
    # Initialize a list to store the clicked points
    points = []
    points_list = []
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the clicked point to the list
            points.append((x, y))
            points_list.append([x,y])
            
            # Draw a circle at the clicked point
            cv2.circle(image_copy, (x, y), 3, (0, 255, 0), -1)
            
            # Display the image with the clicked points
            cv2.imshow("Image", image_copy)
            
            # Check if four points have been clicked
            if len(points) == 4:
                # Draw the parallelogram on the image
                cv2.line(image_copy, points[0], points[1], (0, 255, 0), 2)
                cv2.line(image_copy, points[1], points[2], (0, 255, 0), 2)
                cv2.line(image_copy, points[2], points[3], (0, 255, 0), 2)
                cv2.line(image_copy, points[3], points[0], (0, 255, 0), 2)
                
                # Display the final image with the parallelogram
                cv2.imshow("Image", image_copy)
                
                # Print the coordinates of the four points
                for i, point in enumerate(points):
                    print(f"Point {i+1}: {point}")
    
    # Create a window and set the mouse callback function
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)
    
    # Display the image
    cv2.imshow("Image", image)
    
    # Wait for the user to close the window
    cv2.waitKey(0)
    
    # Close all windows
    cv2.destroyAllWindows()

    return points_list

def zero_runs(a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges