# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import pandas as pd
import math


class resting_tracker():

    #state_thresh = 0.0005
    #time_update_moving = 6
    #update_int = 5


    #time = 0

    def __init__(self,settings_track):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.activity = OrderedDict()
        #self.maxDisappeared = self.maxDisappeared
        self.previous_centroid = OrderedDict()

        self.max_numb_object = settings_track["max_numb_object"]
        self.maxDisappeared = settings_track["maxDisappeared"]
        self.maxdisttracking = settings_track["maxdisttracking"]


    def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        #self.activity[self.nextObjectID] = state
        #self.previous_centroid[self.nextObjectID] = centroid
        #print("ID "+ str(self.nextObjectID) + " was registered")
        self.nextObjectID += 1
        return self.nextObjectID-1
        

    def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
        centroid = self.objects[objectID]
        del self.objects[objectID]
        del self.disappeared[objectID]

        return centroid
        #del self.activity[objectID]
        #del self.previous_centroid[objectID]
        #print("ID "+ str(objectID) + " was deregistered")

    def update(self, centroids_still): #need both the centroids on the states
        new_IDs = []
        lost_IDs = []
        lost_objects = []
        # self.time +=1
        # if self.time > self.time_update_moving:
        #     self.time = 0
        ################ No objects to track ###############################
        if len(centroids_still) == 0: # if no objects
			# loop over any existing tracked objects and mark them
			# as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1 # if we have reached a maximum number of consecutive frames where a given object has been marked asmissing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects,new_IDs,lost_IDs,lost_objects # return early as there are no centroids or tracking info to update
        
        # for s in np.arange(len(States)):
        #     if States[s]>self.state_thresh:
        #         States[s] = 1
        #     else:
        #         States[s] = 0
        ################ Trackers is still empty ###############################
        if len(self.objects) == 0: #if we are currently not tracking any objects take the input centroids and register each of them
            for i in range(0, len(centroids_still)):
                #if States[i] <10: # if initially not a moving object
                new_ID = self.register(centroids_still[i])
                new_IDs.append(new_ID)

        ################ Trackers is NOT empty, already tracking objects ###############################
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            #objectPreviousCentroids = list(self.previous_centroid.values())
            # compute the distance between each pair of object
            X_t = np.array(objectCentroids)
            D = dist.cdist(X_t, centroids_still)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                #print(dist.cdist(np.array(inputCentroids[col]).reshape(1,-1),np.array(self.objects[objectIDs[row]]).reshape(1,-1)))
                if dist.cdist(np.array(centroids_still[col]).reshape(1,-1),np.array(self.objects[objectIDs[row]]).reshape(1,-1))<self.maxdisttracking: ### Maximum distance
                    #if States[col] == self.activity[objectIDs[row]]: #if states are matching (both moving or both resting)
                    objectID = objectIDs[row]
                    #self.previous_centroid[objectID] = self.objects[objectID]
                    self.objects[objectID] = centroids_still[col]
                    self.disappeared[objectID] = 0
                    #self.activity[objectID] = States[col]
            # indicate that we have examined each of the row and
            # column indexes, respectively
                    usedRows.add(row)
                    usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            #print(len(self.objects))
            #print(D.shape[0])
            #print(D.shape[1])
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        centroid = self.deregister(objectID)
                        lost_IDs.append(objectID)
                        lost_objects.append(centroid)


            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            
            else:
                for col in unusedCols:
                    if len(self.objects) <self.max_numb_object:# and self.time == self.update_int:
                        D = dist.cdist(np.array(centroids_still[col]).reshape(1,-1), objectCentroids)
                        if D[0].min(axis=0) > self.maxdisttracking : # if far enough and not moving
                            new_ID = self.register(centroids_still[col])
                            new_IDs.append(new_ID)
        # return the set of trackable objects


        # ## Remove ID than are disappeared
        # objectIDs = list(self.objects.keys())
        # for objectID in objectIDs:
        #     if self.disappeared[objectID] > self.maxDisappeared:
        #         centroid = self.deregister(objectID)
        #         lost_IDs.append(objectID)
        #         lost_objects.append(centroid)

        return self.objects,new_IDs,lost_IDs,lost_objects
