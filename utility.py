import numpy as np
import cv2
import joblib
import pickle
from keras.models import Sequential
from keras.models import load_model
import torch
import urllib.request
class Solution:
    def __init__(self,depth_map,areaOfInterest_threshold,forward_threshold):
        self.theta_forward=forward_threshold
        self.grid=np.copy(depth_map)
        self.space_blobs = {}
        self.count = 0
        self.minimum = np.min(depth_map)
        self.areaOfInterest_threshold=areaOfInterest_threshold
        self.inside=False

    def isAreaofInterest(self,i,j):
        if(self.grid[i][j] <= self.minimum+self.areaOfInterest_threshold):
            return True
        return False

    def dfs(self,i, j):
        if i < 0 or j < 0 or i >= len(self.grid) or j >= len(self.grid[0]) or self.grid[i][j] == -1 or (not self.isAreaofInterest(i,j)):
            return
        self.grid[i][j] = 10000

        self.space_blobs[self.count] = np.concatenate((self.space_blobs[self.count], [[i, j]]))

        self.dfs(i + 1, j)
        self.dfs(i - 1, j)
        self.dfs(i, j + 1)
        self.dfs(i, j - 1)

    def identify_spaces(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.isAreaofInterest(i,j):
                    self.space_blobs[self.count] = np.empty([0, 2])
                    self.dfs(i, j)
                    self.count = self.count + 1
        return self.space_blobs

    def identify_spaces_iterative(self):

        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.isAreaofInterest(i,j):
                    #self.space_blobs[self.count] = np.empty([0, 2])
                    self.space_blobs[self.count] = []
                    stack=[]
                    stack.append([i,j])

                    while(len(stack)!=0):
                        cood_x=stack[-1][0]
                        cood_y=stack[-1][1]
                        #print(stack)
                        stack.pop(len(stack)-1)

                        if cood_x < 0 or cood_y < 0 or cood_x >= len(self.grid) or cood_y >= len(self.grid[0]) or self.grid[cood_x][cood_y] == -1 or (not self.isAreaofInterest(cood_x, cood_y)):
                            #print(cood_x,cood_y)
                            continue
                        self.grid[cood_x][cood_y] = 100000
                        #self.space_blobs[self.count] = np.concatenate((self.space_blobs[self.count], [[cood_x, cood_y]]))
                        self.space_blobs[self.count].append([cood_x, cood_y])
                        stack.append([cood_x + 1, cood_y])
                        stack.append([cood_x - 1, cood_y])
                        stack.append([cood_x, cood_y+1])
                        stack.append([cood_x, cood_y-1])

                    self.count = self.count + 1
        return self.space_blobs
    def decide_action(self):
        n=len(self.grid[0])
        centre=n/2

        spaces_identified = self.identify_spaces_iterative()
        if len(spaces_identified)==0:
            return [-1,-1],"unable to decide"

        centroids=[]
        for i in range(len(spaces_identified)):
            centroid = calculate_centroid(spaces_identified[i])

            centroids.append(centroid)

        x_dist=100000000
        centroid_final=[-1,-1]
        for i in range(len(centroids)):
            if np.abs(centroids[i][1]-centre)<x_dist:
                centroid_final=centroids[i]
                x_dist=np.abs(centroids[i][1]-centre)
            '''
            if (centroids[i][1] - centre)**2+(centroids[i][0] - centre)**2 < x_dist:
                centroid_final = centroids[i]
                x_dist = (centroids[i][1] - centre)**2+(centroids[i][0] - centre)**2
            '''

        #this is an issue and needs to be fixed remove loop, use only if
        for i in range(len(centroids)):
            #error possible
            if centroid_final[1]<=centre+self.theta_forward and centroid_final[1]>=centre-self.theta_forward:
                return centroid_final,"go forward"
            elif centroid_final[1]>centre+self.theta_forward:
                return centroid_final,"turn right"
            else:
                return centroid_final,"turn left"

        return "error"
    def decide_action2(self):
        n=len(self.grid[0])
        centre=n/2

        spaces_identified = self.identify_spaces_iterative()
        if len(spaces_identified)==0:
            return [-1,-1],"unable to decide"

        centroids=[]

        for i in range(len(spaces_identified)):
            centroid = self.calculate_weighted_centroid(spaces_identified[i])
            print(centroid)
            centroids.append(centroid)

        x_dist=100000000
        centroid_final=[-1,-1]
        for i in range(len(centroids)):
            if np.abs(centroids[i][1]-centre)<x_dist:
                centroid_final=centroids[i]
                x_dist=np.abs(centroids[i][1]-centre)
            '''
            if (centroids[i][1] - centre)**2+(centroids[i][0] - centre)**2 < x_dist:
                centroid_final = centroids[i]
                x_dist = (centroids[i][1] - centre)**2+(centroids[i][0] - centre)**2
            '''

        #this is an issue and needs to be fixed remove loop, use only if
        for i in range(len(centroids)):
            #error possible
            if centroid_final[1]<=centre+self.theta_forward and centroid_final[1]>=centre-self.theta_forward:
                return centroid_final,"go forward"
            elif centroid_final[1]>centre+self.theta_forward:
                return centroid_final,"turn right"
            else:
                return centroid_final,"turn left"

        return "error"
    def weighted_centroid(self):
        m = len(self.grid)
        n = len(self.grid[0])

        centroid_x = 0
        centroid_y = 0
        pixel_sum=0

        for i in range(m):
            for j in range(n):
                centroid_x = centroid_x + (1/(self.grid[i][j]+0.00000000001)) * i
                centroid_y = centroid_y + (1/(self.grid[i][j]+0.00000000001)) * j
                pixel_sum=pixel_sum+(1/(self.grid[i][j]+0.00000000001))
        centroid_x = centroid_x / pixel_sum
        centroid_y = centroid_y / pixel_sum

        #change it
        return [centroid_x, centroid_y]
    def decide_action_weighted_centroid(self):
        centroid=self.weighted_centroid()
        centre=len(self.grid[0])/2

        if centroid[1] <= centre + self.theta_forward and centroid[1] >= centre - self.theta_forward:
            return centroid, "go forward"
        elif centroid[1] > centre + self.theta_forward:
            return centroid, "turn right"
        else:
            return centroid, "turn left"

        return "error"

    def calculate_weighted_centroid(self,coordinates):
        if len(coordinates) == 0:
            return (-1, -1)

        centroid_x = 0
        centroid_y = 0
        pixel_sum = 0
        for i in range(len(coordinates)):
            centroid_x = centroid_x + (1/(self.grid[coordinates[i][0]][coordinates[i][1]]+0.00000000001)) * coordinates[i][0]
            centroid_y = centroid_y + (1/(self.grid[coordinates[i][0]][coordinates[i][1]]+0.00000000001)) * coordinates[i][1]
            pixel_sum=pixel_sum+(1/(self.grid[coordinates[i][0]][coordinates[i][1]]+0.00000000001))

        centroid_x = centroid_x / pixel_sum
        centroid_y = centroid_y / pixel_sum

        return (centroid_x, centroid_y)


def calculate_centroid(coordinates):
    if len(coordinates)==0:
        return (-1,-1)

    centroid_x=0
    centroid_y=0

    for i in range(len(coordinates)):
        centroid_x=centroid_x+coordinates[i][0]
        centroid_y=centroid_y+coordinates[i][1]

    centroid_x=centroid_x/len(coordinates)
    centroid_y=centroid_y/len(coordinates)

    return (centroid_x,centroid_y)


'''
def identify_space(depth_map):
    standard_deviation=np.std(depth_map)
    mean=np.mean(depth_map)
    minimum=np.min(depth_map)
    print("Mean: ",mean)
    print("Standard Deviation: ",standard_deviation)
    space_coordinates=np.empty(shape=[0,2])
    #print(space_coordinates)
    for i in range(len(depth_map)):
        for j in range(len(depth_map[0])):
            if(depth_map[i][j]<minimum+20):
                cood=np.array([i,j])
                space_coordinates=np.concatenate((space_coordinates,[cood]))

    return space_coordinates
'''
