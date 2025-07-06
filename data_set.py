
"""
Datasets is a class for creating a raw dataset from Nym or RIPE, making it usable for emulating 
the link delays between mix nodes.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import math
import json

class Dataset(object):
    def __init__(self,W,L):
        self.N = W*L
        self.W = W
        self.L = L
        self.P = math.pi
        self.R = 6378.137 #Radius of earth
        self.alpha = 4


    def Omega1(self):
        
        with open('ripe_November_12_2023_cleaned.json') as json_file:
            data0 = json.load(json_file)
        self.n = len(data0)


        for i in range(self.n):
            data0[i]['Omega'] = (1+self.alpha*(np.random.rand(1)[0]))

        with open('ripe_November_12_2023_cleaned_Omega.json','w') as json_file:
            json.dump(data0,json_file)

    def Omega2(self):
        with open('Interpolated_NYM_250_DEC_2023.json') as json_file:
            data = json.load(json_file)
        self.n = len(data)


        for i in range(self.n):
            data[i]['Omega'] = (1+self.alpha*(np.random.rand(1)[0]))
        with open('Interpolated_NYM_250_DEC_2023_Omega.json','w') as json_file:
            json.dump(data,json_file)
    def RIPE(self):
 
        with open('Ripe_dataset.json') as json_file:
            data = json.load(json_file)
        Lon =   []
        Lat =   []
        I_Key = []
        Mix_nodes = []
        Omega = []
            
        for i in range(self.N):
            item = round((np.shape(data)[0]-1)*np.random.rand(1)[0])
            while (item) in Mix_nodes:
                item = round((np.shape(data)[0]-1)*np.random.rand(1)[0])
            Mix_nodes.append(item)
            
            I_Key.append(data[item]['i_key'])
            Lon.append(float(data[item]['longitude']))            
            Lat.append(float(data[item]['latitude'])) 
            Omega.append(data[item]['Omega'])

        NYM_Data = {'lat':Lat, 'lon':Lon, 'i_key':I_Key}
        Latitude = NYM_Data['lat']   #Phi
        Longitude = NYM_Data['lon']  #Teta
        self.Dta = {'lat':Lat, 'lon':Lon}
        M_Latitude = np.matrix(Latitude)
        M_Longtitude = np.matrix(Longitude)
        Phi =90 - M_Latitude

        Theta = M_Longtitude

        Phi = self.P*(Phi/180)

        Theta = self.P*(Theta/180)

        R_E = self.R
        Z = R_E*np.cos(Phi)

        R = R_E*np.sin(Phi)

        STheta = np.sin(Theta)

        CTheta = np.cos(Theta)

        Y = np.multiply(R ,STheta)

        X = np.multiply(R ,CTheta)

        Mix_Location = np.concatenate((X,Y), axis=0)

        Mix_Location = np.concatenate((Mix_Location,Z), axis=0)
        x = Mix_Location[0,:].tolist()[0]
        y = Mix_Location[1,:].tolist()[0]
        z = Mix_Location[2,:].tolist()[0]
        
        Matrix = np.ones((self.N,self.N))
        for i in range(self.N):
            ID1 = i
            for j in range(self.N):
                ID2 = j
                if i == j:
                    Matrix[i,j] = 100000
                else:
                    I_key = data[ID2]['i_key']
                    In_Latency = data[ID1]['latency_measurements'][str(I_key)]
                    delay_distance = int(In_Latency)
                    if delay_distance == 0:
                        delay_distance =1                    
                    Matrix[i,j] = abs(delay_distance)/2000   

        
        return [Lat,Lon] , np.array([x,y,z]), Matrix,Omega     




    def NYM(self):

        with open('Nym_dataset.json') as json_file:
            data = json.load(json_file)
        Lon =   []
        Lat =   []
        I_Key = []
        Mix_nodes = []
        Omega = []
            
            
        for i in range(self.N):
            item = round((np.shape(data)[0]-1)*np.random.rand(1)[0])
            while (item) in Mix_nodes:
                item = round((np.shape(data)[0]-1)*np.random.rand(1)[0])
            Mix_nodes.append(item)
            
            I_Key.append(data[item]['i_key'])
            Lon.append(float(data[item]['longitude']))            
            Lat.append(float(data[item]['latitude'])) 
            Omega.append(data[item]['Omega'])

        NYM_Data = {'lat':Lat, 'lon':Lon, 'i_key':I_Key}
        Latitude = NYM_Data['lat']   #Phi
        Longitude = NYM_Data['lon']  #Teta
        self.Dta = {'lat':Lat, 'lon':Lon}
        M_Latitude = np.matrix(Latitude)
        M_Longtitude = np.matrix(Longitude)
        Phi =90 - M_Latitude

        Theta = M_Longtitude

        Phi = self.P*(Phi/180)

        Theta = self.P*(Theta/180)

        R_E = self.R
        Z = R_E*np.cos(Phi)

        R = R_E*np.sin(Phi)

        STheta = np.sin(Theta)

        CTheta = np.cos(Theta)

        Y = np.multiply(R ,STheta)

        X = np.multiply(R ,CTheta)

        Mix_Location = np.concatenate((X,Y), axis=0)

        Mix_Location = np.concatenate((Mix_Location,Z), axis=0)
        x = Mix_Location[0,:].tolist()[0]
        y = Mix_Location[1,:].tolist()[0]
        z = Mix_Location[2,:].tolist()[0]
        
        Matrix = np.ones((self.N,self.N))
        for i in range(self.N):
            ID1 = i
            for j in range(self.N):
                ID2 = j
                if i == j:
                    Matrix[i,j] = 100000
                else:
                    I_key = data[ID2]['i_key']
                    In_Latency = data[ID1]['latency_measurements'][str(I_key)]
                    delay_distance = int(In_Latency)
                    if delay_distance == 0:
                        delay_distance =1                    
                    Matrix[i,j] = abs(delay_distance)/2000   

        
        return [Lat,Lon] , np.array([x,y,z]), Matrix, Omega  





    

    def convert_to_lat_lon(self,x, y, z):
        radius = 6371  # Earth's radius in kilometers
    
        # Convert Cartesian coordinates to spherical coordinates
        longitude = math.atan2(y, x)
        hypotenuse = math.sqrt(x**2 + y**2)
        latitude = math.atan2(z, hypotenuse)
    
        # Convert radians to degrees
        latitude = math.degrees(latitude)
        longitude = math.degrees(longitude)
    
        return latitude, longitude
    








