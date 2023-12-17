
import functools,operator,os,pandas as pd,math,random, numpy as np, copy, matplotlib.pyplot as plt, itertools
from tqdm import tqdm
from scipy.spatial import distance_matrix
from operator import itemgetter
import csv
from functools import reduce
import warnings
from itertools import chain
warnings.filterwarnings("ignore")
random.seed(0)


# Loop that reads the data for Baltic, WAF and Mediterranean
# and executes the gridsearch and service creation for the flow-second stage
for inst in range(3):
    
    # Batlic instance
    if inst==0:
        name="BAL"

        # Read the data from LINER-LIB
        baltic_demand = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/Demand_Baltic.csv", delimiter='\t')
        ports = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/ports.csv", delimiter='\t')
        distances = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/dist_dense.csv", delimiter=',')
        fleet_baltic = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/fleet_Baltic.csv", delimiter=';')
        fleet_data = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/fleet_data.csv", delimiter=';')

        # The fleet data stored seperately
        OGship=copy.deepcopy(fleet_data) 

        # Unique port names in baltic_ports
        ports_baltic = baltic_demand.drop_duplicates(subset='Destination').merge(ports, left_on='Destination', right_on='UNLocode', how='inner')
        
        # Find distances between all ports in Baltic
        dist_baltic = distances[(distances['fromUNLOCODe'].isin(ports_baltic.iloc[:,1])) & (distances['ToUNLOCODE'].isin(ports_baltic.iloc[:,1]))]
        dist_baltic = dist_baltic.drop_duplicates(subset=['fromUNLOCODe', 'ToUNLOCODE'])

        # Baltic fleet with corresponding vessel data
        vesseldata_baltic = fleet_baltic.loc[fleet_baltic.index.repeat(fleet_baltic['Quantity'])].reset_index(drop=True).merge(fleet_data, on='Vessel class')
        
        # Distance matirx of the ports
        dist_matrix = dist_baltic.pivot(index='fromUNLOCODe', columns='ToUNLOCODE', values='Distance')
        ports=ports_baltic

        # List with ports that are looped through when constructing services
        tour=np.column_stack((np.arange(len(ports_baltic)),ports["Destination"]))


        # create the ships dataframe that makes sure that all types of vessels are utilized
        shipold = vesseldata_baltic
        ship = pd.DataFrame().reindex_like(shipold)
        unique_counts = shipold["Vessel class"].nunique()
        unique_values_in_column = shipold["Vessel class"].value_counts().reset_index().values.tolist()
        for i in range(len(shipold)):
            for j in range(len(unique_values_in_column)):
                row_=shipold[shipold['Vessel class'] == unique_values_in_column[j][0]].iloc[0]
                
                if len(row_)!=0:
                    
                    if i==0 and ship.loc[0].isna().any():
                        ship.loc[i] = row_
                        index_to_delete = (shipold == row_).all(axis=1).idxmax()
                        shipold = shipold.drop(index_to_delete)
                        unique_values_in_column = shipold["Vessel class"].value_counts().reset_index().values.tolist()

                    elif ship.iloc[i-1,0]!=row_.iloc[0] and ship.loc[i].isna().any():
                        ship.loc[i] = row_
                        index_to_delete = (shipold == row_).all(axis=1).idxmax()
                        shipold = shipold.drop(index_to_delete)
                        unique_values_in_column = shipold["Vessel class"].value_counts().reset_index().values.tolist()
                        break
                    
                    elif len(unique_values_in_column)==1:
                        ship.loc[i] = row_
                        index_to_delete = (shipold == row_).all(axis=1).idxmax()
                        shipold = shipold.drop(index_to_delete)
                        break
                        
        # Estimate the supply of ports based on both supply and demand
        for i in range(len(ports)):

            avgori=baltic_demand.loc[baltic_demand['Origin'] == tour[i][1]]
            if avgori.empty:
                avg1=0
                er01=0
            else:
                avg1=avgori["FFEPerWeek"].mean()
                er01=1
            
            avgdest=baltic_demand.loc[baltic_demand['Destination'] == tour[i][1]] 
            if avgdest.empty:
                avg2=0
                er02=0
            else:
                avg2=avgdest["FFEPerWeek"].mean()
                er02=1

            avg=(avg1+avg2)/(er01+er02)
            ports.iloc[i,2]=avg 


        # List of hubs - ports with more than 20 market orders
        value_counts = {'Origin': pd.Series(baltic_demand['Origin'].value_counts()), 
                    'Destination': pd.Series(baltic_demand['Destination'].value_counts()),
                    'Total':pd.Series(np.add(baltic_demand['Origin'].value_counts(),baltic_demand['Destination'].value_counts()))} 
        Value_count = pd.DataFrame(value_counts)
        hubs=Value_count.loc[Value_count['Total'] >= 20].index.tolist()

    elif inst==1:
        name="WAF"

        # Read the data from LINER-LIB
        baltic_demand = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/Demand_WAF.csv", delimiter='\t')
        ports = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/ports.csv", delimiter='\t')
        distances = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/dist_dense.csv", delimiter=',')
        fleet_baltic = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/fleet_WAF.csv", delimiter=';')
        fleet_data = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/fleet_data.csv", delimiter=';')

        # The fleet data stored seperately
        OGship=copy.deepcopy(fleet_data) 

        # Unique port names in baltic_ports
        ports_baltic = baltic_demand.drop_duplicates(subset='Destination').merge(ports, left_on='Destination', right_on='UNLocode', how='inner')
        
        # Find distances between all ports in Baltic
        dist_baltic = distances[(distances['fromUNLOCODe'].isin(ports_baltic.iloc[:,1])) & (distances['ToUNLOCODE'].isin(ports_baltic.iloc[:,1]))]
        dist_baltic = dist_baltic.drop_duplicates(subset=['fromUNLOCODe', 'ToUNLOCODE'])

        # Baltic fleet with corresponding vessel data
        vesseldata_baltic = fleet_baltic.loc[fleet_baltic.index.repeat(fleet_baltic['Quantity'])].reset_index(drop=True).merge(fleet_data, on='Vessel class')
        
        # Distance matirx of the ports
        dist_matrix = dist_baltic.pivot(index='fromUNLOCODe', columns='ToUNLOCODE', values='Distance')
        ports=ports_baltic

        # List with ports that are looped through when constructing services
        tour=np.column_stack((np.arange(len(ports_baltic)),ports["Destination"]))


        # create the ships dataframe that makes sure that all types of vessels are utilized
        shipold = vesseldata_baltic
        ship = pd.DataFrame().reindex_like(shipold)
        unique_counts = shipold["Vessel class"].nunique()
        unique_values_in_column = shipold["Vessel class"].value_counts().reset_index().values.tolist()
        for i in range(len(shipold)):
            for j in range(len(unique_values_in_column)):
                row_=shipold[shipold['Vessel class'] == unique_values_in_column[j][0]].iloc[0]
                
                if len(row_)!=0:
                    
                    if i==0 and ship.loc[0].isna().any():
                        ship.loc[i] = row_
                        index_to_delete = (shipold == row_).all(axis=1).idxmax()
                        shipold = shipold.drop(index_to_delete)
                        unique_values_in_column = shipold["Vessel class"].value_counts().reset_index().values.tolist()

                    elif ship.iloc[i-1,0]!=row_.iloc[0] and ship.loc[i].isna().any():
                        ship.loc[i] = row_
                        index_to_delete = (shipold == row_).all(axis=1).idxmax()
                        shipold = shipold.drop(index_to_delete)
                        unique_values_in_column = shipold["Vessel class"].value_counts().reset_index().values.tolist()
                        break
                    
                    elif len(unique_values_in_column)==1:
                        ship.loc[i] = row_
                        index_to_delete = (shipold == row_).all(axis=1).idxmax()
                        shipold = shipold.drop(index_to_delete)
                        break

        # Estimate the supply of ports based on both supply and demand
        for i in range(len(ports)):
        #print(baltic_demand.loc[baltic_demand['Origin'] == tour[i][1]])
            avgori=baltic_demand.loc[baltic_demand['Origin'] == tour[i][1]]
            if avgori.empty:
                avg1=0
                er01=0
            else:
                avg1=avgori["FFEPerWeek"].mean()
                er01=1
            
            avgdest=baltic_demand.loc[baltic_demand['Destination'] == tour[i][1]] 
            if avgdest.empty:
                avg2=0
                er02=0
            else:
                avg2=avgdest["FFEPerWeek"].mean()
                er02=1

            avg=(avg1+avg2)/(er01+er02)
            ports.iloc[i,2]=avg 
            

        # List of hubs - ports with more than 20 market orders
        value_counts = {'Origin': pd.Series(baltic_demand['Origin'].value_counts()), 
                    'Destination': pd.Series(baltic_demand['Destination'].value_counts()),
                    'Total':pd.Series(np.add(baltic_demand['Origin'].value_counts(),baltic_demand['Destination'].value_counts()))} 
        Value_count = pd.DataFrame(value_counts)
        hubs=Value_count.loc[Value_count['Total'] >= 20].index.tolist()

    elif inst==2:
        name="MED"
        # Read the data from LINER-LIB
        baltic_demand = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/Demand_Mediterranean.csv", delimiter='\t')
        ports = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/ports.csv", delimiter='\t')
        distances = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/dist_dense.csv", delimiter=',')
        fleet_baltic = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/fleet_Mediterranean.csv", delimiter=';')
        fleet_data = pd.read_csv("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/LINERLIB-master/data/fleet_data.csv", delimiter=';')

        # The fleet data stored seperately
        OGship=copy.deepcopy(fleet_data) 

        # Unique port names in baltic_ports
        ports_baltic = baltic_demand.drop_duplicates(subset='Destination').merge(ports, left_on='Destination', right_on='UNLocode', how='inner')
        
        # Find distances between all ports in Baltic
        dist_baltic = distances[(distances['fromUNLOCODe'].isin(ports_baltic.iloc[:,1])) & (distances['ToUNLOCODE'].isin(ports_baltic.iloc[:,1]))]
        dist_baltic = dist_baltic.drop_duplicates(subset=['fromUNLOCODe', 'ToUNLOCODE'])

        # Baltic fleet with corresponding vessel data
        vesseldata_baltic = fleet_baltic.loc[fleet_baltic.index.repeat(fleet_baltic['Quantity'])].reset_index(drop=True).merge(fleet_data, on='Vessel class')
        
        # Distance matirx of the ports
        dist_matrix = dist_baltic.pivot(index='fromUNLOCODe', columns='ToUNLOCODE', values='Distance')
        ports=ports_baltic

        # List with ports that are looped through when constructing services
        tour=np.column_stack((np.arange(len(ports_baltic)),ports["Destination"]))


        # create the ships dataframe that makes sure that all types of vessels are utilized
        shipold = vesseldata_baltic
        ship = pd.DataFrame().reindex_like(shipold)
        unique_counts = shipold["Vessel class"].nunique()
        unique_values_in_column = shipold["Vessel class"].value_counts().reset_index().values.tolist()
        
        for i in range(len(shipold)):
            for j in range(len(unique_values_in_column)):
                row_=shipold[shipold['Vessel class'] == unique_values_in_column[j][0]].iloc[0]
                
                if len(row_)!=0:
                    
                    if i==0 and ship.loc[0].isna().any():
                        ship.loc[i] = row_
                        index_to_delete = (shipold == row_).all(axis=1).idxmax()
                        shipold = shipold.drop(index_to_delete)
                        unique_values_in_column = shipold["Vessel class"].value_counts().reset_index().values.tolist()

                    elif (ship.iloc[i-1,0]!=row_.iloc[0] and ship.iloc[i-2,0]!=row_.iloc[0]) and ship.loc[i].isna().any():
                        ship.loc[i] = row_
                        index_to_delete = (shipold == row_).all(axis=1).idxmax()
                        shipold = shipold.drop(index_to_delete)
                        unique_values_in_column = shipold["Vessel class"].value_counts().reset_index().values.tolist()
                        break


                    elif ship.iloc[i-1,0]!=row_.iloc[0] and ship.loc[i].isna().any() and len(unique_values_in_column)==2:
                        ship.loc[i] = row_
                        index_to_delete = (shipold == row_).all(axis=1).idxmax()
                        shipold = shipold.drop(index_to_delete)
                        unique_values_in_column = shipold["Vessel class"].value_counts().reset_index().values.tolist()
                        break
                    
                    elif len(unique_values_in_column)==1:
                        ship.loc[i] = row_
                        index_to_delete = (shipold == row_).all(axis=1).idxmax()
                        shipold = shipold.drop(index_to_delete)
                        break

        # Estimate the supply of ports based on both supply and demand
        for i in range(len(ports)):
            avgori=baltic_demand.loc[baltic_demand['Origin'] == tour[i][1]]
            if avgori.empty:
                avg1=0
                er01=0
            else:
                avg1=avgori["FFEPerWeek"].mean()
                er01=1
            
            avgdest=baltic_demand.loc[baltic_demand['Destination'] == tour[i][1]] 
            if avgdest.empty:
                avg2=0
                er02=0
            else:
                avg2=avgdest["FFEPerWeek"].mean()
                er02=1

            avg=(avg1+avg2)/(er01+er02)
            ports.iloc[i,2]=avg 
           
        # List of hubs - ports with more than 20 market orders
        value_counts = {'Origin': pd.Series(baltic_demand['Origin'].value_counts()), 
                    'Destination': pd.Series(baltic_demand['Destination'].value_counts()),
                    'Total':pd.Series(np.add(baltic_demand['Origin'].value_counts(),baltic_demand['Destination'].value_counts()))} 
        Value_count = pd.DataFrame(value_counts)
        hubs=Value_count.loc[Value_count['Total'] >= 20].index.tolist()







    # From https://www.programiz.com/dsa/kruskal-algorithm
    # Kruskal's algorithm 

    class Graph:
        def __init__(self, vertices):
            self.V = vertices
            self.graph = []

        def add_edge(self, u, v, w):
            self.graph.append([u, v, w])

        # Search function

        def find(self, parent, i):
            if parent[i] == i:
                return i
            return self.find(parent, parent[i])

        def apply_union(self, parent, rank, x, y):
            xroot = self.find(parent, x)
            yroot = self.find(parent, y)
            if rank[xroot] < rank[yroot]:
                parent[xroot] = yroot
            elif rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
            else:
                parent[yroot] = xroot
                rank[xroot] += 1

        #  Applying Kruskal algorithm
        def kruskal_algo(self):
            mst=[]
            result = []
            i, e = 0, 0
            self.graph = sorted(self.graph, key=lambda item: item[2])
            parent = []
            rank = []
            for node in range(self.V):
                parent.append(node)
                rank.append(0)
            while e < self.V - 1:
                u, v, w = self.graph[i]
                i = i + 1
                x = self.find(parent, u)
                y = self.find(parent, v)
                if x != y:
                    e = e + 1
                    result.append([u, v, w])
                    self.apply_union(parent, rank, x, y)
            for u, v, weight in result:
                mst.append([u,v,round(weight,2)])
            return mst



    # Distance function that returns the distance of port 'origin' to port ' destination'
    def distance(origin, destination):
        return dist_matrix.loc[origin,destination]

    # Distance function that returns the length of a service
    def distancesubtour(subtour):
        d=0
        for j in range(0,len(subtour)-1):
                axy1 = subtour[j]
                axy2 = subtour[j+1]
                d = d + distance(axy1, axy2)
        return d

    # Function that add ports that have not been inserted in a service, to a service
    def addport(finaltour,index,tour):
        portmiss=tour[index:] # List of ports not in a service

        for i in range(len(portmiss)): # Loops through ports not added to a service
            portsInSer=tour[:index] 
            dist=distance(portmiss[0,1],tour[0][1]) 
            portadd=tour[0][1] 

            for g in range(len(portsInSer)-1): #find the port that the missing port i closest to
                if dist>=distance(portmiss[i,1],tour[g][1]) and portmiss[i,1]!=tour[g][1]: 
                    dist=distance(portmiss[i,1],tour[g][1]) 
                    portadd=tour[g][1] 

            for l in range(len(finaltour)): 
                # Find the service that contains the city closest to the missing service, and insert
                # The missing port in that service such that the added distance is minimized 
                if portadd in finaltour[l]: 
                    idx=finaltour[l].index(portadd) 

                    if len(finaltour[l])>1: 
                        dnext = []
                        for f in range (2):
                            placetour=copy.deepcopy(finaltour)
                            placetour[l].insert(idx+f,portmiss[i,1]) 
                            
                            dnext.append(distancesubtour(placetour[l]))

                        if dnext[0]<=dnext[1]:
                            finaltour[l].insert(idx,portmiss[i,1]) 
                            break
                        else:
                            finaltour[l].insert(idx+1,portmiss[i,1])
                            break
                        
                    elif portadd in finaltour[l]: 
                        finaltour[l].insert(idx,portmiss[i,1]) 
                        break 
        return finaltour


    # Make sure that all services contain at least one hub
    def addhub(finaltour): 
        for i in range(len(finaltour)): # Loops through all services
            dist=distance(hubs[0],finaltour[i][0])
            hu=hubs[0]

            # If the service only contains a hub, then a random port is added to that service
            if any(x in finaltour[i] for x in hubs)==True and len(finaltour[i])==1:
                tempa=copy.deepcopy(tour[:,1])
                tempa=np.delete(tempa, np.where(tempa == finaltour[i][0]))
                randby=np.random.choice(tempa)
                finaltour[i].append(randby)
            
            # If there is not a hub in the service, then the hub closest to one of the ports in the service is added
            # such that the added distance of adding the hub to the service is minimized
            if any(x in finaltour[i] for x in hubs)==False:
                idx=0
                for k in range(len(finaltour[i])): 
                    for j in range(len(hubs)): 
                        if dist>distance(hubs[j],finaltour[i][k]): 
                            dist=distance(hubs[j],finaltour[i][k]) 
                            hu=hubs[j] 
                            idx=k
                    dnext = []
                    
                    for f in range (2):
                        placetour=copy.deepcopy(finaltour)
                        placetour[i].insert(idx+f,hu) 
                        dnext.append(distancesubtour(placetour[i]))

                    if dnext[0]<=dnext[1]:
                        finaltour[i].insert(idx,hu)
                        break
                    else:
                        finaltour[i].insert(idx+1,hu)
                        idx+=1
                        break

                # Make sure that the service is cyclic around the hub and add the type of vessel that sails the service as the first element   
                finaltour[i]=[ship.iloc[i,0]]+finaltour[i][idx:]+finaltour[i][:idx+1]

            else:
                idx=finaltour[i].index(list(set(finaltour[i])&set(hubs))[0])
                finaltour[i]=[ship.iloc[i,0]]+finaltour[i][idx:]+finaltour[i][:idx+1]
        return finaltour

    # Create common ports in the services such that it is possible to go from one service to any other - i.e. transhipment
    def transship(finaltour): 
        xy=[]
        varnames=[]

        # Find the average coordinates of services - (Latitude,Longitude)
        for i in range(len(finaltour)):
            x=0
            y=0
            for k in range(1,len(finaltour[i])): # loop over subturerne og finder average af x og y
                var="Service " + str(i)
                x+=ports_baltic.loc[ports_baltic['Destination'] == finaltour[i][k], 'Latitude'].item() # x
                y+=ports_baltic.loc[ports_baltic['Destination'] == finaltour[i][k], 'Longitude'].item()
            varnames.append(var)
            xy.append([np.round(x/len(finaltour[i]),2),np.round(y/len(finaltour[i]),2)])
        
        # Find the distance between the average coordinates of the services
        avgdist=pd.DataFrame(distance_matrix(xy, xy, p=2))
        avgdist.columns=varnames
        avgdist.index=varnames

        # Defining a graph consisting of the distances between the average coordinates of the services
        g=Graph(len(finaltour))
        j=1
        edges=[]

        # Defines the distance as wieghts on the edges
        for i in range(len(finaltour)-1):
            for j in reversed(range(len(finaltour)-1-i)):
                edges.append([i,(len(finaltour)-1)-j,round(avgdist.iloc[i:,(len(finaltour)-1)-j].iloc[0],2)])

            
        # Adds edges to the graph
        edges=sorted(edges, key=itemgetter(2))
        for i in range(len(edges)):
            g.add_edge(edges[i][0], edges[i][1], edges[i][2])
        
        # Calculates the MST using Kruskal's algorithm
        mst=g.kruskal_algo()
        for h in range(len(mst)):
            
            # Checks if there exists common ports in the services Based on the edges from the MST. 
            # If they are conneced through common port - transshipment is possible
            if any(i in finaltour[mst[h][0]] for i in finaltour[mst[h][1]]):
                continue
                
            # If we need to insert a port such that transshipment is possible
            else:   
                # Initialize variables for distance and ports from the two services that needs to be connected        
                dist=distance(finaltour[mst[h][0]][1],finaltour[mst[h][1]][1]) 
                portclose1=finaltour[mst[h][0]][1] 
                portclose2=finaltour[mst[h][1]][1] 
            
                # Find the two ports in the two services that lies the closest to eachother
                for g in range(1,len(finaltour[mst[h][0]])):
                    for e in range(1,len(finaltour[mst[h][1]])):

                        if distance(finaltour[mst[h][0]][g],finaltour[mst[h][1]][e])<dist:
                            dist=distance(finaltour[mst[h][0]][g],finaltour[mst[h][1]][e])
                            portclose1=finaltour[mst[h][0]][g]
                            portclose2=finaltour[mst[h][1]][e]
                
                portclose=[portclose1,portclose2] 
                dfinal = []
                ind=0

                # Find which port that needs to be inserted in which service such that the added distance is minimized
                for t in range(2): 
                    if any(x in portclose[t] for x in hubs)==True:
                        ind=[2,-1] 
                        dnext = []
                        dnext.append(distancesubtour(finaltour[mst[h][t]][1:]))
                        whichplacetour=copy.deepcopy(finaltour)

                        for f in range (2):
                            placetour=copy.deepcopy(whichplacetour)
                            placetour[mst[h][t]].insert(ind[f],portclose[1-t]) 
                            dnext.append(distancesubtour(placetour[mst[h][t]][1:])-dnext[0])
                        dfinal.append([min(dnext[1:]),ind[dnext.index(min(dnext[1:]))-1]]) 
            
                    else:
                        ind=finaltour[mst[h][t]].index(portclose[t]) 
                        dnext = []
                        dnext.append(distancesubtour(finaltour[mst[h][t]][1:]))
                        whichplacetour=copy.deepcopy(finaltour)
                        
                        for f in range (2):
                            placetour=copy.deepcopy(whichplacetour)
                            placetour[mst[h][t]].insert(ind+f,portclose[1-t]) 
                            dnext.append(distancesubtour(placetour[mst[h][t]][1:])-dnext[0])
                        dfinal.append([min(dnext[1:]),ind+dnext.index(min(dnext[1:]))-1]) 

                if dfinal[0][0]<=dfinal[1][0]:
                    finaltour[mst[h][0]].insert(dfinal[0][1],portclose[1])

                else:
                    finaltour[mst[h][1]].insert(dfinal[1][1],portclose[0])

        return finaltour

    # Function that creates the initial services, and calls the functions: addport(finaltour,index,tour), addhub(finaltour)
    # and transship(finaltour), in order to finish them
    def subtourslice(tour, ship):

        capacityused = np.zeros((len(ship),2)) # List that keeps track of the ships capacity
        k = 0 # Variable that allows for keeping track of which port that is investigated
        finaltour=[] # The list of final services
        capplace=ports.copy() # Dataframe that keeps track of how much supply the port have left

        for i in range(len(ship)): # Loops through the available vessels
            subtour=[] # List that stores the services begun by a vessels before they are added to list of final services

            while k <= len(tour): # Iterates as long as all ports are not added to a services
                maxdist=fleet_data.loc[fleet_data['Vessel class'] == ship.iloc[i,0],'designSpeed'].iloc[0]*(7-len(subtour))*24 # Max dist vessel[i] can travel

                # Case 1
                if k==len(tour): # If all ports are in a services
                    if len(subtour)!=0:
                        finaltour.append(subtour)
                    break
                
                # Case 2
                elif len(subtour)<1 and ship.iloc[i,2]-capacityused[i,0]>=capplace.iloc[tour[k][0],2]:
                    capacityused[i,0] += capplace.iloc[tour[k][0],2]
                    capplace.iloc[tour[k][0],2]=0
                    subtour.append(tour[k][1]) 
                    k = k + 1
                    continue

                # Case 3
                elif len(subtour)>=1 and ship.iloc[i,2]-capacityused[i,0]>=capplace.iloc[tour[k][0],2] and capacityused[i,1]+distance(tour[k-1][1], tour[k][1])<=maxdist: # tjek om skibet kan have al kapaciteten med
                    capacityused[i,0] += capplace.iloc[tour[k][0],2]
                    capacityused[i,1] += distance(tour[k-1][1], tour[k][1]) 
                    capplace.iloc[tour[k][0],2]=0
                    subtour.append(tour[k][1]) 
                    k = k + 1 
                    continue

                # Case 4
                elif len(subtour)<1 and capacityused[i,0]+capplace.iloc[tour[k][0],2] > ship.iloc[i,2]: 
                    capplace.iloc[tour[k][0],2]-=(ship.iloc[i,2]-capacityused[i,0]) 
                    capacityused[i,0]=ship.iloc[i,2] 
                    subtour.append(tour[k][1]) 
                    finaltour.append(subtour) 
                    break

                # Case 5
                elif len(subtour)>=1 and capacityused[i,0]+capplace.iloc[tour[k][0],2] > ship.iloc[i,2] and capacityused[i,1]+distance(tour[k-1][1], tour[k][1])<=maxdist:
                    capplace.iloc[tour[k][0],2]-=(ship.iloc[i,2]-capacityused[i,0]) 
                    capacityused[i,0]=ship.iloc[i,2] 
                    capacityused[i,1] += distance(tour[k-1][1], tour[k][1]) 
                    subtour.append(tour[k][1]) 
                    finaltour.append(subtour) 
                    break
                
                # Case 6
                elif len(subtour)>=1 and capacityused[i,1]+distance(tour[k-1][1], tour[k][1])>=maxdist:
                    finaltour.append(subtour)
                    break

        # Index that keeps track of last city added - necessary if not all ports are part of a service      
        index=int(np.where(tour[:,1]==finaltour[-1][-1])[0]+1)

        # Calls the functions below in order to finish the services
        addport(finaltour,index,tour)
        addhub(finaltour)
        transship(finaltour)
        
        return finaltour

    # Function that calculates the total costs of all services constructed and contained in finaltour (finalservices in the thesis)
    def subtourcost(finaltour):

        cost=0
        for i in range(len(finaltour)):

            # Fixed costs of one vessel of a specific vesseltype on a weekly basis
            fcost_one_ship=OGship.loc[OGship['Vessel class'] == finaltour[i][0],'TC rate daily (fixed Cost)'].iloc[0]*7 
            d=0 # Distance of the service
            port=0 # Fixed portcall cost

            # Calculate the distance and total fixed portcall cost of the service
            for k in range(1,len(finaltour[i])-1):
                axy1 = finaltour[i][k]
                axy2 = finaltour[i][k+1] 
                d = d + distance(axy1, axy2)
                port=port+ports_baltic.loc[ports_baltic['Destination']==finaltour[i][k+1],'PortCallCostFixed'].iloc[0]
            
            # Calculate cost of bunker fuel when not idling
            ton_hour_des=OGship.loc[OGship['Vessel class'] == finaltour[i][0],'Bunker ton per day at designSpeed'].iloc[0]/24
            ton_knot=ton_hour_des/OGship.loc[OGship['Vessel class'] == finaltour[i][0],'designSpeed'].iloc[0]
            ton_sub=ton_knot*d
            fuel=ton_sub*600

            # Calculate distace of service where the distance the vesseltype of the service can travel 
            # in 24 hours are added for each port that the service visits
            shipdistplace=d+((len(finaltour[i])-2)*(OGship.loc[OGship['Vessel class'] == finaltour[i][0],'designSpeed'].iloc[0]*24))
            
            # Distance the given vessel type of the service can travel in a week
            distship=OGship.loc[OGship['Vessel class'] == finaltour[i][0],'designSpeed'].iloc[0]*7*24

            # Calculate how many vessels that needs to sail on the service in order for all ports to be visited once a week
            no_ships=math.ceil(shipdistplace/distship)

            # Portcall costs on the service
            portcost=port # portcost for hver service
            
            # Cost of idling at ports for specific vesseltype
            idlecost=OGship.loc[OGship['Vessel class'] == finaltour[i][0],'Idle Consumption ton/day'].iloc[0]*600*(len(finaltour[i])-2)

            # Calculates total cost
            cost+=round(portcost+(fcost_one_ship*no_ships)+fuel+idlecost,2)# fuel cost pr. bunker ton = 600 usd, distnace for subtur
        return cost

    # Calculate fixed cost and the number of vessels for each service
    def fixed_cost_services(finaltour):
        
        for i in range(len(finaltour)):
            cost=0

            # Fixed costs of one vessel of a specific vesseltype on a weekly basis
            fcost_one_ship=OGship.loc[OGship['Vessel class'] == finaltour[i][0],'TC rate daily (fixed Cost)'].iloc[0]*7 
            d=0 # Distance of the service
            port=0 # Fixed portcall cost

            # Calculate the distance and total fixed portcall cost of the service
            for k in range(1,len(finaltour[i])-1):
                axy1 = finaltour[i][k]
                axy2 = finaltour[i][k+1] 
                d = d + distance(axy1, axy2)
                port=port+ports_baltic.loc[ports_baltic['Destination']==finaltour[i][k+1],'PortCallCostFixed'].iloc[0]

            # Calculate cost of bunker fuel when not idling
            ton_hour_des=OGship.loc[OGship['Vessel class'] == finaltour[i][0],'Bunker ton per day at designSpeed'].iloc[0]/24
            ton_knot=ton_hour_des/OGship.loc[OGship['Vessel class'] == finaltour[i][0],'designSpeed'].iloc[0]
            ton_sub=ton_knot*d
            fuel=ton_sub*600

            # Calculate distace of service where the distance the vesseltype of the service can travel 
            # in 24 hours are added for each port that the service visits
            shipdistplace=d+((len(finaltour[i])-2)*(OGship.loc[OGship['Vessel class'] == finaltour[i][0],'designSpeed'].iloc[0]*24))
            
            # Distance the given vessel type of the service can travel in a week
            distship=OGship.loc[OGship['Vessel class'] == finaltour[i][0],'designSpeed'].iloc[0]*7*24

            # Calculate how many vessels that needs to sail on the service in order for all ports to be visited once a week
            no_ships=math.ceil(shipdistplace/distship)

            # Portcall costs on the service
            portcost=port # portcost for hver service
            
            # Cost of idling at ports for specific vesseltype
            idlecost=OGship.loc[OGship['Vessel class'] == finaltour[i][0],'Idle Consumption ton/day'].iloc[0]*600*(len(finaltour[i])-2)

            #calculates cost for all subtours
            cost=round(portcost+(fcost_one_ship*no_ships)+fuel+idlecost,2)

            # Add fixed cost to the service
            finaltour[i].insert(0,cost)

            # Add the number of vessels on the service
            finaltour[i].insert(0,no_ships)
        
        return finaltour
    

    # Grid search for tuning the temperature schedule for each instance of the LINER-LIB dataset
    def gridsearch(tour):     
        
        # Defining the grid
        # Initial temperature
        temp0=[10,9,8,7,6,5,4,3,2,1]
        # Final temperature
        tempend=[0,1000,10000,20000,50000]# 

        # Number of iterations of each gridfield in gridsearch
        q=5

        # Number of iterations in the simulated annealing metaheuristic
        n=5000

        # Dataframe containing the average total fixed cost of services for gridfield in the gridsearch
        finalval = pd.DataFrame(0,index=tempend, columns=temp0)

        # Lists for storing the Total fixed costs of each iteration in the simulated annealing metaheuristic
        losninger=[]

        # List that stores the services with minimized total fixed costs for each run of the simulated annealing metaheuristic
        alltours=[]

        # List that stores the initial sequence of ports for each run of the simulated annealing metaheuristic
        allinit=[]
        iter_=0
        random.seed(0)

        #Loops through the gridfields
        for t in range(q):
                for w in range(len(temp0)):
                    for r in range(len(tempend)):
                        
                        # Initializes the algorithm
                        [i,j] = sorted(random.sample(range(len(ports)),2))
                        bestinit=np.concatenate([tour[:i],tour[j:j+1],tour[i+1:j],tour[i:i+1],tour[j+1:]])
                        bestsol=subtourcost(subtourslice(tour, ship))
                        besttour=fixed_cost_services(subtourslice(tour, ship))

                        # Total fixed cost of services in the first iteration
                        existingDistances=subtourcost(subtourslice(tour, ship))
                        
                        print(iter_)
                        iter_+=1

                        # Makes sure that the initial tempereture is higher than the final temperature
                        if 10**temp0[w]+1>tempend[r]:
                            
                            # The simulated annealing metaheuristic
                            for temp in tqdm(np.logspace(tempend[r],temp0[w]+1,num=n)[::-1]): # num tal i intervallet 0,5 - overvej om vi skal bruge linspace
                                
                                # Swap two ports in the sequence of ports
                                [i,j] = sorted(random.sample(range(len(ports)),2)) 
                                updateTour = np.concatenate([tour[:i],tour[j:j+1],tour[i+1:j],tour[i:i+1],tour[j+1:]])
                                
                                # Create new services
                                newtour=copy.copy(subtourslice(updateTour, ship))

                                # Calculate Total fixed cost of new services
                                updateDistances=copy.deepcopy(subtourcost(newtour)) 
                                losninger.append(updateDistances)

                                # Generate random float between 0 and 1 - used in the metropolis function
                                rand=random.random()

                                # If the new solution is better than the current solution
                                # then we accept the new solution as the current solution
                                if updateDistances-existingDistances<0: 
                                    tour = copy.copy(updateTour)
                                    existingDistances=updateDistances

                                    # If the new solution is the best solution found so far, then we save that solution 
                                    if updateDistances-bestsol<0:
                                        bestsol=copy.deepcopy(updateDistances)
                                        besttour=fixed_cost_services(newtour)
                                        bestinit=copy.copy(updateTour)

                                # If we accept a worse solution as a new solution due to the metropolis function
                                elif math.exp(-(abs(existingDistances-updateDistances))/(temp))>=rand: 
                                    tour=copy.copy(updateTour)
                                    existingDistances=updateDistances

                                # If we dont accept a worse solution as a new solution due to the metropolis function
                                elif math.exp(-(abs(existingDistances-updateDistances))/(temp))<=rand:
                                    continue
                            
                            # Store values of the best soltuion
                            allinit.append(np.concatenate(bestinit))
                            finalval.iloc[r,w]=(finalval.iloc[r,w]+bestsol)
                            alltours.append(np.concatenate((np.concatenate(bestinit),np.array(besttour,dtype=object),np.array([finalval.iloc[r,w]]),np.array([t]),np.array([w]),np.array([r]))))
                        
                        # If the final temperature is higher than the initial, then the below is executed
                        # everything added in the else-statement wont be considered further, and is only added due to
                        # compatibility issues
                        else:
                            bestsol=1
                            allinit.append(np.concatenate(bestinit))
                            finalval.iloc[r,w]=(finalval.iloc[r,w]+bestsol)
                            alltours.append(np.concatenate((np.concatenate(bestinit),np.array(besttour,dtype=object),np.array([finalval.iloc[r,w]]),np.array([t]),np.array([w]),np.array([r]))))

        # Find the average total fixed cost of each gridfield
        Rfinalval=finalval/q

        # Store the results
        Rfinalval.to_excel(name+"_output.xlsx")
        DF = pd.DataFrame(alltours)
        ALL=pd.DataFrame(allinit)
        DF.to_excel(name+"_tours.xlsx")
        ALL.to_excel(name+"_allinit.xlsx")
        return

    # Call the gridsearch
    print(gridsearch(tour))




    # Construct the set of services S used in the LS-MCFP
    def finalservice(temp0,tempend,tour):

        # List containing the services
        Finalservice= []

        # All solution from the simulated annealing metaheuristic
        losninger=[]
        iteration=0
        
        # Scaler for constructing scenarios
        antal_serv=[3,6,9]
        random.seed(0)
        
        # Loop that constructs each scenario
        for s in range(len(antal_serv)):
            while len(Finalservice) <= len(tour)*antal_serv[s]:
                
                # Number of iterations in the simulated annealing metaheuristic
                n=5000 

                # Capacity and distance multipliers
                cappercent=round(random.random()+0.5,2)
                distpercent=round(random.random()+0.5,2)
                ship.iloc[:,2] = ship.iloc[:,2]*cappercent
                fleet_data.loc[fleet_data['Vessel class'] == ship.iloc[0,0],'designSpeed']=fleet_data.loc[fleet_data['Vessel class'] == ship.iloc[0,0],'designSpeed']*distpercent

                # Initializes the algorithm
                [i,j] = sorted(random.sample(range(len(ports)),2)) # finder to indeks at random
                bestsol=subtourcost(subtourslice(tour, ship))
                besttour=fixed_cost_services(subtourslice(tour, ship))
                    
                # Total fixed cost of services in the first iteration
                existingDistances=subtourcost(subtourslice(tour, ship))
                
                # The simulated annealing metaheuristic
                for temp in tqdm(np.logspace(tempend,temp0,num=n)[::-1]): # num tal i intervallet 0,5 - overvej om vi skal bruge linspace
                    
                    # Swap two ports in the sequence of ports
                    [i,j] = sorted(random.sample(range(len(ports)),2)) 
                    updateTour = np.concatenate([tour[:i],tour[j:j+1],tour[i+1:j],tour[i:i+1],tour[j+1:]])
                    
                    # Create new services
                    newtour=copy.copy(subtourslice(updateTour, ship))

                    # Calculate Total fixed cost of new services
                    updateDistances=copy.deepcopy(subtourcost(newtour)) 
                    losninger.append(updateDistances)

                    # Generate random float between 0 and 1 - used in the metropolis function
                    rand=random.random()

                    # If the new solution is better than the current solution
                    # then we accept the new solution as the current solution
                    if updateDistances-existingDistances<0: 
                        tour = copy.copy(updateTour)
                        existingDistances=updateDistances

                        # If the new solution is the best solution found so far, then we save that solution 
                        if updateDistances-bestsol<0:
                            bestsol=copy.deepcopy(updateDistances)
                            besttour=fixed_cost_services(newtour)
                            bestinit=copy.copy(updateTour)

                    # If we accept a worse solution as a new solution due to the metropolis function
                    elif math.exp(-(abs(existingDistances-updateDistances))/(temp))>=rand: 
                        tour=copy.copy(updateTour)
                        existingDistances=updateDistances

                    # If we dont accept a worse solution as a new solution due to the metropolis function
                    elif math.exp(-(abs(existingDistances-updateDistances))/(temp))<=rand:
                        continue

                print(iteration)

                # Checks if the constructed services have been found before, if not, then add them to the scenario
                for i in range(len(besttour)):
                    if besttour[i] in [item[3:] for item in Finalservice]: 
                        continue 
                    else: 
                        Finalservice.append([iteration]+[cappercent]+[distpercent]+besttour[i])
                iteration+=1
                
                        

            # Stores the services of each scenario and solutions of alle iterations of the simulated annealing metaheuristic
            DF = pd.DataFrame(Finalservice)
            DF.to_excel(name+"_"+str(s)+"_services_"+str(temp0)+"_"+str(tempend)+".xlsx")
            ALLlos=pd.DataFrame(losninger)
            ALLlos.to_excel(name+"_"+str(s)+"_losninger_"+str(temp0)+"_"+str(tempend)+".xlsx") 

        return
    
    # Read files containing gridsearch
    out=pd.read_excel("C:/Users/joebr/iCloudDrive/Sas/Bachelor/BA julia/"+name+'_output.xlsx')
    out = out.drop(columns=out.columns[0])
    out.replace(1, np.inf, inplace=True)

    # Find the row and column names of the minimum value in the entire DataFrame
    min_index = out.stack().idxmin()

    # Extract the row and column names from the index
    end, start = min_index
    arrtempend=[0,1000,10000,20000,50000]
    tempend=arrtempend[end]
    temp0=start

    # Call the function that constructs the services of the scenarios
    print(finalservice(temp0,tempend,tour))