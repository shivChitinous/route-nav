import numpy as np
import scipy as sp
import xarray as xr

#we create points with some orientation given boundaries  
def random_view(xbound, ybound, n, zAnt):
    xpoints = np.random.uniform(xbound[0], xbound[1], n)
    ypoints = np.random.uniform(ybound[0], ybound[1], n)
    theta = np.random.rand(n)*2*np.pi
    zpoints = zAnt*np.ones(n)
    routeNames = np.array([0])
    return np.array([np.vstack([xpoints, ypoints, zpoints, theta]).T]), routeNames

def snapLocs(dWorld, zAnt, n):
    xbound = [np.min(dWorld.loc['X',:,:].values), np.max(dWorld.loc['X',:,:].values)]
    ybound = [np.min(dWorld.loc['Y',:,:].values), np.max(dWorld.loc['Y',:,:].values)]
    views, routeNames = random_view(xbound, ybound, n, zAnt)
    snapLocs = xr.DataArray(data = views, coords = {'routes': routeNames, 'frames': range(views.shape[1]), 'vars': ['X', 'Y', 'Z', 'th']})
    return snapLocs

def scanLocs(learntID, learntLocs, learntHDs, zAnt,
             r, boundary, thetaRes, thetaLim):

    sceneLocs = learntLocs[learntID:learntID+2] #from 1 learnt frame to the next
    sceneHDs = learntHDs[learntID:learntID+2] #from 1 learnt frame to the next

    x = np.arange(sceneLocs[:,0].min()-boundary*r,sceneLocs[:,0].max()+boundary*r,r)
    y = np.arange(sceneLocs[:,1].min()-boundary*r,sceneLocs[:,1].max()+boundary*r,r)
    X,Y = np.meshgrid(x, y)
    
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    
    scan = np.arange(sceneHDs.min()
                     -thetaLim*np.pi/180,sceneHDs.max()+
                     thetaLim*np.pi/180,thetaRes*np.pi/180)
    
    routex = np.stack([np.meshgrid(grid[:,0],zAnt,scan)]).T.reshape(-1,3)
    routey = np.stack([np.meshgrid(grid[:,1],zAnt,scan)]).T.reshape(-1,3)
    routes = np.zeros([1,routex.shape[0],4])
    
    routes[0,:,0] = routex[:,0]
    routes[0,:,1] = routey[:,0]
    routes[0,:,2] = routex[:,1]
    routes[0,:,3] = routex[:,2]
    
    dVirtual = xr.DataArray(data = routes, coords = {'routes': np.array([0]),
                    'frames': range(routes.shape[1]),
                    'vars': ['X', 'Y', 'Z', 'th']})
    
    return grid, scan, dVirtual