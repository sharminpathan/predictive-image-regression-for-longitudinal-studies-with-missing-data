#!/usr/bin/python2
# Below is for plotting when X server is not available such as on cluster machines
import matplotlib 
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!


# configuration files related modules
from Configs import Config, Optim, Compute, VMConfig

# pyca modules
import PyCA.Core as ca
import PyCA.Common as common
import PyCA.Display as display

# vector momentum modules
from Libraries import CAvmCommon

# others
import numpy as np
import matplotlib.pyplot as plt
import os, errno

import logging
import sys
import copy
import math
import time

StudySpec = {
    'I0':
    Config.Param(default='I0.mhd', required=True,
                    comment="Initial (moving) image file"),
    'm0':
    Config.Param(default='m0.mhd', required=True,
                    comment="Initial momenta direction to shoot towards"),
    'scaleMomenta':
    Config.Param(default=1.0, required=True,
                    comment="Scale initial momenta before shooting.")    
}

GeodesicShootingConfigSpec = {
    'study': StudySpec,
    'diffOpParams':
        Config.Param(default=[0.01, 0.01, 0.001],
                     required=True,
                     comment="Differential operator parameters: alpha, beta and gamma"),
    'integration': {'integMethod':
                       Config.Param(default="RK4",
                                    comment="Integration scheme.  EULER or RK4"),
                   'nTimeSteps':
                       Config.Param(default=10,
                                    comment="Number of time discretization steps for integrating geodesic"),
                   'NIterForInverse':
                       Config.Param(default=20,
                                    comment="Iterations for computing fixed point iterative inverse of a diffeomorphism.")},

    'io': {
        'outputPrefix':
        Config.Param(default="./",
                     comment="Where to put output.  Don't forget trailing "
                     + "slash"),
        'plotSlice':
        Config.Param(default=None,
                     comment="Slice to plot.  Defaults to mid axial"),
        'plotSliceDim':
        Config.Param(default='z',
                     comment="What dimension to plot.  Defaults is z"),
        'quiverEvery':
        Config.Param(default=1,
                     comment="How much to downsample for quiver plots"),
        'gridEvery':
        Config.Param(default=1,
                     comment="How much to downsample for deformation grid plots"),
        'saveFrames':
        Config.Param(default=True,
                     comment="Save frames for every timestep for creating video later?")
},

    'useCUDA':
        Config.Param(default=False,
                 comment="Use GPU if available"),
    '_resource': 'VectorMomentum_GeodesicShooting'}



def GeodesicShooting(cf):

    # prepare output directory
    common.Mkdir_p(os.path.dirname(cf.io.outputPrefix))

    # Output loaded config
    if cf.io.outputPrefix is not None:
        cfstr = Config.ConfigToYAML(GeodesicShootingConfigSpec, cf)
        with open(cf.io.outputPrefix + "parsedconfig.yaml", "w") as f:
            f.write(cfstr)

    mType = ca.MEM_DEVICE if cf.useCUDA else ca.MEM_HOST
    #common.DebugHere()
    I0 = common.LoadITKImage(cf.study.I0, mType)
    m0 = common.LoadITKField(cf.study.m0, mType)
    grid = I0.grid()

    ca.ThreadMemoryManager.init(grid, mType, 1)    
    # set up diffOp
    if mType == ca.MEM_HOST:
        diffOp = ca.FluidKernelFFTCPU()
    else:
        diffOp = ca.FluidKernelFFTGPU()
    diffOp.setAlpha(cf.diffOpParams[0])
    diffOp.setBeta(cf.diffOpParams[1])
    diffOp.setGamma(cf.diffOpParams[2])
    diffOp.setGrid(grid)    

    g = ca.Field3D(grid,mType)
    ginv = ca.Field3D(grid,mType)
    mt = ca.Field3D(grid,mType)
    It = ca.Image3D(grid,mType)
    t = [x*1./cf.integration.nTimeSteps for x in range(cf.integration.nTimeSteps+1)]
    checkpointinds = range(1,len(t))
    checkpointstates =  [(ca.Field3D(grid,mType),ca.Field3D(grid,mType)) for idx in checkpointinds]

    scratchV1 = ca.Field3D(grid,mType)
    scratchV2 = ca.Field3D(grid,mType)
    scratchV3 = ca.Field3D(grid,mType)
    # scale momenta to shoot
    cf.study.scaleMomenta = float(cf.study.scaleMomenta)
    if abs(cf.study.scaleMomenta) > 0.000000:
        ca.MulC_I(m0,float(cf.study.scaleMomenta))
        CAvmCommon.IntegrateGeodesic(m0,t,diffOp, mt, g, ginv,\
                                     scratchV1,scratchV2,scratchV3,\
                                     keepstates=checkpointstates,keepinds=checkpointinds,
                                     Ninv=cf.integration.NIterForInverse, integMethod = cf.integration.integMethod)
    else:
        ca.Copy(It,I0)
        ca.Copy(mt,m0)
        ca.SetToIdentity(ginv)        
        ca.SetToIdentity(g)        

    # write output
    if cf.io.outputPrefix is not None: 
        # scale back shotmomenta before writing
        if abs(cf.study.scaleMomenta) > 0.000000:
            ca.ApplyH(It,I0,ginv)
            ca.CoAd(mt,ginv, m0)
            ca.DivC_I(mt,float(cf.study.scaleMomenta))

        common.SaveITKImage(It, cf.io.outputPrefix+"I1.mhd")
        common.SaveITKField(mt, cf.io.outputPrefix+"m1.mhd")
        common.SaveITKField(ginv, cf.io.outputPrefix+"phiinv.mhd")
        common.SaveITKField(g, cf.io.outputPrefix+"phi.mhd")
        GeodesicShootingPlots(g, ginv, I0, It, cf)
        if cf.io.saveFrames:
            SaveFrames(checkpointstates, checkpointinds, I0, It,m0,mt, cf)
    # end if
# end GeodesicShooting


def SaveFrames(checkpointstates, checkpointinds, I0, It,m0,mt, cf):
    momentathresh=0.00002
    common.Mkdir_p(os.path.dirname(cf.io.outputPrefix)+'/frames/')
    image_idx = 0;
    fig = plt.figure(1,frameon=False)
    plt.clf()    
    display.DispImage(I0, '', newFig=False, cmap='gray', dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice)
    plt.draw()
    outfilename = cf.io.outputPrefix+'/frames/I'+str(image_idx).zfill(5)+'.png'
    fig.set_size_inches(4,4)
    plt.savefig(outfilename,bbox_inches='tight', pad_inches=0,dpi=100)

    fig = plt.figure(2,frameon=False)  
    plt.clf()
    temp = ca.Field3D(I0.grid(),I0.memType())
    ca.SetToIdentity(temp)
    common.DebugHere()
    CAvmCommon.MyGridPlot(temp,every=cf.io.gridEvery,color='k', dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice, isVF=False, plotBase=False)
    #fig.patch.set_alpha(0)
    #fig.patch.set_visible(False)
    a=fig.gca()
    #a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('tight')
    plt.axis('image')
    plt.axis('off')
    plt.draw()
    fig.set_size_inches(4,4)
    outfilename = cf.io.outputPrefix+'/frames/invdef'+str(image_idx).zfill(5)+'.png'
    plt.savefig(outfilename,bbox_inches='tight', pad_inches=0,dpi=100)

    fig = plt.figure(3,frameon=False)  
    plt.clf()
    CAvmCommon.MyGridPlot(temp,every=cf.io.gridEvery,color='k', dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice, isVF=False, plotBase=False)
    #fig.patch.set_alpha(0)
    #fig.patch.set_visible(False)
    a=fig.gca()
    #a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('tight')
    plt.axis('image')
    plt.axis('off')
    plt.draw()
    fig.set_size_inches(4,4)
    outfilename = cf.io.outputPrefix+'/frames/def'+str(image_idx).zfill(5)+'.png'
    plt.savefig(outfilename,bbox_inches='tight', pad_inches=0,dpi=100)
    
    fig = plt.figure(4,frameon=False)  
    plt.clf()
    display.DispImage(I0, '', newFig=False, cmap='gray', dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice)
    plt.hold('True')
    CAvmCommon.MyQuiver(m0, dim=cf.io.plotSliceDim,sliceIdx=cf.io.plotSlice,every=cf.io.quiverEvery,thresh=momentathresh,scaleArrows=0.25,arrowCol='r',lineWidth=0.5, width=0.005)
    plt.draw()

    plt.hold('False')

    outfilename = cf.io.outputPrefix+'/frames/m'+str(image_idx).zfill(5)+'.png'
    fig.set_size_inches(4,4)
    plt.savefig(outfilename,bbox_inches='tight', pad_inches=0,dpi=100)
    
    for i in range(len(checkpointinds)):
        image_idx = image_idx + 1
        ca.ApplyH(It, I0, checkpointstates[i][1])
        fig = plt.figure(1,frameon=False)
        plt.clf()
        display.DispImage(It, '', newFig=False, cmap='gray', dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice)
        plt.draw()
        outfilename = cf.io.outputPrefix+'/frames/I'+str(image_idx).zfill(5)+'.png'
        fig.set_size_inches(4,4)
        plt.savefig(outfilename,bbox_inches='tight', pad_inches=0,dpi=100)

        fig = plt.figure(2,frameon=False)  
        plt.clf()
        CAvmCommon.MyGridPlot(checkpointstates[i][1],every=cf.io.gridEvery,color='k', dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice, isVF=False, plotBase=False)
        #fig.patch.set_alpha(0)
        #fig.patch.set_visible(False)
        a=fig.gca()
        #a.set_frame_on(False)
        a.set_xticks([]); a.set_yticks([])
        plt.axis('tight')
        plt.axis('image')
        plt.axis('off')
        plt.draw()
        outfilename = cf.io.outputPrefix+'/frames/invdef'+str(image_idx).zfill(5)+'.png'
        fig.set_size_inches(4,4)
        plt.savefig(outfilename,bbox_inches='tight', pad_inches=0,dpi=100)

        fig = plt.figure(3,frameon=False)  
        plt.clf()
        CAvmCommon.MyGridPlot(checkpointstates[i][0],every=cf.io.gridEvery,color='k', dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice, isVF=False, plotBase=False)
        #fig.patch.set_alpha(0)
        #fig.patch.set_visible(False)
        a=fig.gca()
        #a.set_frame_on(False)
        a.set_xticks([]); a.set_yticks([])
        plt.axis('tight')
        plt.axis('image')
        plt.axis('off')
        plt.draw()
        outfilename = cf.io.outputPrefix+'/frames/def'+str(image_idx).zfill(5)+'.png'
        fig.set_size_inches(4,4)
        plt.savefig(outfilename,bbox_inches='tight', pad_inches=0,dpi=100)

        ca.CoAd(mt, checkpointstates[i][1], m0 )
        fig = plt.figure(4,frameon=False)  
        plt.clf()
        display.DispImage(It, '', newFig=False, cmap='gray', dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice)
        plt.hold('True')
        CAvmCommon.MyQuiver(mt, dim=cf.io.plotSliceDim,sliceIdx=cf.io.plotSlice,every=cf.io.quiverEvery,thresh=momentathresh,scaleArrows=0.40,arrowCol='r',lineWidth=0.5, width=0.005)
        plt.draw()
        plt.hold('False')
        outfilename = cf.io.outputPrefix+'/frames/m'+str(image_idx).zfill(5)+'.png'
        fig.set_size_inches(4,4)
        plt.savefig(outfilename,bbox_inches='tight', pad_inches=0,dpi=100)
 
        
def GeodesicShootingPlots(g, ginv, I0, It, cf):
    fig = plt.figure(3)
    plt.clf()
    fig.patch.set_facecolor('white')
    plt.subplot(2,2,1)
    CAvmCommon.MyGridPlot(g,every=cf.io.gridEvery,color='k', dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice, isVF=False)
    plt.axis('equal')
    plt.axis('off')
    plt.title('\phi')

    plt.subplot(2,2,2)
    CAvmCommon.MyGridPlot(ginv,every=cf.io.gridEvery,color='k', dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice, isVF=False)
    plt.axis('equal')
    plt.axis('off')
    plt.title('\phi^{-1}')
    
    plt.subplot(2,2,3)
    display.DispImage(I0, 'I0', newFig=False, dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice)
    plt.subplot(2,2,4)
    display.DispImage(It, 'I1', newFig=False, dim=cf.io.plotSliceDim, sliceIdx=cf.io.plotSlice)
    plt.draw()
    plt.show()
    if cf.io.outputPrefix != None: plt.savefig(cf.io.outputPrefix+'shooting.pdf')


if __name__ == '__main__':

    try:
        usercfg = Config.Load(spec=GeodesicShootingConfigSpec, argv=sys.argv)
    except Config.MissingConfigError:
        # Don't freak out, this should have printed a sample config to STDOUT
        sys.exit(1)

    GeodesicShooting(usercfg)
