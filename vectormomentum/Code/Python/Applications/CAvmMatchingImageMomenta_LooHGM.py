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

# HGM modules
from Libraries import CAvmHGMCommon

# HGM application module
from Applications import CAvmHGM

# others
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import csv
#import mpi4py

StudySpec = {
    'I':
    Config.Param(default='I.mhd', required=True,
                    comment="Initial (moving) image file, I to be matched to J"),
    'm':
    Config.Param(default='m.mhd', required=True,
                    comment="Initial momenta, m at I to be matched to n"),
    'J':
    Config.Param(default='J.mhd', required=True,
                    comment="Target (fixed) image file, J"),
    'n':
    Config.Param(default='n.mhd', required=True,
                    comment="Initial momenta, n at the target J")
    
}

MatchingImageMomentaVMConfigSpec = {
    'sigmai':
    Config.Param(default=0.1,
                 required=True,
                 comment="Regularization weight on image match term in residual"),
    'sigmaS':
    Config.Param(default=0.1,
                 required=True,
                 comment="Regularization weight on slope match term as per HGM"),
    'sigmaI':
    Config.Param(default=0.1,
                 required=True,
                 comment="Regularization weight on intercept match term as per HGM"),
    'diffOpParams':
    Config.Param(default=[0.01, 0.01, 0.001],
                 required=True,
                 comment="Differential operator parameters: alpha, beta and gamma"),
    'matchImOnly':
    Config.Param(default=False,
                 comment="When True, it runs vector momenta image matching. Supplied source and target momenta are ignored.")
    }

MatchingImageMomentaConfigSpec = {
    'compute': Compute.ComputeConfigSpec,
    'vectormomentum': MatchingImageMomentaVMConfigSpec,
    'study': StudySpec,
    'optim': Optim.OptimConfigSpec,
    'io': {
        'plotEvery':
        Config.Param(default=10,
                     comment="Update plots every N iterations"),
        'plotSlice':
        Config.Param(default=None,
                     comment="Slice to plot.  Defaults to mid axial"),
        'quiverEvery':
        Config.Param(default=1,
                     comment="How much to downsample for quiver plots"),
        'outputPrefix':
        Config.Param(default="./",
                     comment="Where to put output.  Don't forget trailing "
                     + "slash")},
    '_resource': 'VectorMomentum_MatchingImageMomenta'}

def MatchingImageMomenta(cf):
    """Runs matching for image momenta pair."""
    if cf.compute.useCUDA and cf.compute.gpuID is not None:
        ca.SetCUDADevice(cf.compute.gpuID)
    
    common.DebugHere()
    # prepare output directory
    common.Mkdir_p(os.path.dirname(cf.io.outputPrefix))
    
    # Output loaded config
    if cf.io.outputPrefix is not None:
        cfstr = Config.ConfigToYAML(MatchingImageMomentaConfigSpec, cf)
        with open(cf.io.outputPrefix + "parsedconfig.yaml", "w") as f:
            f.write(cfstr)

    # mem type is determined by whether or not we're using CUDA
    mType = ca.MEM_DEVICE if cf.compute.useCUDA else ca.MEM_HOST
    
    # load image data in memory
    I0 = common.LoadITKImage(cf.study.I, mType)
    J1 = common.LoadITKImage(cf.study.J, mType)

    # get imGrid from data
    imGrid = I0.grid()
    # load vectorfield in memory
    if cf.vectormomentum.matchImOnly:
        m0 = ca.Field3D(imGrid, mType)        
        n1 = ca.Field3D(imGrid, mType)        
        ca.SetMem(m0,0.0)
        ca.SetMem(n1,0.0)
    else:
        m0 = common.LoadITKField(cf.study.m, mType)
        n1 = common.LoadITKField(cf.study.n, mType)
    
    # create time array with checkpointing info for this geodesic to be estimated
    (s, scratchInd, rCpinds) = CAvmHGM.HGMSetUpTimeArray(cf.optim.nTimeSteps, [1.0], 0.001)
    tDiscGeodesic = CAvmHGMCommon.HGMSetupTimeDiscretizationResidual(s, rCpinds, imGrid, mType)    

    # create the state variable for geodesic that is going to hold all info
    p0 = ca.Field3D(imGrid,mType)
    geodesicState = CAvmHGMCommon.HGMResidualState(I0, p0, imGrid, mType, cf.vectormomentum.diffOpParams[0], cf.vectormomentum.diffOpParams[1], cf.vectormomentum.diffOpParams[2], s, cf.optim.NIterForInverse, cf.vectormomentum.sigmaI, cf.vectormomentum.sigmaS, cf.vectormomentum.sigmai, cf.optim.stepSize, integMethod = cf.optim.integMethod)
    # initialize with zero
    ca.SetMem(geodesicState.p0,0.0)
    # start up the memory manager for scratch variables
    ca.ThreadMemoryManager.init(imGrid, mType, 0)
    EnergyHistory = []
    # run the loop
    for it in range(cf.optim.Niter):
        # shoot the geodesic forward
        CAvmHGMCommon.HGMIntegrateGeodesic(geodesicState.p0, geodesicState.s,geodesicState.diffOp, geodesicState.p, geodesicState.rho,geodesicState.rhoinv, tDiscGeodesic, geodesicState.Ninv, geodesicState.integMethod)    
        # integrate the geodesic backward
        CAvmHGMCommon.HGMIntegrateAdjointsResidual(geodesicState, tDiscGeodesic, m0, J1, n1)

        # TODO: verify it should just be log map/simple image matching when sigmaM=\infty       
        # gradient descent step for geodesic.p0
        CAvmHGMCommon.HGMTakeGradientStepResidual(geodesicState)        

        # compute and print energy
        (VEnergy, IEnergy, MEnergy) = MatchingImageMomentaComputeEnergy(geodesicState, m0, J1, n1)
        EnergyHistory.append([VEnergy+IEnergy+MEnergy,VEnergy, IEnergy, MEnergy])
        print "Iter", it, "of", cf.optim.Niter,":" ,VEnergy+IEnergy+MEnergy, '(Total) = ',VEnergy, '(Vector) + ',IEnergy,'(Image Match) + ',MEnergy,'(Momenta Match)'
        
        # plots
        if cf.io.plotEvery > 0 and (((it+1) % cf.io.plotEvery == 0) or (it == cf.optim.Niter - 1)):            
            MatchingImageMomentaPlots(cf, geodesicState, tDiscGeodesic, EnergyHistory, m0, J1, n1, writeOutput=True)        

    # write output
    MatchingImageMomentaWriteOuput(cf, geodesicState, EnergyHistory,m0,n1)

# end MatchingImageMomenta

def MatchingImageMomentaComputeEnergy(geodesicState, m0, J1, n1):
    vecEnergy = 0.0 
    imageMatchEnergy = 0.0
    momentaMatchEnergy = 0.0

    grid = geodesicState.J0.grid()
    mType = geodesicState.J0.memType()

    imdiff = ca.ManagedImage3D(grid, mType)
    vecdiff = ca.ManagedField3D(grid, mType)
     
    # image match energy
    ca.ApplyH(imdiff,geodesicState.J0,geodesicState.rhoinv)
    ca.Sub_I(imdiff, J1)
    imageMatchEnergy = 0.5*ca.Sum2(imdiff)/(float(geodesicState.p0.nVox())*geodesicState.Sigma*geodesicState.Sigma*geodesicState.SigmaIntercept*geodesicState.SigmaIntercept) # save for use in intercept energy term 
     
    # momenta match energy
    ca.CoAd(geodesicState.p,geodesicState.rhoinv, m0)
    ca.Sub_I(geodesicState.p, n1)
    ca.Copy(vecdiff, geodesicState.p) # save for use in slope energy term
    geodesicState.diffOp.applyInverseOperator(geodesicState.p)
    momentaMatchEnergy = ca.Dot(vecdiff, geodesicState.p)/(float(geodesicState.p0.nVox())*geodesicState.SigmaSlope*geodesicState.SigmaSlope)

    # vector energy. p is used as scratch variable
    ca.Copy(geodesicState.p, geodesicState.p0)
    geodesicState.diffOp.applyInverseOperator(geodesicState.p)
    vecEnergy = ca.Dot(geodesicState.p0, geodesicState.p)/(float(geodesicState.p0.nVox())*geodesicState.SigmaIntercept*geodesicState.SigmaIntercept) 

    return (vecEnergy, imageMatchEnergy, momentaMatchEnergy)

# end MatchingImageMomentaComputeEnergy

def MatchingImageMomentaWriteOuput(cf, geodesicState,EnergyHistory,m0,n1):
    grid = geodesicState.J0.grid()
    mType = geodesicState.J0.memType()

    # save momenta for the gedoesic
    common.SaveITKField(geodesicState.p0, cf.io.outputPrefix + "p0.mhd")         

    # save matched momenta for the geodesic
    if cf.vectormomentum.matchImOnly:
        m0 = common.LoadITKField(cf.study.m, mType)
        

    ca.CoAd(geodesicState.p,geodesicState.rhoinv, m0)
    common.SaveITKField(geodesicState.p, cf.io.outputPrefix + "m1.mhd")

    # momenta match energy
    if cf.vectormomentum.matchImOnly:
        vecdiff = ca.ManagedField3D(grid, mType)
        ca.Sub_I(geodesicState.p, n1)
        ca.Copy(vecdiff, geodesicState.p)
        geodesicState.diffOp.applyInverseOperator(geodesicState.p)
        momentaMatchEnergy = ca.Dot(vecdiff, geodesicState.p)/(float(geodesicState.p0.nVox())*geodesicState.SigmaSlope*geodesicState.SigmaSlope)
        # save energy
        energyFilename =  cf.io.outputPrefix + "testMomentaMatchEnergy.csv"
        with open(energyFilename, 'w') as f:
            print>>f, momentaMatchEnergy

    # save matched image for the geodesic
    tempim = ca.ManagedImage3D(grid, mType)
    ca.ApplyH(tempim,geodesicState.J0, geodesicState.rhoinv)
    common.SaveITKImage(tempim, cf.io.outputPrefix + "I1.mhd")

    # save energy
    energyFilename =  cf.io.outputPrefix + "energy.csv"
    MatchingImageMomentaWriteEnergyHistoryToFile(EnergyHistory,energyFilename)            
# end MatchingImageMomentaWriteOuput

def MatchingImageMomentaWriteEnergyHistoryToFile(listname,filename):
    try:
        os.remove(filename)
    except OSError:
        pass
    theFile = open(filename, 'w')
    csv_writer = csv.writer(theFile, delimiter='\t')
    csv_writer.writerows(listname)
    theFile.close()
#end HGMWriteEnergyHistoryToFile

def MatchingImageMomentaPlots(cf, geodesicState, tDiscGeodesic, EnergyHistory, m0, J1, n1, writeOutput=True):
    """
    Do some summary plots for MatchingImageMomenta
    """

    #ENERGY
    fig = plt.figure(1)
    plt.clf()
    fig.patch.set_facecolor('white')
    
    TE = [row[0] for row in EnergyHistory] 
    VE = [row[1] for row in EnergyHistory] 
    IE = [row[2] for row in EnergyHistory] 
    ME = [row[3] for row in EnergyHistory] 
    plt.subplot(2,2,1)
    plt.plot(TE)
    plt.title('Total Energy')
    plt.hold(False)
    plt.subplot(2,2,2)
    plt.plot(VE)
    plt.title('Vector Energy')
    plt.hold(False)
    plt.subplot(2,2,3)
    plt.plot(IE)
    plt.title('Image Match Energy')
    plt.hold(False)
    plt.subplot(2,2,4)
    plt.plot(ME)
    plt.title('Momenta Match Energy')
    plt.hold(False)
    plt.draw()
    plt.show()
    if cf.io.outputPrefix != None and writeOutput: plt.savefig(cf.io.outputPrefix+'energy.pdf')

    # GEODESIC INITIAL CONDITIONS and RHO and RHO inv
    CAvmHGMCommon.HGMIntegrateGeodesic(geodesicState.p0, geodesicState.s,geodesicState.diffOp, geodesicState.p, geodesicState.rho,geodesicState.rhoinv, tDiscGeodesic, geodesicState.Ninv,geodesicState.integMethod)

    fig = plt.figure(2)
    plt.clf()
    fig.patch.set_facecolor('white')

    plt.subplot(2,2,1)
    display.DispImage(geodesicState.J0, 'J0', newFig=False, sliceIdx=cf.io.plotSlice)
    plt.subplot(2,2,2)
    ca.ApplyH(geodesicState.J,geodesicState.J0,geodesicState.rhoinv)
    display.DispImage(geodesicState.J, 'J1', newFig=False, sliceIdx=cf.io.plotSlice)

    plt.subplot(2,2,3)
    display.GridPlot(geodesicState.rhoinv,every=cf.io.quiverEvery,color='k', sliceIdx=cf.io.plotSlice, isVF=False)
    plt.axis('equal')
    plt.axis('off')
    plt.title('rho^{-1}')
    plt.subplot(2,2,4)
    display.GridPlot(geodesicState.rho,every=cf.io.quiverEvery,color='k', sliceIdx=cf.io.plotSlice, isVF=False)
    plt.axis('equal')
    plt.axis('off')
    plt.title('rho')    
    if cf.io.outputPrefix != None and writeOutput: plt.savefig(cf.io.outputPrefix+'def.pdf')
        
    # MATCHING DIFFERENCE IMAGES    
    grid = geodesicState.J0.grid()
    mType = geodesicState.J0.memType()
    imdiff =  ca.ManagedImage3D(grid, mType)

    # Image matching
    ca.Copy(imdiff,geodesicState.J)
    ca.Sub_I(imdiff, J1)
    fig = plt.figure(3)
    plt.clf()
    fig.patch.set_facecolor('white')

    plt.subplot(1,3,1)
    display.DispImage(geodesicState.J0, 'Source J0', newFig=False, sliceIdx=cf.io.plotSlice)
    plt.colorbar()

    plt.subplot(1,3,2)
    display.DispImage(J1, 'Target J1', newFig=False, sliceIdx=cf.io.plotSlice)
    plt.colorbar()

    plt.subplot(1,3,3)
    display.DispImage(imdiff, 'rho.J0-J1', newFig=False, sliceIdx=cf.io.plotSlice)
    plt.colorbar()
    if cf.io.outputPrefix != None and writeOutput: plt.savefig(cf.io.outputPrefix+'diffImage.pdf')    
    

    # Momenta matching 
    if mType == ca.MEM_DEVICE:
        scratchV1 =  ca.Field3D(grid, mType) 
        scratchV2 =  ca.Field3D(grid, mType) 
        scratchV3 =  ca.Field3D(grid, mType) 
    else:
        scratchV1 =  ca.ManagedField3D(grid, mType) 
        scratchV2 =  ca.ManagedField3D(grid, mType) 
        scratchV3 =  ca.ManagedField3D(grid, mType) 

    fig = plt.figure(4)
    plt.clf()
    fig.patch.set_facecolor('white')
    ca.Copy(scratchV1,m0)
    scratchV1.toType(ca.MEM_HOST)
    m0_x, m0_y, m0_z = scratchV1.asnp()
    plt.subplot(2,3,1)
    plt.imshow(np.squeeze(m0_x)); plt.colorbar(); plt.title('X: Source m0 ')
    plt.subplot(2,3,4)
    plt.imshow(np.squeeze(m0_y)); plt.colorbar(); plt.title('Y: Source m0')

    ca.Copy(scratchV2,n1)
    scratchV2.toType(ca.MEM_HOST)
    n1_x, n1_y, n1_z = scratchV2.asnp()
    plt.subplot(2,3,2)
    plt.imshow(np.squeeze(n1_x)); plt.colorbar(); plt.title('X: Target n1')
    plt.subplot(2,3,5)
    plt.imshow(np.squeeze(n1_y)); plt.colorbar(); plt.title('Y: Target n1')

    ca.CoAd(scratchV3,geodesicState.rhoinv,m0)
    ca.Sub_I(scratchV3, n1)  
    scratchV3.toType(ca.MEM_HOST)
    diff_x, diff_y, diff_z = scratchV3.asnp()
    plt.subplot(2,3,3)
    plt.imshow(np.squeeze(diff_x)); plt.colorbar(); plt.title('X: rho.m0-n1')
    plt.subplot(2,3,6)
    plt.imshow(np.squeeze(diff_y)); plt.colorbar(); plt.title('Y: rho.m0-n1')

    if cf.io.outputPrefix != None and writeOutput: plt.savefig(cf.io.outputPrefix+'diffMomenta.pdf')

    del scratchV1, scratchV2, scratchV3
    del imdiff
# end MatchingImageMomentalots


if __name__ == '__main__':
    #common.DebugHere()
    try:
        usercfg = Config.Load(spec=MatchingImageMomentaConfigSpec, argv=sys.argv)
    except Config.MissingConfigError:
        # Don't freak out, this should have printed a sample config to STDOUT
        sys.exit(1)

    Compute.Compute(MatchingImageMomenta, usercfg)  
