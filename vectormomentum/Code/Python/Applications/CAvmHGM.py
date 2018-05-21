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

# others
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import mpi4py
import csv
import socket
import datetime
StudySpec = {
    'numSubjects':
    Config.Param(default=4, required=True,
                    comment="Total number of subjects."),
    'subjectIds':
    Config.Param(default=['sid1','sid2','sid3','sid4'], required=True,
                 comment="List of subject ids. This should be unique names for each individuals"),
    'subjectIntercepts':
    Config.Param(default=['subject1_I.mhd','subject2_I.mhd','subject3_I.mhd','subject4_I.mhd'], required=True,
                    comment="List of subject initial intercept image files at their baseline times"),
    'subjectSlopes':
    Config.Param(default=['subject1_S.mhd','subject2_S.mhd','subject3_S.mhd','subject4_S.mhd'], required=True,
                    comment="List of subject initial slope vector field image files at their baseline times"),
    'subjectBaselineTimes':
    Config.Param(default=[0,0.33,0.66,1.0], required=True,
                    comment="List of subject baseline times, t (t is in [0,1])."),
    'setUnitSpacing':
    Config.Param(default=True,
                 comment="Ignore the spacing in images and set it to (1,1,1)"),
    'setZeroOrigin':
    Config.Param(default=True,
                 comment="Ignore the origin in images and set it to (0,0,0)")
}
HGMConfigSpec = {
    'compute': Compute.ComputeConfigSpec,
    'vectormomentum': VMConfig.VMLongitudinalConfigSpec,
    'study': StudySpec,
    'optim': Optim.OptimLongitudinalConfigSpec,
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
    '_resource': 'VectorMomentum_HGM'}

def BuildHGM(cf):
    """Worker for running Hierarchical Geodesic Model (HGM) 
n    for group geodesic estimation on a subset of individuals. 
    Runs HGM on this subset sequentially. The variations retuned
    are summed up to get update for all individuals"""

    size = Compute.GetMPIInfo()['size']
    rank = Compute.GetMPIInfo()['rank']
    name = Compute.GetMPIInfo()['name']
    localRank = Compute.GetMPIInfo()['local_rank']
    nodename = socket.gethostname()

    # prepare output directory
    common.Mkdir_p(os.path.dirname(cf.io.outputPrefix))

    # just one reporter process on each node
    isReporter = rank == 0
    cf.study.numSubjects = len(cf.study.subjectIntercepts)    
    if isReporter:
        # Output loaded config
        if cf.io.outputPrefix is not None:
            cfstr = Config.ConfigToYAML(HGMConfigSpec, cf)
            with open(cf.io.outputPrefix + "parsedconfig.yaml", "w") as f:
                f.write(cfstr)
    #common.DebugHere()

    # if MPI check if processes are greater than number of subjects. it is okay if there are more subjects than processes

    if cf.compute.useMPI and (cf.study.numSubjects < cf.compute.numProcesses):
        raise Exception("Please don't use more processes "+
                                "than total number of individuals")
    
    # subdivide data, create subsets for this thread to work on
    nodeSubjectIds = cf.study.subjectIds[rank::cf.compute.numProcesses]
    nodeIntercepts = cf.study.subjectIntercepts[rank::cf.compute.numProcesses]
    nodeSlopes = cf.study.subjectSlopes[rank::cf.compute.numProcesses]
    nodeBaselineTimes = cf.study.subjectBaselineTimes[rank::cf.compute.numProcesses]
    sys.stdout.write(
        "This is process %d of %d with name: %s on machinename: %s and local rank: %d.\nnodeIntercepts: %s\n nodeSlopes: %s\n nodeBaselineTimes: %s\n"
        % (rank, size, name, nodename,localRank,nodeIntercepts,nodeSlopes,nodeBaselineTimes))
   
    # mem type is determined by whether or not we're using CUDA
    mType = ca.MEM_DEVICE if cf.compute.useCUDA else ca.MEM_HOST
    
    # load data in memory
    # load intercepts
    J = [common.LoadITKImage(f, mType)
         if isinstance(f, str) else f for f in nodeIntercepts]

    # load slopes
    n = [common.LoadITKField(f, mType)
         if isinstance(f, str) else f for f in nodeSlopes]    

    # get imGrid from data
    imGrid = J[0].grid()

    # create time array with checkpointing info for group geodesic
    (t, Jind, gCpinds) = HGMSetUpTimeArray(cf.optim.nTimeStepsGroup, nodeBaselineTimes, 0.0000001)
    tdiscGroup = CAvmHGMCommon.HGMSetupTimeDiscretizationGroup(t, J, n, Jind, gCpinds, mType, nodeSubjectIds)
    
    # create time array with checkpointing info for residual geodesic
    (s, scratchInd, rCpinds) = HGMSetUpTimeArray(cf.optim.nTimeStepsResidual, [1.0], 0.0000001)
    tdiscResidual = CAvmHGMCommon.HGMSetupTimeDiscretizationResidual(s, rCpinds, imGrid, mType)    

    # create group state and residual state 
    groupState = CAvmHGMCommon.HGMGroupState(imGrid, mType, cf.vectormomentum.diffOpParamsGroup[0], cf.vectormomentum.diffOpParamsGroup[1], cf.vectormomentum.diffOpParamsGroup[2], t, cf.optim.NIterForInverse, cf.vectormomentum.varIntercept, cf.vectormomentum.varSlope, cf.vectormomentum.varInterceptReg, cf.optim.stepSizeGroup, integMethod = cf.optim.integMethodGroup)

    #ca.Copy(groupState.I0, common.LoadITKImage('/usr/sci/projects/ADNI/nikhil/software/vectormomentumtest/TestData/FlowerData/Longitudinal/GroupGeodesic/I0.mhd', mType))
    
    # note that residual state is treated a scratch variable in this algorithm and reused for computing residual geodesics of multiple individual
    residualState = CAvmHGMCommon.HGMResidualState(None, None, imGrid, mType, cf.vectormomentum.diffOpParamsResidual[0], cf.vectormomentum.diffOpParamsResidual[1], cf.vectormomentum.diffOpParamsResidual[2], s, cf.optim.NIterForInverse, cf.vectormomentum.varIntercept, cf.vectormomentum.varSlope, cf.vectormomentum.varInterceptReg,cf.optim.stepSizeResidual, integMethod = cf.optim.integMethodResidual)
    
    # start up the memory manager for scratch variables
    ca.ThreadMemoryManager.init(imGrid, mType, 0)

    # need some host memory in np array format for MPI reductions
    if cf.compute.useMPI:
        mpiImageBuff = None if mType == ca.MEM_HOST else ca.Image3D(imGrid,
                                                              ca.MEM_HOST)
        mpiFieldBuff = None if mType == ca.MEM_HOST else ca.Field3D(imGrid,
                                                              ca.MEM_HOST)
    for i in range(len(groupState.t)-1,-1,-1):
        if tdiscGroup[i].J is not None:
            indx_last_individual = i
            break
    '''
    # initial template image
    ca.SetMem(groupState.I0, 0.0)
    tmp = ca.ManagedImage3D(imGrid, mType)

    for tdisc in tdiscGroup:
        if tdisc.J is not None:
            ca.Copy(tmp, tdisc.J)
            groupState.I0 += tmp
    del tmp
    if cf.compute.useMPI:
        Compute.Reduce(groupState.I0, mpiImageBuff)
    
    # divide by total num subjects
    groupState.I0 /= cf.study.numSubjects
    '''

    # run the loop

    for it in range(cf.optim.Niter):
        # compute HGM variation for group 
        HGMGroupVariation(groupState, tdiscGroup, residualState, tdiscResidual, cf.io.outputPrefix, rank, it)       
        common.CheckCUDAError("Error after HGM iteration") 
        # compute gradient for momenta (m is used as scratch)
        # if there are multiple nodes we'll need to sum across processes now
        if cf.compute.useMPI:
            # do an MPI sum
            Compute.Reduce(groupState.sumSplatI,mpiImageBuff)
            Compute.Reduce(groupState.sumJac,mpiImageBuff)
            Compute.Reduce(groupState.madj,mpiFieldBuff)
            # also sum up energies of other nodes
            # intercept            
            Eintercept = np.array([groupState.EnergyHistory[-1][1]])
            mpi4py.MPI.COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE,
                                            Eintercept,
                                            op=mpi4py.MPI.SUM)
            groupState.EnergyHistory[-1][1] = Eintercept[0]

            Eslope = np.array([groupState.EnergyHistory[-1][2]])
            mpi4py.MPI.COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE,
                                            Eslope,
                                            op=mpi4py.MPI.SUM)
            groupState.EnergyHistory[-1][2] = Eslope[0]
        
        ca.Copy(groupState.m, groupState.m0)
        groupState.diffOp.applyInverseOperator(groupState.m)        
        ca.Sub_I(groupState.m, groupState.madj)
        #groupState.diffOp.applyOperator(groupState.m)                
        # now take gradient step in momenta for group
        if cf.optim.method == 'FIXEDGD':
            # take fixed stepsize gradient step
            ca.Add_MulC_I(groupState.m0,groupState.m,-cf.optim.stepSizeGroup)
        else:
            raise Exception("Unknown optimization scheme: "+cf.optim.method)
        # end if

        # now divide to get the new base image for group
        ca.Div(groupState.I0, groupState.sumSplatI, groupState.sumJac)
        
        # keep track of energy in this iteration
        if isReporter and cf.io.plotEvery > 0 and (((it+1) % cf.io.plotEvery == 0) or (it == cf.optim.Niter - 1)):            
            HGMPlots(cf, groupState, tdiscGroup, residualState, tdiscResidual, indx_last_individual, writeOutput=True)

        if isReporter:
            (VEnergy, IEnergy, SEnergy) = groupState.EnergyHistory[-1]
            print datetime.datetime.now().time(), " Iter", it, "of", cf.optim.Niter,":" ,VEnergy+IEnergy+SEnergy, '(Total) = ',VEnergy, '(Vector) + ',IEnergy,'(Intercept) + ',SEnergy,'(Slope)'
    
    # write output images and fields
    HGMWriteOutput(cf, groupState, tdiscGroup, isReporter)
# end BuildHGM
def HGMPreprocessInput(cf,J,n):
    for i in range(len(J)):
        if cf.study.setUnitSpacing:
            J[i].setSpacing(ca.Vec3Df(1.0, 1.0, 1.0))
            n[i].setSpacing(ca.Vec3Df(1.0, 1.0, 1.0))
        if cf.study.setZeroOrigin:
            J[i].setOrigin(ca.Vec3Df(0, 0, 0))        
            n[i].setOrigin(ca.Vec3Df(0, 0, 0))        
    # end for
# end HGMPreprocessInput
def HGMGroupVariation(groupState, tDiscGroup, residualState, tDiscResidual, outputPrefix, rank, it):
    
    # shoot group geodesic forward
    CAvmHGMCommon.HGMIntegrateGeodesic(groupState.m0,groupState.t,groupState.diffOp, groupState.m,groupState.g,groupState.ginv, tDiscGroup, groupState.Ninv,groupState.integMethod)

    # integrate group geodesic backward    
    CAvmHGMCommon.HGMIntegrateAdjoints(groupState, tDiscGroup, residualState, tDiscResidual)    
# end HGMVariation

def HGMWriteOutput(cf, groupState, tDiscGroup, isReporter):
    # save initial momenta for residual geodesics, p, for all individuals
    for i in range(len(groupState.t)):
        if tDiscGroup[i].J is not None:
            common.SaveITKField(tDiscGroup[i].p0, cf.io.outputPrefix + str(tDiscGroup[i].subjectId).replace('.','_')+"_p0.mhd")           
            # write individual's energy history
            energyFilename =  cf.io.outputPrefix + str(tDiscGroup[i].subjectId).replace('.','_')+"ResidualEnergy.csv"
            HGMWriteEnergyHistoryToFile(tDiscGroup[i].Energy,energyFilename)            

    # save initial image and momenta for group gedoesic
    if isReporter:
        common.SaveITKImage(groupState.I0, cf.io.outputPrefix + "I0.mhd")
        common.SaveITKField(groupState.m0, cf.io.outputPrefix + "m0.mhd")
        # write energy history
        energyFilename =  cf.io.outputPrefix+"TotalEnergyHistory.csv"
        HGMWriteEnergyHistoryToFile(groupState.EnergyHistory,energyFilename)
# end HGMWriteOutput
def HGMWriteEnergyHistoryToFile(listname,filename):
    try:
        os.remove(filename)
    except OSError:
        pass
    theFile = open(filename, 'w')
    csv_writer = csv.writer(theFile, delimiter='\t')
    csv_writer.writerows(listname)
    theFile.close()
#end HGMWriteEnergyHistoryToFile

def HGMWriteListToFile(listname,filename):    
    try:
        os.remove(filename)
    except OSError:
        pass
    theFile = open(filename, 'w')
    for item in listname:
        print>>theFile, item
    theFile.close()
# end HGMWriteListToFile
def HGMPlots(cf, groupState, tDiscGroup, residualState, tDiscResidual, index_individual,writeOutput=True):
    """
    Do some summary plots for HGM
    """

    #ENERGY
    fig = plt.figure(1)
    plt.clf()
    fig.patch.set_facecolor('white')
    
    TE = [sum(x) for x in groupState.EnergyHistory] 
    VE = [row[0] for row in groupState.EnergyHistory] 
    IE = [row[1] for row in groupState.EnergyHistory] 
    SE = [row[2] for row in groupState.EnergyHistory] 
    TE = TE[1:]
    VE = VE[1:]
    IE = IE[1:]
    SE = SE[1:]
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
    plt.title('Intercept Energy')
    plt.hold(False)
    plt.subplot(2,2,4)
    plt.plot(SE)
    plt.title('Slope Energy')
    plt.hold(False)
    plt.draw()
    plt.show()
    if cf.io.outputPrefix != None and writeOutput: plt.savefig(cf.io.outputPrefix+'energy.pdf')

    # GROUP INITIAL CONDITIONS and PSI and PSI inv
    # shoot group geodesic forward
    CAvmHGMCommon.HGMIntegrateGeodesic(groupState.m0,groupState.t,groupState.diffOp, groupState.m,groupState.g,groupState.ginv, tDiscGroup, groupState.Ninv,groupState.integMethod)

    fig = plt.figure(2)
    plt.clf()
    fig.patch.set_facecolor('white')

    plt.subplot(2,2,1)
    display.DispImage(groupState.I0, 'I0', newFig=False, sliceIdx=cf.io.plotSlice)

    plt.subplot(2,2,2)
    ca.ApplyH(groupState.I,groupState.I0,groupState.ginv)
    display.DispImage(groupState.I, 'I1', newFig=False, sliceIdx=cf.io.plotSlice)

    plt.subplot(2,2,3)
    display.GridPlot(groupState.ginv,every=cf.io.quiverEvery,color='k', sliceIdx=cf.io.plotSlice, isVF=False)
    plt.axis('equal')
    plt.axis('off')
    plt.title('psi^{-1}')
    plt.subplot(2,2,4)
    display.GridPlot(groupState.g,every=cf.io.quiverEvery,color='k', sliceIdx=cf.io.plotSlice, isVF=False)
    plt.axis('equal')
    plt.axis('off')
    plt.title('psi')    
    if cf.io.outputPrefix != None and writeOutput: plt.savefig(cf.io.outputPrefix+'groupdef.pdf')

    # RESIDUAL INITIAL CONDITIONS and RHO and RHO inv
    ca.ApplyH(groupState.I,groupState.I0,groupState.ginv)
    residualState.J0 = groupState.I
    residualState.p0 = tDiscGroup[index_individual].p0
    CAvmHGMCommon.HGMIntegrateGeodesic(residualState.p0, residualState.s,residualState.diffOp, residualState.p, residualState.rho,residualState.rhoinv, tDiscResidual, residualState.Ninv,residualState.integMethod)

    fig = plt.figure(3)
    plt.clf()
    fig.patch.set_facecolor('white')

    plt.subplot(2,2,1)
    display.DispImage(residualState.J0, 'J0', newFig=False, sliceIdx=cf.io.plotSlice)
    plt.subplot(2,2,2)
    ca.ApplyH(residualState.J,residualState.J0,residualState.rhoinv)
    display.DispImage(residualState.J, 'J1', newFig=False, sliceIdx=cf.io.plotSlice)

    plt.subplot(2,2,3)
    display.GridPlot(residualState.rhoinv,every=cf.io.quiverEvery,color='k', sliceIdx=cf.io.plotSlice, isVF=False)
    plt.axis('equal')
    plt.axis('off')
    plt.title('rho^{-1}')
    plt.subplot(2,2,4)
    display.GridPlot(residualState.rho,every=cf.io.quiverEvery,color='k', sliceIdx=cf.io.plotSlice, isVF=False)
    plt.axis('equal')
    plt.axis('off')
    plt.title('rho')    
    if cf.io.outputPrefix != None and writeOutput: plt.savefig(cf.io.outputPrefix+'resdef.pdf')
        
    # MATCHING DIFFERENCE IMAGES    
    grid = groupState.I0.grid()
    mType = groupState.I0.memType()
    imdiff =  ca.ManagedImage3D(grid, mType)
    vecdiff =  ca.ManagedField3D(grid, mType)

    # Intercept matching
    ca.Copy(imdiff,residualState.J)
    ca.Sub_I(imdiff, tDiscGroup[index_individual].J)
    fig = plt.figure(4)
    plt.clf()
    fig.patch.set_facecolor('white')

    plt.subplot(1,3,1)
    display.DispImage(residualState.J0, 'Source J0', newFig=False, sliceIdx=cf.io.plotSlice)
    plt.colorbar()

    plt.subplot(1,3,2)
    display.DispImage(tDiscGroup[index_individual].J, 'Target J1', newFig=False, sliceIdx=cf.io.plotSlice)
    plt.colorbar()

    plt.subplot(1,3,3)
    display.DispImage(imdiff, 'rho.J0-J1', newFig=False, sliceIdx=cf.io.plotSlice)
    plt.colorbar()
    if cf.io.outputPrefix != None and writeOutput: plt.savefig(cf.io.outputPrefix+'diffintercept.pdf')    
    
    # Slope matching    
    '''
    ca.CoAd(groupState.m,groupState.ginv,groupState.m0)
    ca.CoAd(vecdiff,residualState.rhoinv,groupState.m)
    n0 = ca.Field3D(grid, ca.MEM_HOST)
    n1 = ca.Field3D(grid, ca.MEM_HOST)
    ca.Copy(n0,groupState.m)
    ca.Copy(n1,tDiscGroup[index_individual].n)    
    ca.Sub_I(vecdiff, tDiscGroup[index_individual].n)
    vecdiff.toType(ca.MEM_HOST)
    n0_x, n0_y, n0_z = n0.asnp()
    n1_x, n1_y, n1_z = n1.asnp()
    diff_x, diff_y, diff_z = vecdiff.asnp()

    fig = plt.figure(5)
    plt.clf()
    fig.patch.set_facecolor('white')

    plt.subplot(2,3,1)
    plt.imshow(np.squeeze(n0_x)); plt.colorbar(); plt.title('X: Source n0 ')

    plt.subplot(2,3,2)
    plt.imshow(np.squeeze(n1_x)); plt.colorbar(); plt.title('X: Target n1')

    plt.subplot(2,3,3)
    plt.imshow(np.squeeze(diff_x)); plt.colorbar(); plt.title('X: rho.n0-n1')

    plt.subplot(2,3,4)
    plt.imshow(np.squeeze(n0_y)); plt.colorbar(); plt.title('Y: Source n0')

    plt.subplot(2,3,5)
    plt.imshow(np.squeeze(n1_y)); plt.colorbar(); plt.title('Y: Target n1')

    plt.subplot(2,3,6)
    plt.imshow(np.squeeze(diff_y)); plt.colorbar(); plt.title('Y: rho.n0-n1')

    if cf.io.outputPrefix != None and writeOutput: plt.savefig(cf.io.outputPrefix+'diffslope.pdf')
    '''
    del imdiff
    del vecdiff
# end HGMPlots

def HGMSetUpTimeArray(nTimeSteps, subjectsTimeArray, epsilon):
    numSubjects = len(subjectsTimeArray)
    tempT = [x*1./nTimeSteps for x in range(nTimeSteps+1)]
    t = [x*1./nTimeSteps for x in range(nTimeSteps+1)]
    Sind = [0]*numSubjects

    # first create the time array
    for i in range(numSubjects):
        foundSub = False
        for j in range(len(tempT)):
            if subjectsTimeArray[i]<(tempT[j]+epsilon) and  subjectsTimeArray[i]>(tempT[j]-epsilon):
                foundSub = True
                break

        if not foundSub: # need to insert a timepoint for this subject
             t.append(subjectsTimeArray[i])
             t.sort()

    # now create the subject index array
    for i in range(numSubjects):
        for j in range(len(t)):
            if subjectsTimeArray[i]<(t[j]+epsilon) and  subjectsTimeArray[i]>(t[j]-epsilon): # matches update Sind
                Sind[i] = j
                break;
    
    cpinds=range(1,len(t))
    return (t,Sind, cpinds)

# end HGMSetUpCheckpointing

if __name__ == '__main__':
    #common.DebugHere()
    try:
        usercfg = Config.Load(spec=HGMConfigSpec, argv=sys.argv)
    except Config.MissingConfigError:
        # Don't freak out, this should have printed a sample config to STDOUT
        sys.exit(1)

    Compute.Compute(BuildHGM, usercfg)  
