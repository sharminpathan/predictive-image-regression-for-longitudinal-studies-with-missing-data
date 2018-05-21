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
# from Libraries import CAvmHGMCommon

# others
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import mpi4py
import itertools
import csv
StudySpec = {
    'numSubjects':
    Config.Param(default=4, required=True,
                    comment="Total number of subjects."),
    'subjectFile':
    Config.Param(default="FilePath.csv",
                 comment="Path to the file that lists details of all subjects every timepoint, as pair of rows of images and times.  Each row should be comma separated."),
    'initializationsFile':
    Config.Param(default=None,
                 comment="Path to the file that lists details of all initializations of initial image and momenta."),    
    'setUnitSpacing':
    Config.Param(default=True,
                 comment="Ignore the spacing in images and set it to (1,1,1)"),
    'setZeroOrigin':
    Config.Param(default=True,
                 comment="Ignore the origin in images and set it to (0,0,0)")
}

GeoRegConfigSpec = {
    'compute': Compute.ComputeConfigSpec,
    'vectormomentum': VMConfig.VMConfigSpec,
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
    '_resource': 'VectorMomentum_GeoReg'}

class GeoRegVariables:
    """
    Parameters for doing  regression.

    The constructor actually does the allocation of all scratch variables
    required for computation of  regression gradients
    """
    def __init__(self, grid, mType, alpha, beta, gamma,  nInv, sigma, StepSize,  integMethod='EULER'):
        """
        Initialize everything with the size and type given
        """        
        self.grid = grid
        self.memtype = mType

        # initial conditions
        self.I0 = ca.Image3D(self.grid, self.memtype)
        self.m0 = ca.Field3D(self.grid, self.memtype)

        # state variables
        self.g = ca.Field3D(self.grid, self.memtype)
        self.ginv = ca.Field3D(self.grid, self.memtype)
        self.m = ca.Field3D(self.grid, self.memtype)
        self.I = ca.Image3D(self.grid, self.memtype)

        # adjoint variables
        self.madj = ca.Field3D(self.grid, self.memtype)
        self.Iadj = ca.Image3D(self.grid, self.memtype)
        self.madjtmp = ca.Field3D(self.grid, self.memtype)
        self.Iadjtmp = ca.Image3D(self.grid, self.memtype)

        # image variables for closed-form template update
        self.sumSplatI =  ca.Image3D(self.grid, self.memtype)
        self.sumJac =  ca.Image3D(self.grid, self.memtype)

        # set up diffOp
        if self.memtype == ca.MEM_HOST:
            self.diffOp = ca.FluidKernelFFTCPU()
        else:
            self.diffOp = ca.FluidKernelFFTGPU()
        self.diffOp.setAlpha(alpha)
        self.diffOp.setBeta(beta)
        self.diffOp.setGamma(gamma)
        self.diffOp.setGrid(self.grid)            

        # some extras
        self.nInv = nInv # for interative update to inverse deformation
        self.integMethod = integMethod
        self.sigma = sigma
        self.stepSize = StepSize

        # TODO: scratch variables to be changed to using managed memory
        self.scratchV1 = ca.Field3D(self.grid, self.memtype)
        self.scratchV2 = ca.Field3D(self.grid, self.memtype)
        self.scratchV3 = ca.Field3D(self.grid, self.memtype)   
        self.scratchV4 = ca.Field3D(self.grid, self.memtype)
        self.scratchV5 = ca.Field3D(self.grid, self.memtype)
        self.scratchV6 = ca.Field3D(self.grid, self.memtype)
        self.scratchV7 = ca.Field3D(self.grid, self.memtype)
        self.scratchV8 = ca.Field3D(self.grid, self.memtype)
        self.scratchV9 = ca.Field3D(self.grid, self.memtype)
        self.scratchV10 = ca.Field3D(self.grid, self.memtype)
        self.scratchV11 = ca.Field3D(self.grid, self.memtype)
        self.scratchI1 = ca.Image3D(self.grid, self.memtype) #only used  for geodesic regression with RK4
        
    # end __init__
# end GeoRegVariables
def BuildGeoReg(cf):
    """Worker for running geodesic estimation on a subset of individuals
    """
    #common.DebugHere()
    localRank = Compute.GetMPIInfo()['local_rank']
    rank = Compute.GetMPIInfo()['rank']

    # prepare output directory
    common.Mkdir_p(os.path.dirname(cf.io.outputPrefix))

    # just one reporter process on each node
    isReporter = rank == 0

    # load filenames and times for all subjects
    (subjectsIds, subjectsImagePaths,subjectsTimes) = GeoRegLoadSubjectsDetails(cf.study.subjectFile)    
    cf.study.numSubjects = len(subjectsIds)
    if isReporter:
        # Output loaded config
        if cf.io.outputPrefix is not None:
            cfstr = Config.ConfigToYAML(GeoRegConfigSpec, cf)
            with open(cf.io.outputPrefix + "parsedconfig.yaml", "w") as f:
                f.write(cfstr)
    
    # if MPI check if processes are greater than number of subjects. it is okay if there are more subjects than processes
    if cf.compute.useMPI and (len(subjectsIds) < cf.compute.numProcesses):
        raise Exception("Please don't use more processes "+
                                "than total number of individuals")

    nodeSubjectsIds = subjectsIds[rank::cf.compute.numProcesses]    
    nodeSubjectsImagePaths = subjectsImagePaths[rank::cf.compute.numProcesses]
    nodeSubjectsTimes = subjectsTimes[rank::cf.compute.numProcesses] 
    
    numLocalSubjects = len(nodeSubjectsImagePaths)
    if cf.study.initializationsFile is not None:
        (subjectsInitialImages, subjectsInitialMomenta) = GeoRegLoadSubjectsInitializations(cf.study.initializationsFile)
        nodeSubjectsInitialImages = subjectsInitialImages[rank::cf.compute.numProcesses] 
        nodeSubjectsInitialMomenta = subjectsInitialMomenta[rank::cf.compute.numProcesses] 

    print 'rank:',rank,', localRank:',localRank,', numberSubjects/TotalSubjects:',len(nodeSubjectsImagePaths),'/',cf.study.numSubjects,', nodeSubjectsImagePaths:',nodeSubjectsImagePaths,', nodeSubjectsTimes:',nodeSubjectsTimes

    # mem type is determined by whether or not we're using CUDA
    mType = ca.MEM_DEVICE if cf.compute.useCUDA else ca.MEM_HOST    
    
    # setting gpuid should be handled in gpu
    # if using GPU  set device based on local rank 
    #if cf.compute.useCUDA:
    #    ca.SetCUDADevice(localRank)
        
    # get image size information
    dummyImToGetGridInfo = common.LoadITKImage(nodeSubjectsImagePaths[0][0], mType)
    imGrid = dummyImToGetGridInfo.grid();
    if cf.study.setUnitSpacing:
        imGrid.setSpacing(ca.Vec3Df(1.0, 1.0, 1.0))
    if cf.study.setZeroOrigin:
        imGrid.setOrigin(ca.Vec3Df(0, 0, 0))
    #del dummyImToGetGridInfo;
    
    # start up the memory manager for scratch variables
    ca.ThreadMemoryManager.init(imGrid, mType, 0)

    # allocate memory     
    p = GeoRegVariables(imGrid, mType, cf.vectormomentum.diffOpParams[0], cf.vectormomentum.diffOpParams[1], cf.vectormomentum.diffOpParams[2], cf.optim.NIterForInverse, cf.vectormomentum.sigma, cf.optim.stepSize,  integMethod=cf.optim.integMethod)
    # for each individual run geodesic regression for each subject
    for i in range(numLocalSubjects):

        # initializations for this subject
        if cf.study.initializationsFile is not None:
            # assuming the initializations are already preprocessed, in terms of intensities, origin and voxel scalings.
            p.I0=common.LoadITKImage(nodeSubjectsInitialImages[i], mType)
            p.m0=common.LoadITKField(nodeSubjectsInitialMomenta[i], mType)
        else:
            ca.SetMem(p.m0,0.0)
            ca.SetMem(p.I0,0.0)
        
        # allocate memory specific to this subject in steps a, b and c        
        # a. create time array with checkpointing info for regression geodesic, allocate checkpoint memory
        (t, msmtinds, cpinds) = GeoRegSetUpTimeArray(cf.optim.nTimeSteps, nodeSubjectsTimes[i], 0.001)
        cpstates =  [(ca.Field3D(imGrid,mType),ca.Field3D(imGrid,mType)) for idx in cpinds]
        # b. allocate gradAtMeasurements of the length of msmtindex for storing residuals
        gradAtMsmts =  [ca.Image3D(imGrid,mType) for idx in msmtinds]        
        
        # c. load timepoint images for this subject
        Imsmts = [common.LoadITKImage(f, mType)
                  if isinstance(f, str) else f for f in nodeSubjectsImagePaths[i]]
        # reset stepsize if adaptive stepsize changed it inside
        p.stepSize = cf.optim.stepSize
        # preprocessimages
        GeoRegPreprocessInput(nodeSubjectsIds[i], cf,p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts)

        # run regression for this subject
        # REMEMBER
        # msmtinds index into cpinds
        # gradAtMsmts is parallel to msmtinds
        # cpinds index into t
        EnergyHistory = RunGeoReg(nodeSubjectsIds[i], cf,p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts)

        # write output images and fields for this subject
        # TODO: BEWARE There are hardcoded numbers inside preprocessing code specific for ADNI/OASIS brain data.
        GeoRegWriteOuput(nodeSubjectsIds[i], cf,p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts, EnergyHistory)

        # clean up memory specific to this subject
        del t,Imsmts, cpinds, cpstates, msmtinds, gradAtMsmts
        
    # end for
    
# end BuildGeoReg
def GeoRegPreprocessInput(subid, cf,p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts):
    for i in range(len(Imsmts)):
        if cf.study.setUnitSpacing:
            Imsmts[i].setSpacing(ca.Vec3Df(1.0, 1.0, 1.0))
        if cf.study.setZeroOrigin:
            Imsmts[i].setOrigin(ca.Vec3Df(0, 0, 0))        
        ca.MulC_I(Imsmts[i],1.0/float(255.0))
    # end for
# end GeoRegPreprocessInput
def GeoRegWriteOuput(subjectId, cf,p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts, EnergyHistory):
    # save initial image and momenta for regression geodesic
    common.SaveITKImage(p.I0, cf.io.outputPrefix + subjectId + "I0.mhd")
    common.SaveITKField(p.m0, cf.io.outputPrefix + subjectId + "m0.mhd")


    # save residual images for regression geodesic
    # TODO:

    # write Energy details
    energyFilename = cf.io.outputPrefix + subjectId + "Energy.csv"
    with open(energyFilename,'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerows(EnergyHistory)
# end GeoRegWriteOuput

def GeoRegSetUpTimeArray(nTimeSteps, subjectsTimeArray, epsilon):
    numSubjects = len(subjectsTimeArray)
    tempT = [x*1./nTimeSteps for x in range(nTimeSteps+1)]
    t = [x*1./nTimeSteps for x in range(nTimeSteps+1)]
    msmtinds = [0]*numSubjects

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

    # now create the subject index array indexed into checkpoint indices, 
    # NOTE: it is different in HGM code subjects array is indexed into time array.
    # Both of the above approaches coincide if checkpointing is done at every time for this code
    cpinds=range(1,len(t))
    for i in range(numSubjects):
        for j in range(len(t)):
            if subjectsTimeArray[i]<(t[j]+epsilon) and  subjectsTimeArray[i]>(t[j]-epsilon): # matches update Sind
                if j!=0:
                    msmtinds[i] = cpinds.index(j)
                else:
                    msmtinds[i] = -1
                break;
    

    return (t,msmtinds, cpinds)

# end GeoRegSetupTimeArray

def RunGeoReg(subid, cf,p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts):
    EnergyHistory = []
    #common.DebugHere()
    for it in range(cf.optim.Niter):
        EnergyHistory = GeoRegIteration(subid, cf,p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts,EnergyHistory, it)
    # end for       
    return EnergyHistory
# end RunGeoReg

def GeoRegPlots(subid, cf,p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts,EnergyHistory):
    """
    Do some summary plots for geodesic regression
    """
    #common.DebugHere()
    fig = plt.figure(1)
    fig.patch.set_facecolor('white')

    #TE = [sum(x) for x in EnergyHistory] 
    TE = [row[0] for row in EnergyHistory] 
    VE = [row[1] for row in EnergyHistory] 
    IE = [row[2] for row in EnergyHistory] 

    plt.subplot(1,3,1)
    plt.plot(TE)
    plt.title('Total Energy')
    plt.hold(False)
    plt.subplot(1,3,2)
    plt.plot(VE)
    plt.title('Vector Energy')
    plt.hold(False)
    plt.subplot(1,3,3)
    plt.plot(IE)
    plt.title('Image Energy')
    plt.hold(False)
    plt.draw()
    plt.show()

    if cf.io.outputPrefix != None: 
        energyFilename = cf.io.outputPrefix + subid + "Energy.pdf"
        plt.savefig(energyFilename)
# end GeoRegPlots


def GeoRegGradient(p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts):
    # shoot the geodesic forward    
    CAvmCommon.IntegrateGeodesic(p.m0,t,p.diffOp, \
                      p.m, p.g, p.ginv,\
                      p.scratchV1, p.scratchV2,p. scratchV3,\
                      cpstates, cpinds,\
                      Ninv=p.nInv, integMethod = p.integMethod, RK4=p.scratchV4,scratchG=p.scratchV5)

    IEnergy = 0.0
    # compute residuals for each measurement timepoint along with computing energy
    for i in range(len(Imsmts)):
        # TODO: check these indexings for cases when timepoint 0 
        # is not checkpointed
        if msmtinds[i] != -1:
            (g,ginv) = cpstates[msmtinds[i]]    
            ca.ApplyH(gradAtMsmts[i],p.I0,ginv)
            ca.Sub_I(gradAtMsmts[i],Imsmts[i])
            # while we have residual, save the image energy
            IEnergy += ca.Sum2(gradAtMsmts[i])/(2*p.sigma*p.sigma*float(p.I0.nVox()))
            ca.DivC_I(gradAtMsmts[i], p.sigma*p.sigma) # gradient at measurement        
        elif msmtinds[i] == -1:
            ca.Copy(gradAtMsmts[i],p.I0)
            ca.Sub_I(gradAtMsmts[i],Imsmts[i])
            # while we have residual, save the image energy
            IEnergy += ca.Sum2(gradAtMsmts[i])/(2*p.sigma*p.sigma*float(p.I0.nVox()))
            ca.DivC_I(gradAtMsmts[i], p.sigma*p.sigma) # gradient at measurement        
    
    # integrate backward
    CAvmCommon.IntegrateAdjoints(p.Iadj,p.madj,\
                                 p.I,p.m,p.Iadjtmp, p.madjtmp,p.scratchV1,\
                                 p.scratchV2,p.scratchV3,\
                                 p.I0,p.m0,\
                                 t, cpstates, cpinds,\
                                 gradAtMsmts,msmtinds,\
                                 p.diffOp,\
                                 p.integMethod, p.nInv, \
                                 scratchV3=p.scratchV7, scratchV4=p.g,scratchV5=p.ginv,scratchV6=p.scratchV8, scratchV7=p.scratchV9, \
                                 scratchV8=p.scratchV10,scratchV9=p.scratchV11,\
                                 RK4=p.scratchV4, scratchG=p.scratchV5, scratchGinv=p.scratchV6,\
                                 scratchI = p.scratchI1)                      
    
    # compute gradient
    ca.Copy(p.scratchV1, p.m0)
    p.diffOp.applyInverseOperator(p.scratchV1)
    # while we have velocity, save the vector energy
    VEnergy = 0.5*ca.Dot(p.m0,p.scratchV1)/float(p.I0.nVox())

    ca.Sub_I(p.scratchV1, p.madj)
    #p.diffOp.applyOperator(p.scratchV1)

    # compute closed from terms for image update    
    # p.Iadjtmp and p.I will be used as scratch images 
    scratchI = p.scratchI1 #reference assigned
    imOnes = p.I #reference assigned
    ca.SetMem(imOnes,1.0)
    ca.SetMem(p.sumSplatI,0.0)
    ca.SetMem(p.sumJac,0.0)
    #common.DebugHere()
    for i in range(len(Imsmts)):
        # TODO: check these indexings for cases when timepoint 0 
        # is not checkpointed
        if msmtinds[i] != -1:
            (g,ginv) = cpstates[msmtinds[i]]        
            CAvmCommon.SplatSafe(scratchI, ginv, Imsmts[i])
            ca.Add_I(p.sumSplatI, scratchI)
            CAvmCommon.SplatSafe(scratchI, ginv, imOnes)
            ca.Add_I(p.sumJac, scratchI)
        elif msmtinds[i]==-1:
            ca.Add_I(p.sumSplatI, Imsmts[i])
            ca.Add_I(p.sumJac, imOnes)
    return (p.scratchV1, p.sumJac, p.sumSplatI, VEnergy, IEnergy)

# end GeoRegGradient 

def GeoRegIteration(subid, cf,p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts,EnergyHistory, it):
    # compute gradient for regression
    (grad_m, sumJac, sumSplatI, VEnergy, IEnergy) = GeoRegGradient(p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts)

    # do energy related stuff for printing and bookkeeping
    #if it>0:
    EnergyHistory.append([VEnergy+IEnergy,VEnergy, IEnergy])
    print VEnergy+IEnergy, '(Total) = ',VEnergy, '(Vector)+',IEnergy,'(Image)'
    # plot some stuff
    if cf.io.plotEvery > 0 and (((it+1) % cf.io.plotEvery) == 0 or it == cf.optim.Niter-1):
        GeoRegPlots(subid, cf,p,t,Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts,EnergyHistory)
    # end if    

    if cf.optim.method == 'FIXEDGD':        
        # automatic stepsize selection in the first three steps
        if it==1:
            # TODO: BEWARE There are hardcoded numbers here for 2D and 3D
            #first find max absolute value across voxels in gradient
            temp = ca.Field3D(grad_m.grid(), ca.MEM_HOST)
            ca.Copy(temp,grad_m)
            temp_x, temp_y, temp_z=temp.asnp()
            temp1=np.square(temp_x.flatten())+np.square(temp_y.flatten())+np.square(temp_z.flatten())
            medianval = np.median(temp1[temp1>0.0000000001])
            del temp, temp1, temp_x, temp_y, temp_z
            #2D images for 2000 iters
            #p.stepSize = float(0.000000002*medianval)
            #3D images for 2000 iters
            p.stepSize = float(0.000002*medianval)

            print 'rank:',Compute.GetMPIInfo()['rank'],', localRank:',Compute.GetMPIInfo()['local_rank'], 'subid: ',subid,' Selecting initial step size in the beginning to be ', str(p.stepSize)

        if it>3:
            totalEnergyDiff = EnergyHistory[-1][0]- EnergyHistory[-2][0]
            if totalEnergyDiff > 0.0:
                if cf.optim.maxPert is not None:
                    print 'rank:',Compute.GetMPIInfo()['rank'],', localRank:',Compute.GetMPIInfo()['local_rank'], 'subid: ',subid,' Reducing stepsize for gradient descent by ', str(cf.optim.maxPert*100),'%. The new step size is ',str(p.stepSize*(1-cf.optim.maxPert))
                    p.stepSize = p.stepSize*(1-cf.optim.maxPert)
        # take gradient descent step
        ca.Add_MulC_I(p.m0,grad_m,-p.stepSize)
    else:
        raise Exception("Unknown optimization scheme: "+cf.optim.optMethod)
    # end if

    # now divide to get new base image
    ca.Div(p.I0, sumSplatI, sumJac)
    
    return (EnergyHistory)
# end GeoRegIteration


def GeoRegLoadSubjectsDetails(filename):
    subjectsIds=[]
    subjectsImagePaths=[]
    subjectsTimes=[]
    with open(filename, "r" ) as f:
        for line1,line2,line3 in itertools.izip_longest(*[f]*3):
            singleSubjectId = line1.rstrip()
            subjectsIds.append(singleSubjectId)
            singleSubjectImagePaths = line2.rstrip().split(",")
            subjectsImagePaths.append(singleSubjectImagePaths)
            singleSubjectTimes = map(float,line3.rstrip().split(","))
            subjectsTimes.append(singleSubjectTimes)
    return (subjectsIds,subjectsImagePaths,subjectsTimes)
# end GeoRegLoadSubjectsDetails

def GeoRegLoadSubjectsInitializations(filename):
    subjectsInitialImages=[]
    subjectsInitialMomenta=[]
    with open(filename, "r" ) as f:
        for line1,line2 in itertools.izip_longest(*[f]*2):
            singleSubjectsInitialImage = line1.rstrip()
            subjectsInitialImages.append(singleSubjectsInitialImage)
            singleSubjectsInitialMomentum = line2.rstrip()
            subjectsInitialMomenta.append(singleSubjectsInitialMomentum)

    return (subjectsInitialImages,subjectsInitialMomenta)

# end GeoRegLoadSubjectsInitializations

if __name__ == '__main__':
    try:
        usercfg = Config.Load(spec=GeoRegConfigSpec, argv=sys.argv)
    except Config.MissingConfigError:
        # Don't freak out, this should have printed a sample config to STDOUT
        sys.exit(1)

    Compute.Compute(BuildGeoReg, usercfg)  
