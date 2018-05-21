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
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import mpi4py

StudySpec = {
    'numSubjects':
    Config.Param(default=4, required=True,
                    comment="Total number of subjects."),
    'subjectIds':
    Config.Param(default=['sid1','sid2','sid3','sid4'], required=True,
                 comment="List of subject ids. This should be unique names for each individuals"),
    'subjectImages':
    Config.Param(default=['subject1_I.mhd','subject2_I.mhd','subject3_I.mhd','subject4_I.mhd'], required=True,
                    comment="List of subject image files"),
    'subjectWeights':
    Config.Param(default=[1.0,1.0,1.0,1.0], required=False,
                    comment="List of weights, w_i (w_i is in [0,1]).")}
AtlasConfigSpec = {
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
    '_resource': 'VectorMomentum_Atlas'}

class WarpVariables:
    """
    Parameters for computing single warp.

    The constructor does the allocation of all scratch variables
    required for computation of  regression gradients
    """
    def __init__(self, grid, mType, alpha, beta, gamma,  nInv, sigma, StepSize,  integMethod='EULER'):
        """
        Initialize everything with the size and type given
        """        
        self.grid = grid
        self.memtype = mType

        # initial conditions
        self.I0 = None # this is a reference that always points to the atlas image
        self.m0 = None # this is a reference that gets assigned to momenta for an individual each time

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
# end WarpVariables

def BuildAtlas(cf):
    """Worker for running Atlas construction on a subset of individuals.
    Runs Atlas on this subset sequentially. The variations retuned are
    summed up to get update for all individuals
    """

    localRank = Compute.GetMPIInfo()['local_rank']
    rank = Compute.GetMPIInfo()['rank']

    # prepare output directory
    common.Mkdir_p(os.path.dirname(cf.io.outputPrefix))

    # just one reporter process on each node
    isReporter = rank == 0
    cf.study.numSubjects = len(cf.study.subjectImages)

    if isReporter:
        # Output loaded config
        if cf.io.outputPrefix is not None:
            cfstr = Config.ConfigToYAML(AtlasConfigSpec, cf)
            with open(cf.io.outputPrefix + "parsedconfig.yaml", "w") as f:
                f.write(cfstr)
    #common.DebugHere()

    # if MPI check if processes are greater than number of subjects. it is okay if there are more subjects than processes

    if cf.compute.useMPI and (cf.study.numSubjects < cf.compute.numProcesses):
        raise Exception("Please don't use more processes "+
                                "than total number of individuals")
    
    # subdivide data, create subsets for this thread to work on
    nodeSubjectIds = cf.study.subjectIds[rank::cf.compute.numProcesses]
    nodeImages = cf.study.subjectImages[rank::cf.compute.numProcesses]
    nodeWeights = cf.study.subjectWeights[rank::cf.compute.numProcesses]

    numLocalSubjects = len(nodeImages)
    print 'rank:',rank,', localRank:',localRank,', nodeImages:',nodeImages,', nodeWeights:',nodeWeights

    # mem type is determined by whether or not we're using CUDA
    mType = ca.MEM_DEVICE if cf.compute.useCUDA else ca.MEM_HOST
    
    # load data in memory
    # load intercepts
    J_array = [common.LoadITKImage(f, mType)
         if isinstance(f, str) else f for f in nodeImages]

    # get imGrid from data
    imGrid = J_array[0].grid()
    
    # atlas image
    atlas = ca.Image3D(imGrid, mType)

    # allocate memory to store only the initial momenta for each individual in this thread
    m_array =  [ca.Field3D(imGrid,mType) for i in range(numLocalSubjects)]

    # allocate only one copy of scratch memory to be reused for each local individual in this thread in loop
    p = WarpVariables(imGrid, mType, cf.vectormomentum.diffOpParams[0], cf.vectormomentum.diffOpParams[1], cf.vectormomentum.diffOpParams[2], cf.optim.NIterForInverse, cf.vectormomentum.sigma, cf.optim.stepSize,  integMethod=cf.optim.integMethod)    

    # memory to accumulate numerators and denominators for atlas from
    # local individuals which will be summed across MPI threads
    sumSplatI = ca.Image3D(imGrid, mType)
    sumJac = ca.Image3D(imGrid, mType)

    # start up the memory manager for scratch variables
    ca.ThreadMemoryManager.init(imGrid, mType, 0)

    # need some host memory in np array format for MPI reductions
    if cf.compute.useMPI:
        mpiImageBuff = None if mType == ca.MEM_HOST else ca.Image3D(imGrid,
                                                                ca.MEM_HOST)

    t = [x*1./(cf.optim.nTimeSteps) for x in range(cf.optim.nTimeSteps+1)]
    cpinds = range(1,len(t))
    msmtinds = [len(t)-2] # since t=0 is not in cpinds, thats just identity deformation so not checkpointed
    cpstates =  [(ca.Field3D(imGrid,mType),ca.Field3D(imGrid,mType)) for idx in cpinds]
    gradAtMsmts =  [ca.Image3D(imGrid,mType) for idx in msmtinds]       

    EnergyHistory = []

    # TODO: better initializations
    # initialize atlas image with zeros.
    ca.SetMem(atlas, 0.0)
    # initialize momenta with zeros

    for m0_individual in m_array:
        ca.SetMem(m0_individual, 0.0)
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

    # preprocessinput

    # assign atlas reference to p.I0. This reference will not change.
    p.I0 = atlas

    # run the loop
    for it in range(cf.optim.Niter):
        # run one iteration of warp for each individual and update
        # their own initial momenta and also accumulate SplatI and Jac
        ca.SetMem(sumSplatI, 0.0)
        ca.SetMem(sumJac, 0.0)
        TotalVEnergy = np.array([0.0])
        TotalIEnergy = np.array([0.0])

        for itsub in range(numLocalSubjects):
            # initializations for this subject, this only assigns
            # reference to image variables
            p.m0 = m_array[itsub]
            Imsmts = [J_array[itsub]]

            # run warp iteration 
            VEnergy, IEnergy = RunWarpIteration(nodeSubjectIds[itsub], cf, p, t, Imsmts, cpinds, cpstates, msmtinds, gradAtMsmts, it)
            
            # gather relevant results        
            ca.Add_I(sumSplatI, p.sumSplatI)
            ca.Add_I(sumJac, p.sumJac)
            TotalVEnergy[0] += VEnergy
            TotalIEnergy[0] += IEnergy

        # if there are multiple nodes we'll need to sum across processes now
        if cf.compute.useMPI:
            # do an MPI sum
            Compute.Reduce(sumSplatI,mpiImageBuff)
            Compute.Reduce(sumJac,mpiImageBuff)
               
            # also sum up energies of other nodes        
            mpi4py.MPI.COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE,
                                            TotalVEnergy,
                                            op=mpi4py.MPI.SUM)
            mpi4py.MPI.COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE,
                                            TotalIEnergy,
                                            op=mpi4py.MPI.SUM)

        EnergyHistory.append([TotalVEnergy[0], TotalIEnergy[0]])

        # now divide to get the new atlas image 
        ca.Div(atlas, sumSplatI, sumJac)
        
        # keep track of energy in this iteration
        if isReporter and cf.io.plotEvery > 0 and (((it+1) % cf.io.plotEvery == 0) or (it == cf.optim.Niter - 1)):
            # plots
            AtlasPlots(cf, p, atlas, m_array, EnergyHistory)

        if isReporter:
            # print out energy
            (VEnergy, IEnergy) = EnergyHistory[-1]
            print "Iter", it, "of", cf.optim.Niter,":" ,VEnergy+IEnergy, '(Total) = ',VEnergy, '(Vector) + ',IEnergy,'(Image)'

    # write output images and fields
    AtlasWriteOutput(cf, atlas, m_array, nodeSubjectIds, isReporter)

# end BuildAtlas


def RunWarpIteration(subid, cf, p, t, Imsmts, cpinds, cpstates, msmtinds, gradAtMsmts, it):

    # compute gradients
    (grad_m, p.sumJac, p.sumSplatI, VEnergy, IEnergy) = WarpGradient(p, t, Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts)

    # TODO: do energy related stuff for printing and bookkeeping
    # WarpPlots()

    if cf.optim.method == 'FIXEDGD':        
        # take gradient descent step
        ca.Add_MulC_I(p.m0,grad_m,-p.stepSize)
    else:
        raise Exception("Unknown optimization scheme: "+cf.optim.optMethod)
    # end if
    return VEnergy, IEnergy

def WarpGradient(p, t, Imsmts,  cpinds, cpstates, msmtinds, gradAtMsmts):

    # shoot the geodesic forward    
    CAvmCommon.IntegrateGeodesic(p.m0,t,p.diffOp, \
                                 p.m, p.g, p.ginv,\
                                 p.scratchV1, p.scratchV2,p. scratchV3,\
                                 cpstates, cpinds,\
                                 Ninv=p.nInv, integMethod = p.integMethod, RK4=p.scratchV4,scratchG=p.scratchV5)

    IEnergy = 0.0
    # compute residuals for each measurement timepoint along with computing energy
    for i in range(len(Imsmts)):
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


def AtlasWriteOutput(cf, atlas, m_array, nodeSubjectsIds, isReporter):
    # save initial momenta for all individuals
    for itsub in range(len(nodeSubjectsIds)):
        common.SaveITKField(m_array[itsub], cf.io.outputPrefix + str(nodeSubjectsIds[itsub]).replace('.','_')+"_m0.mhd")           
            
    # save the atlas
    if isReporter:
        common.SaveITKImage(atlas, cf.io.outputPrefix + "atlas.mhd")

# end AtlasWriteOutput

def AtlasWriteListToFile(listname,filename):    
    try:
        os.remove(filename)
    except OSError:
        pass
    theFile = open(filename, 'w')
    for item in listname:
        print>>theFile, item
    theFile.close()
# end AtlasWriteListToFile

def WarpPlots():
    raise Exception("Individual warp plots not implemented")

def AtlasPlots(cf, p, atlas, m_array, EnergyHistory):
    """
    Do some summary plots for Atlas
    """ 

    fig = plt.figure(1)
    fig.patch.set_facecolor('white')

    TE = [sum(x) for x in EnergyHistory]     
    VE = [row[0] for row in EnergyHistory] 
    IE = [row[1] for row in EnergyHistory] 

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
        energyFilename = cf.io.outputPrefix  + "Energy.pdf"
        plt.savefig(energyFilename)

# end AtlasPlots

if __name__ == '__main__':
    #common.DebugHere()
    try:
        usercfg = Config.Load(spec=AtlasConfigSpec, argv=sys.argv)
    except Config.MissingConfigError:
        # Don't freak out, this should have printed a sample config to STDOUT
        sys.exit(1)

    Compute.Compute(BuildAtlas, usercfg)  
