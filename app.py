import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga

def sphere(x):
    return sum(x**2)+16


#problem definition
problem=structure()
problem.costfunc=sphere
problem.nvar=5
problem.varmin=-10
problem.varmax=10
#Ga parameters
params=structure()
params.maxit=100
params.npop=50
params.beta=1
params.gamma=0.1
params.pc=1
params.mu=0.1
params.sigma=0.1

#run GA
out=ga.run(problem,params)

#results
plt.plot(out.bestcost)
plt.xlim(0,params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm GA')
plt.grid(True)
plt.show()