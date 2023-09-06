from ypstruct import structure
import numpy as np
def run(problem,params):
    
    #problem information
    costfunc=problem.costfunc
    nvar=problem.nvar
    varmin=problem.varmin
    varmax=problem.varmax

    #parameters
    maxit=params.maxit
    npop=params.npop
    beta=params.beta
    gamma=params.gamma
    pc=params.pc
    mu=params.mu
    sigma=params.sigma
    nc=int(np.round(pc*npop/2)*2)

    #initialization--empty individual template
    empty_individual =structure()
    empty_individual.position=None
    empty_individual.cost=None

    #best solution
    bestsol=empty_individual.deepcopy()
    bestsol.cost=np.inf

    #Initialize population
    pop=empty_individual.repeat(npop)
    for i in range(0,npop):
        pop[i].position=np.random.uniform(varmin,varmax,nvar)
        pop[i].cost=costfunc(pop[i].position)
        if pop[i].cost <bestsol.cost:
            bestsol=pop[i].deepcopy()

    #best cost of iterations
    bestcost=np.empty(maxit)
    #Main loop
    for it in range(maxit):
        costs=np.array([x.cost for x in pop])
        avg_cost=np.mean(costs)
        if avg_cost!=0:
            costs=costs/avg_cost
        probs=np.exp(-beta*costs)
        popc=[]
        for k in range(nc//2):
            #Select Parents
            q=np.random.permutation(npop)
            p1=pop[q[0]]
            p2=pop[q[1]]

            #Roulette wheel selection
            p1=pop[roulette_wheel_selection(probs)]
            p2=pop[roulette_wheel_selection(probs)]
            #perform crossover
            c1,c2=crossover(p1,p2,gamma)
            #perform mutataion
            c1=mutate(c1,mu,sigma)
            c2=mutate(c2,mu,sigma)

            #Apply bounds
            apply_bounds(c1,varmin,varmax)
            apply_bounds(c2,varmin,varmax)

            #Evaluate First Offspring
            c1.cost=costfunc(c1.position)
            if c1.cost<bestsol.cost:
                bestsol=c1.deepcopy()
            c2.cost=costfunc(c2.position)
            if c2.cost<bestsol.cost:
                bestsol=c2.deepcopy()

            #Add offsprings to population
            popc.append(c1)
            popc.append(c2)

        #Merge, Sort and select
        pop+=popc
        pop=sorted(pop,key=lambda x:x.cost)
        pop=pop[0:npop]

        #Store Best Cost
        bestcost[it]=bestsol.cost

        #Show Iteration Information
        print(f"Iteration {it}: Best Cost :{bestcost[it]}")

    #output
    out=structure()
    out.pop=pop
    out.bestsol=bestsol
    out.bestcost=bestcost
    return out

def crossover(p1,p2,gamma=0.1):
    c1=p1.deepcopy()
    c2=p2.deepcopy()
    alpha=np.random.uniform(0,1,*c1.position.shape)
    c1.position=alpha*p1.position+(1-alpha)*p2.position
    c2.position=alpha*p2.position+(1-alpha)*p1.position
    return c1,c2

def mutate(x,mu,sigma):
    y=x.deepcopy()
    flag=(np.random.rand(*x.position.shape) <=mu)
    ind=np.argwhere(flag)
    y.position[ind]+=sigma*np.random.randn(*ind.shape)
    return y

def apply_bounds(x,varmin,varmax):
    x.position=np.maximum(x.position,varmin)
    x.position=np.minimum(x.position,varmax)

def roulette_wheel_selection(p):
    c=np.cumsum(p)
    r=np.random.rand()*sum(p)
    ind=np.argwhere(r<=c)
    return ind[0][0]