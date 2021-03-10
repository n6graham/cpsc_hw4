from os import get_inheritable
from types import new_class
from numpy.core.numeric import _tensordot_dispatcher
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Uniform, Normal
from torch.distributions.transforms import CatTransform
from distributions import *

from daphne import daphne
import numpy as np
import copy

from primitives import PRIMITIVES
env = PRIMITIVES

from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
# env = {'normal': dist.Normal,
#        'sqrt': torch.sqrt}


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) in [int, float]:
        # We use torch for all numerical objects in our evaluator
        return torch.Tensor([float(exp)]).squeeze()
    elif type(exp) is torch.Tensor:
        return exp
    elif type(exp) is bool:
        return torch.tensor(exp, requires_grad=True)
    else:
        print("expression is:", exp)
        print(type(exp))
        raise Exception("Expression type unknown.", exp)

def topological_sort(nodes, edges):
    result = []
    visited = {}
    def helper(node):
        if node not in visited:
            visited[node] = True
            if node in edges:
                for child in edges[node]:
                    helper(child)
            result.append(node)
    for node in nodes:
        helper(node)
    return result[::-1]

def plugin_parent_values(expr, trace):
    if type(expr) == str and expr in trace:
        return trace[expr]
    elif type(expr) == list:
        return [plugin_parent_values(child_expr, trace) for child_expr in expr]
    else:
        return expr

def sample_from_joint(graph,sigma):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    """
    1. Run topological sort on V using V and A, resulting in an array of v's
    2. Iterate through sample sites of the sorted array, and save sampled results on trace dictionary using P and Y
    - If keyword is sample*, first recursively replace sample site names with trace values in the expression from P. Then, run deterministic_eval.
    - If keyword is observe*, put the observation value in the trace dictionary
    3. Filter the trace dictionary for things sample sites you should return
    """
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)

    #sigma = {'G':{}}
    #sigma = {}
    Q = sigma['Q']
    G = sigma['G']
    #logW = sigma['logW']
    trace = {}
    
    for node in sorted_nodes:
        #print("current node: ",node)
        keyword = links[node][0]
        #print("keyword is:", keyword)
        if keyword == "sample*":
            link_expr = links[node][1] #e, expression to evaluate for sampling
            link_expr = plugin_parent_values(link_expr, trace) 
            dist_obj  = deterministic_eval(link_expr) #prior!!!
            #print("dist_obj is", dist_obj) #aka p
            c,g = grad_log_prob(dist_obj)
            trace[node] = c #updating the trace
            #trace[node] = dist_obj.sample() #updating the trace
            if node not in list(Q.keys()): #won't execute if Q is populated
                Q[node] = dist_obj #initialize proposal using prior
                # not sure if I should update the trace here
            else:
                #c = dist_obj.sample()
                #trace[node] = c # not sure if I should do this
                G[node]=g
                #G[node] = dist_obj.log_prob(c).backward() #modifying sigma
                logWv = dist_obj.log_prob(c) - Q[node].log_prob(c)
                sigma['logW'] = sigma['logW'] + logWv #modifying sigma
        elif keyword == "observe*": #(observe v e_1 e_2) v=node
            #trace[node] = obs[node]
            #print("node is", node)
            link_expr1 = links[node][1] #e_1
            #print("link_expr1:", link_expr1)
            #print("trace is: ", trace)
            link_expr1 = plugin_parent_values(link_expr1,trace)
            #print("link_expr1:", link_expr1)
            dist_obj = deterministic_eval(link_expr1)
            link_expr2 = links[node][2] #e_2
            link_expr2 = plugin_parent_values(link_expr2,trace)
            c = deterministic_eval(link_expr2)
            sigma['logW'] = sigma['logW'] + dist_obj.log_prob(c)
    
    expr = plugin_parent_values(expr, trace)

    return deterministic_eval(expr), sigma, trace


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)


#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../HW3/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')



def run_probabilistic_tests():
    
    #TODO: 
    num_samples = 1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../HW3/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        print(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


def print_tensor(tensor):
    tensor = np.round(tensor.numpy(), decimals=3)
    print(tensor)


def grad_log_prob(dist):
    dist = dist.make_copy_with_grads()
    c = dist.sample()
    lp = dist.log_prob(c)
    lp.backward()
    return c, torch.tensor([v.grad for v in dist.Parameters()])


def BBVI(graph,T,L):

    expr = graph[2]
    sigma = {'G':{}, 'Q':{}, 'logW':0}


    #pre-evaluating all the link functions to remove sample/observe statements
    _, sigma, trace = sample_from_joint(graph,sigma) # all deteministic link functions
    
    
    def optimizer_step(q, ghat,t):
        
        #print("before renormalization ghat is", ghat)
        '''
        for v,d in ghat.items():
            norm = sum(ghat[v])
            ghat[v] = np.divide(d,norm)
        '''
        #print("after renormalization ghat is", ghat)
        #print("q before update is ", q)
        for v,d in q.items():
            print("norm of gradient:", np.linalg.norm(ghat[v]))
            i = 0
            for params in d.Parameters():
                params.data = params.data + ghat[v][i]/(t+10)
                #params.data = params.data + ghat[v][i]/((t+1)*10)
                i = i+1
        #print("q after update is", q)

        max_norm = np.max([ np.linalg.norm(ghat[v]) for v,d in q.items()])





        return q,max_norm


    def elbo_grad(Glist,logWlist): 
        L = len(Glist)
        Flist = list([{} for i in range(0,L)])
        ghat = {}
        U = list(set([u for G in Glist for u in G]))


        for v in U:
            # get number of parameters for v
            for i in range(0,L):
                if v in list(Glist[i].keys()):
                    num_params = len(Glist[i][v])
                    break

            for i in range(0,L):
                if v in list(Glist[i].keys()):
                    x = Glist[i][v]*logWlist[i]
                    if i == (L-1): print("x is ", x)
                    Flist[i][v] = x 
                else:
                    Flist[i][v] = torch.tensor([0 for j in range(num_params)])
                    Glist[i][v] = torch.tensor([0 for j in range(num_params)])

            Fv = [ Flist[i][v] for i in range(0,L)]
            Gv = [ Glist[i][v] for i in range(0,L)]
            Fv = torch.stack(Fv)
            Gv = torch.stack(Gv)
            Fv = Fv.detach().numpy()

            varG = [ np.var(np.array(Gv[:,j])) for j in range(num_params) ]
            denom = sum(varG)
            
            C = np.array([ np.cov(Fv[:,j],Gv[:,j], bias=True) for j in range(num_params) ])
            numerator = np.array([ np.sum(C[j]) for j in range(num_params) ])

            bhat = numerator/denom
            ghat[v] = sum( np.divide((Fv - bhat*np.array(Gv)),L) )


        print("returning ghat:", ghat)

        return ghat
    

    #Q = {**P} #not sure about this
    #sigma = {'Q':Q, 'logW':0, 'G':{} } 
    sigma = {'Q':{}, 'logW':0, 'G':{} } 
    #G = sigma['G']
    #logW = sigma['logW']
    
    weighted_samples = []

    for t in range(0,T): # T is the number of iterations
        Glist = []
        logWlist = []
        # here we compute a batch of gradients
        for l in range(0,L): # L is the batch size
            sigma['logW']=0
            # first we get the trace and update sigma using sample from joint
            val_l, sigma_l, trace_l = sample_from_joint(graph,sigma)
            # then we get the deterministic expression using the trace
            deterministic_expr = plugin_parent_values(expr,trace_l)
            # then we get the value r from calling deterministic eval
            #print("deterministic_expr", deterministic_expr)
            r_l = deterministic_eval(deterministic_expr)
            G_l = copy.deepcopy(sigma_l['G'])
            logW_l = sigma_l['logW']
            Glist.append(G_l)
            logWlist.append(logW_l)



        ghat = elbo_grad(Glist,logWlist)

        sigma['Q'], max_norm = optimizer_step( sigma['Q'],ghat,t) #update the proposal
        print("results on iteration {} are ".format(t), sigma['Q'])
        print("the max gradient is ", max_norm )
        weighted_samples.append((r_l,logW_l))

    #for t in range(N):




    print("sigma['Q'] is ",sigma['Q'])

    return weighted_samples
    


def compute_expectation(weighted_samples):
    '''
    using a stream of weighted samples, compute according
    to eq 4.6 
    '''
    L = len(weighted_samples)
    log_weights = [weighted_samples[i][1] for i in range(0,L)]
    weights = np.exp(np.array(log_weights))
    print(weights)
    r = [ weighted_samples[i][0] for i in range(0,L)]
    denom = sum(weights)
    print("denominator is", denom)
    numerator = sum( [r[i] * weights[i] for i in range(0,L) ])
    #numerator = sum( [weighted_samples[i][0] * weighted_samples[i][1] for i in range(0,L) ])

    return numerator/denom


def compute_variance(weighted_samples, mu):

    L = len(weighted_samples)
    log_weights = [weighted_samples[i][1] for i in range(0,L)]
    weights = np.exp(np.array(log_weights))
    r = [ weighted_samples[i][0] for i in range(0,L)]
    denom = sum(weights)
    numerator = sum( [ (torch.square(r[i]) - torch.square(mu)) * weights[i] for i in range(0,L) ])
    #numerator = sum( [ ( weighted_samples[i][0].long() - torch.square(expectation) ) * weighted_samples[i][1] for i in range(0,L) ] )
    return numerator/denom


'''

def MH_Gibbs(graph, numsamples):
    model = graph[1]
    vertices = model['V']
    arcs = model['A']
    links = model['P'] # link functions aka P

    # sort vertices for ancestral sampling
    V_sorted = topological_sort(vertices,arcs)

    def accept(x, cX, cXnew, Q):
        # compute acceptance ratio to decide whether
        # we keep cX or accept a new sample/trace cXnew
        # cX and cXnew are the proposal mappings (dictionaries)
        # which assign values to latent variables

        # cXnew corresponds to the values for the new samples

        # take the proposal distribution for the current vertex
        # this is Q(x)
        Qx = Q[x][1]

        # we will sample from this with respect to cX and cXnew

        # the difference comes from how we evaluate parents 
        # plugging into eval
        p = plugin_parent_values(Qx,cX)
        pnew = plugin_parent_values(Qx,cXnew)

        # p = Q(x)[X := \mathcal X]
        # p' = Q(x)[X := \mathcal X']
        # note that in this case we only need to worry about
        # the parents of x to sample from the proposal


        # evaluate 
        d = deterministic_eval(p) # d = EVAL(p)
        dnew = deterministic_eval(pnew) #d' = EVAL(p')


        ### compute acceptance ratio ###

        # initialize log alpha

        logAlpha = dnew.log_prob(cXnew[x]) - d.log_prob(cX[x])

        ### V_x = {x} \cup {v:x \in PA(v)} ###
        startindex = V_sorted.index(x)
        Vx = V_sorted[startindex:]

        # compute alpha
        for v in Vx:
            Pv = links[v] 
            v_exp = plugin_parent_values(Pv,cX) 
            dv_new = deterministic_eval(v_exp_new[1])
            #same as we did for p and pnew
            v_exp_new = plugin_parent_values(Pv,cXnew)
            dv_new = deterministic_eval(v_exp_new[1])
            dv = deterministic_eval(v_exp[1])
            
            ## change below
            logAlpha = logAlpha + dv_new.log_prob(cXnew[v])
            logAlpha = logAlpha - dv.log_prob(cX[v])
        return torch.exp(logAlpha)

    
    def Gibbs_step(cX,Q):
        # here we need a list of the latent (unobserved) variables
        Xobsv = list(filter(lambda v: links[v][0] == "sample*", V_sorted))
        #print("Xobsv", Xobsv)

        #print("cX inside gibb-step is", cX)
        for u in Xobsv:
            # here we are doing the step
            # d <- EVAL(Q(u) [X := \cX]) 
            # note it suffices to consider only the non-observed variables
            Qu = Q[u][1]
            u_exp = plugin_parent_values(Qu,cX)
            dist_u = deterministic_eval(u_exp).sample()
            cXnew = {**cX}
            cX[u] = dist_u

            #compute acceptance ratio
            alpha = accept(u,cX,cXnew,Q)
            val = Uniform(0,1).sample()

            if val < alpha:
                cX = cXnew
        return cX


    Q = links # initialize the proposal with P (i.e. using the prior)
    cX_list = [ sample_from_joint(graph)[2] ] # initialize the state/trace

    for i in range(1,numsamples):
        cX_0 = {**cX_list[i-1]} #make a copy of the trace
        cX = Gibbs_step(cX_0,Q)
        cX_list.append(cX)
    
    samples = list(map(lambda cX: deterministic_eval(plugin_parent_values(graph[2], cX)), cX_list))
    #samples = [ deterministic_eval(plugin_parent_values(graph[2],X)) for X in cX_list ]

    return samples

'''

'''
def compute_log_joint(sorted_nodes, links, trace_dict):
    joint_log_prob = 0.
    for node in sorted_nodes:
        link_expr = links[node][1]
        dist      = deterministic_eval(plugin_parent_values(link_expr, trace_dict))
        joint_log_prob += dist.log_prob(trace_dict[node])
    return joint_log_prob
'''

if __name__ == '__main__':
    

    #run_deterministic_tests()
    #run_probabilistic_tests()
    
    T = 10
    L = 20000


    graph = daphne(['graph','-i','../HW3/programs/1.daphne'])
    print(graph)
    print("L is ", L)
    print("T is ", T)

    #samples = BBVI(graph,100,8000)
    samples = BBVI(graph,T,L)

    M = T/4

    print("M is", M)

    tail_index = L*M

    tail_samples = samples[tail_index:]

    


    

    '''
    samples = MH_Gibbs(graph, 10000)
    print("mean for program 1 is: ", np.mean(samples))
    print("variance for program 1 is: ", np.var(samples))
    

    graph = daphne(['graph','-i','../HW3/programs/2.daphne'])
    #print(graph)
    samples = MH_Gibbs(graph, 10000)
    samples1 = [ s[0] for s in samples]
    samples2 = [s[1] for s in samples]
    print("slope mean is: ",np.mean(samples1))
    print("slope variance is: ",np.var(samples1))
    print("bias mean is: ", np.mean(samples2))
    print("bias variance is: ", np.var(samples2))
    #print(samples2)
    

    for i in range(3,5):
        graph = daphne(['graph','-i','../HW3/programs/{}.daphne'.format(i)])
        samples = MH_Gibbs(graph, 10000)
        #print(samples)
        print("the mean for program ", i, " is: ", np.mean(samples))
        print("the variance for program ", i, " is: ", np.var(samples))


    
    for i in range(1,5):
        graph = daphne(['graph','-i','../HW3/programs/{}.daphne'.format(i)])
        samples, n = [], 1000
        for j in range(n):
            sample = sample_from_joint(graph)[0]
            samples.append(sample)

        print(f'\nExpectation of return values for program {i}:')
        if type(samples[0]) is list:
            expectation = [None]*len(samples[0])
            for j in range(n):
                for k in range(len(expectation)):
                    if expectation[k] is None:
                        expectation[k] = [samples[j][k]]
                    else:
                        expectation[k].append(samples[j][k])
            for k in range(len(expectation)):
                print_tensor(sum(expectation[k])/n)
        else:
            expectation = sum(samples)/n
            print_tensor(expectation)
    '''