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
import os


os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
        return torch.tensor(exp)
        #return torch.tensor(exp, requires_grad=True)
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
            c,g = grad_log_prob(dist_obj)
            
            trace[node] = c #updating the trace
            
            if node not in list(Q.keys()): #won't execute if Q is populated
                Q[node] = dist_obj #initialize proposal using prior
            else:
                G[node]=g
                logWv = dist_obj.log_prob(c) - Q[node].log_prob(c)
                #logWv = dist_obj.log_prob(torch.tensor(float(c))) - Q[node].log_prob(torch.tensor(float(c)))
                #logWv = dist_obj.log_prob(c) - Q[node].log_prob(torch.tensor(c))
                sigma['logW'] = sigma['logW'] + logWv #modifying sigma
        
        elif keyword == "observe*": #(observe v e_1 e_2) v=node
            link_expr1 = links[node][1] #e_1
            link_expr1 = plugin_parent_values(link_expr1,trace)
            dist_obj = deterministic_eval(link_expr1)
            link_expr2 = links[node][2] #e_2
            link_expr2 = plugin_parent_values(link_expr2,trace)
            c = deterministic_eval(link_expr2)
            sigma['logW'] = sigma['logW'] + dist_obj.log_prob(c)
            #sigma['logW'] = sigma['logW'] + dist_obj.log_prob(torch.tensor(float(c)))
    
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
    #c = dist.sample()
    #lp = dist.log_prob(torch.tensor(c))
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
                #params.data = params.data + ghat[v][i]/(t+10) #best for program 1
                params.data = params.data + ghat[v][i]/((t+1)*100) #best for program 2
                #params.data = params.data + ghat[v][i]/((t+1)*100) #best for program 4?
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
                    #if i == (L-1): print("x is ", x)
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
            
            C = np.array([ np.cov(Fv[:,j],Gv[:,j], rowvar=True) for j in range(num_params) ])
            cov = [ C[j][1][0] for j in range(num_params) ]
            numerator = sum(cov)
            bhat = numerator/denom

            ghat[v] = sum( np.divide((Fv - bhat*np.array(Gv)),L) )


        #print("returning ghat:", ghat)

        return ghat
    


    sigma = {'Q':{}, 'logW':0, 'G':{} }     
    weighted_samples = []

    for t in range(0,T): # T is the number of iterations
        Glist = []
        logWlist = []
        ELBOlist = []
        normgradlist = []
        
        # here we compute a batch of gradients
        for l in range(0,L): # L is the batch size
            sigma['logW']=0
            r_l, sigma_l, trace_l = sample_from_joint(graph,sigma)
            G_l = copy.deepcopy(sigma_l['G'])
            logW_l = sigma_l['logW']
            Glist.append(G_l)
            logWlist.append(logW_l)

            weighted_samples.append((r_l,logW_l))

        ELBO = sum(logWlist)
        ELBOlist.append(ELBO)
        print("ELBO is", ELBO)

        ghat = elbo_grad(Glist,logWlist)

        sigma['Q'], max_norm = optimizer_step( sigma['Q'],ghat,t) #update the proposal
        normgradlist.append(max_norm)
        print("results on iteration {} are ".format(t), sigma['Q'])
        print("the max gradient is ", max_norm )
        

    print("sigma['Q'] is ",sigma['Q'])

    return weighted_samples, sigma['Q'],ELBOlist,normgradlist

    
    proposal = {**sigma['Q']}
    print(proposal)

    weighted_samples_ = []
    sigma_t = {'Q':proposal, 'logW':0, 'G':{}}

    for t in range(T*10):
        #for l in range(0,L): # L is the batch size
        val_t, sigma_t, trace_t = sample_from_joint(graph,sigma_t)
        logW_t = torch.tensor(sigma_t['logW'].item())
        #print("logW_t is", logW_t)
        weighted_samples_.append((val_t,logW_t))
        sigma_t['logW'] = torch.tensor(0.0)
        #weighted_samples_.append((r_t,logW_t))
        #weighted_samples_.append(list((r_l,logW_l)))

    return weighted_samples_, proposal
    


def normalize(ret):
    weights = []
    samples = []
    for s, w in ret:
        samples.append(s)
        weights.append(w)
    weights = torch.exp(torch.stack(weights))
    sum_weights = torch.sum(weights)
    return samples, weights / sum_weights




def compute_expectation(weighted_samples):
    '''
    using a stream of weighted samples, compute according
    to eq 4.6 
    '''
    L = len(weighted_samples)
    log_weights = [weighted_samples[i][1] for i in range(0,L)]
    weights = np.exp(np.array(log_weights))
    #print(weights)
    r = [ weighted_samples[i][0] for i in range(0,L)]
    denom = sum(weights)
    #print("denominator is", denom)
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
    
    #T = 60 #for program 1

    T = 2000

    #L = 20000
    #L = 200 #for program 1
    L = 20 #for program 2
    #L = 200 #for program 4?


    #graph = daphne(['graph','-i','../HW3/programs/1.daphne'])
    #graph = daphne(['graph','-i','../HW3/programs/2.daphne'])
    graph = daphne(['graph','-i','../HW4/programs/4.daphne'])
    #graph = daphne(['graph','-i','../HW4/programs/4.daphne'])
    #print(graph)
    print("L is ", L)
    print("T is ", T)

    #samples = BBVI(graph,100,8000)
  

    weighted_samples, proposal, ELBO, normgradlist = BBVI(graph,T,L)


    print("proposal is:", proposal)

    dim = len(list(weighted_samples))
    N = dim - 200
    tail = list(weighted_samples)[N:]
    samples, normalized_weights = normalize(tail)
    samples = torch.stack(samples)
    #mean = torch.sum(samples * normalized_weights)
    #variance = torch.sum(samples**2 * normalized_weights) - mean**2
    #print("Mean: ", mean, "Variance: ", variance)

    #figH,axH = plt.subplots()
    #axH.hist(samples)
    #figH.savefig('../HW4/p1histogram',dpi = 150)


    #fig,ax = plt.subplots()
    #ax.plot(normgradlist)
    #fig.savefig('../HW4/p1normgrad',dpi = 150)
    #plt.show()

    #print(weighted_samples)

    #mean = compute_expectation(weighted_samples)

    #print("posterior mean is", mean )

    #var = compute_variance(weighted_samples,mean)

    #print("posterior variance is", var)

    #samples = torch.stack(samples)
    #samples = np.array(samples)
    #samples = samples.detach().numpy()



    



    


    

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