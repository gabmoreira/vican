"""
	pgo.py
	Gabriel Moreira
"""
import time
import copy
import multiprocessing as mp
from itertools import combinations

import cv2 as cv
import numpy as np
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, spsolve
from scipy.sparse import diags
from scipy.stats import vonmises

from time import time
from typing import Tuple, Iterable, Callable
from tqdm.auto import tqdm


def langevin(k_r):
    vec_r = np.random.normal(0,1,size=(3,))
    vec_r = vonmises.rvs(k_r) * vec_r / np.linalg.norm(vec_r, ord=2)
    R = cv.Rodrigues(vec_r)[0]
    return R


def rotx(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0,   c,  -s],
                  [0.0,   s,   c]], dtype=np.float32)
    return R

def roty(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c,   0.0,   s],
                  [0.0, 1.0, 0.0],
                  [-s,  0.0,   c]], dtype=np.float32)
    return R

def rotz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c,    -s, 0.0],
                  [s,     c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return R


def rad2deg(rad: float) -> float:
    return rad * 180 / np.pi

def deg2rad(deg: float) -> float:
    return deg * np.pi / 180

def angle(r: np.ndarray) -> float:
    rad = np.arccos( np.clip((np.trace(r)-1)/2, a_min=-1, a_max=1) )
    return rad2deg(rad)

def so3_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    assert r1.shape == (3,3) and r2.shape == (3,3)
    return angle(r1.T @ r2)


class SE3(object):
    def __init__(self, **kwargs):
        if 'pose' in kwargs.keys():
            self._pose = kwargs['pose'].astype(np.float32)
            self._R = self._pose[:3,:3]
            self._t = self._pose[:3,-1]
        else:
            self._R = kwargs['R']
            self._t = kwargs['t'].flatten()
            self._pose = np.zeros((4,4), dtype=np.float32)
            self._pose[:3,:3] += self._R
            self._pose[:3,-1] += self._t
            self._pose[-1,-1] += 1.0

    def R(self) -> np.ndarray:
        """
            Return SO(3) matrix
        """
        return self._R

    def t(self) -> np.ndarray:
        """
            Return translation
        """
        return self._t
    
    def inv(self):
        """
            Inverse of SE(3) transform 
        """
        inverted = np.zeros_like(self._pose)
        inverted[-1,-1] += 1
        inverted[:3,:3] += self._R.T
        inverted[:3,-1] += -self._R.T @ self._t
        return SE3(pose=inverted)

    def apply(self, x : np.ndarray) -> np.ndarray:
        """
            Apply 3D transformation to 3 x n points
        """
        assert x.ndim == 2
        assert x.shape[0] == 3
        return self._R @ x + self._t.reshape([-1,1])

    def __repr__(self) -> str:
        repr = str(np.round(self._pose, 4))
        return repr

    def __matmul__(self, x):
        return SE3(pose=self._pose @ x._pose)
    

def project_so3(x: np.ndarray):
    u, _, vh = np.linalg.svd(x)
    return u @ vh


def optimize_gauge_so3(poses_a: Iterable[np.ndarray],
                       poses_b: Iterable[np.ndarray]) -> np.ndarray:
    """
    Finds SE(3) transformation G that aligns poses_a with poses_b
    as in min sum|| pose_a - pose_b @ G ||
    """
    assert len(poses_a) == len(poses_b)

    sum = np.zeros((3,3), dtype=np.float64)
    for a, b in zip(poses_a, poses_b):
        sum += a.T @ b
    
    u, _, vh = np.linalg.svd(sum.T)
    gauge_r = u @ vh
    return gauge_r


def optimize_gauge_se3(poses_a: Iterable[SE3],
                       poses_b: Iterable[SE3]) -> SE3:
    """
    Finds SE(3) transformation G that aligns poses_a with poses_b
    as in min sum|| pose_a - pose_b @ G ||
    """
    assert len(poses_a) == len(poses_b)

    sum     = np.zeros((3,3), dtype=np.float64)
    gauge_t = np.zeros((3,1), dtype=np.float64) 
    for a, b in zip(poses_a, poses_b):
        sum += a.R().T @ b.R()
        gauge_t += b.R().T @ (a.t() - b.t()).reshape((-1,1))
    
    u, _, vh = np.linalg.svd(sum.T)
    gauge_r = u @ vh
    gauge = SE3(R=gauge_r, t=gauge_t / len(poses_a))

    return gauge



def sparse_blocks_so3(nodes: Iterable[str],
                      node2idx: dict,
                      edges: dict,
                      std: str,
                      dtype) -> Tuple:
    """
        Builds sparse matrix from pairwise blocks
    """
    assert len(nodes) == len(node2idx)

    n = len(nodes)
    m = len(edges)

    # Adjacency matrix triplets
    a_i    = np.zeros(2*m, dtype=np.int32)
    a_j    = np.zeros(2*m, dtype=np.int32)
    a_data = np.zeros(2*m, dtype=dtype)

    # SO(3) pairwise block matrix triplets
    b_i    = np.zeros(3*3*2*m, dtype=np.int32)
    b_j    = np.zeros(3*3*2*m, dtype=np.int32)
    b_data = np.zeros(3*3*2*m, dtype=dtype)

    b = 0
    a = 0
    for nodes, e in tqdm(edges.items()):
        i = node2idx[nodes[0]]
        j = node2idx[nodes[1]]

        a_i[a]      = i
        a_j[a]      = j
        a_i[a+1]    = j
        a_j[a+1]    = i
        a_data[a]   = e[std]
        a_data[a+1] = e[std]
     
        b_i[b:b+9]       = 3*i + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
        b_j[b:b+9]       = 3*j + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
        b_i[b+9:b+18]    = 3*j + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
        b_j[b+9:b+18]    = 3*i + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
        b_data[b:b+9]    = e[std] * e['pose'].R().flatten()
        b_data[b+9:b+18] = e[std] * e['pose'].R().T.flatten()
     
        b += 18
        a += 2

    blocks = csr_matrix((b_data,(b_i,b_j)), shape=(3*n,3*n))
    adj    = csr_matrix((a_data,(a_i,a_j)), shape=(n,n))

    return blocks, adj



class PGO(object):
    def __init__(self, edges: dict, dtype):
        self.dtype = dtype

        self.nodes = np.unique([n for e in edges.keys() for n in e])
        print("\n----------------* PGO *----------------")
        print('Received {} edges.'.format(len(edges)))
        print('Total of {} nodes.'.format(len(self.nodes)))

        self.graph = nx.Graph()
        self.graph.add_edges_from(edges.keys())
    
        self.edges    = {k:v for k,v in edges.items() if k[0] in self.nodes and k[1] in self.nodes}
        self.node2idx = {n:i for i,n in enumerate(self.nodes)}

        self.n = len(self.nodes)
        self.m = len(self.edges)

        print("Final graph is {}connected.".format('' if nx.is_connected(self.graph) else 'NOT '))
        print("\t{} nodes".format(self.n))
        print("\t{} edges".format(self.m))


    def __getitem__(self, key):
        """
            Override [] operator
        """
        if isinstance(key, str):
            if key in self.nodes:
                return self.graph[key]
            else:
                print("{} is not a node.".format(key))
                return None
        
        elif isinstance(key, tuple):
            if key in self.edges:
                return self.edges[key]
            elif (key[1], key[0]) in self.edges:
                rev_edge = {k:v for k,v in self.edges[key[1], key[0]].items()}
                rev_edge['pose'] = self.edges[key[1], key[0]]['pose'].inv()
                return rev_edge
            else:
                print("{} is not an edge.".format(key))
                return None 
        else:
            print("Unknown key!")
            return None


    def optimize(self,
                 sigma: float=-1e-6) -> dict:
        """
            Motion averaging in SE(3)
        """
        print("Building SO(3) sparse block-matrix...")
        pairwise_r, adj = sparse_blocks_so3(self.nodes, self.node2idx, self.edges, 'k_r', self.dtype)

        # Degree vector
        deg = np.asarray(adj.sum(axis=1)).squeeze()

        lbd_diag = np.zeros(self.n*3, dtype=self.dtype)
        for i in range(self.n):
            lbd_diag[i*3:i*3+3] += deg[i]
        lbd = diags(lbd_diag, 0)

        laplacian = lbd - pairwise_r
        laplacian = 0.5 * (laplacian.T + laplacian) 

        evals, evecs = eigs(laplacian, k=5, sigma=sigma)
        evals = np.real(evals)
        evecs = np.real(evecs)
        print("\tSO(3) Eigenvalues: {}".format(evals))
        print("\tSO(3) Eigengap:    {:1.3e}".format(evals[3]/evals[2]))
        
        evecs = evecs[:,:3] @ np.linalg.inv(evecs[:3,:3])

        r_est = {}
        for i, node in enumerate(self.nodes):
            r_est[node] = project_so3(evecs[i*3:i*3+3,:])

        # SO(3) ends here
        # SE(3) starts here

        pseudoedges = {}
        for nodes, e in self.edges.items():
            pseudoedges[nodes] = {'pose' : SE3(R=r_est[nodes[0]] @ r_est[nodes[1]].T, t=np.zeros((3,1))), 'k_t' : e['k_t']}

        print("Building SO(3) sparse block-matrix...")
        opt_pairwise_r, opt_adj = sparse_blocks_so3(self.nodes, self.node2idx, pseudoedges, 'k_t', self.dtype)

        opt_deg = np.asarray(opt_adj.sum(axis=1)).squeeze()
        opt_lbd_diag = np.zeros(self.n*3, dtype=self.dtype)
        for i in range(self.n):
            opt_lbd_diag[i*3:i*3+3] += opt_deg[i]
        opt_lbd = diags(opt_lbd_diag, 0)

        opt_laplacian = opt_lbd - opt_pairwise_r
        opt_laplacian = 0.5 * (opt_laplacian.T + opt_laplacian) 

        b = np.zeros(3*self.n, dtype=self.dtype)
        for i, k in enumerate(self.nodes):
            for j in self[k]:
                r_k  = r_est[k]
                r_j  = r_est[j]
                k_jk = self[j,k]['k_t']
                t_kj = self[k,j]['pose'].t()
                r_jk = self[j,k]['pose'].R()
                b[i*3:i*3+3] += k_jk * (np.eye(3, dtype=self.dtype) +  r_k @ r_j.T @ r_jk) @ t_kj
        b *= 0.5

        t_est = spsolve(opt_laplacian, b)
        pose_est = {n : SE3(R=r_est[n], t=t_est[i*3:i*3+3]) for i,n in enumerate(self.nodes)}

        print("Done!")
        return pose_est



def bipartite_so3sync(src_edges: dict,
                      constraints: dict,
                      noise_model: Callable, 
                      edge_filter: Callable,
                      maxiter: int,
                      dtype=np.float32) -> dict:
    """
        SO(3) synchronization in bipartite graphs
        with edge constraints
    """
    src_nodes = np.unique([n for e in src_edges.keys() for n in e])
    print("Received graph with {} nodes {} edges".format(len(src_nodes), len(src_edges)))
    print("Applying constraints...")
    edges = {}
    for e, v in tqdm(src_edges.items()):
        if edge_filter(v):
            c = 'c' + e[0]               # camera id
            t = 't' + e[1].split('_')[0] # timestamp

            marker_id = e[1].split('_')[1]
            r_m = constraints[marker_id].R()
            r_0 = constraints['0'].R()

            k_r   = noise_model(v['pose'])
            kr_c0 = k_r * v['pose'].R() @ r_m @ r_0.T

            if (c,t) in edges.keys():
                edges[c,t]['pose']._R += kr_c0
                edges[c,t]['k_r'] += k_r
            else:
                edges[c,t] = {'pose': SE3(R=kr_c0, t=np.zeros(3)),
                              'k_r' : k_r}

    graph = nx.Graph()
    graph.add_edges_from(edges.keys())
    
    if not nx.is_connected(graph):
        print("Error: Graph is disconnected!")
        return None
    
    nodes = np.unique([n for e in edges.keys() for n in e])
    node2idx = {n:i for i,n in enumerate(nodes)}
    n = len(nodes)
    m = len(edges)
    print("New SO(3) graph contains {} nodes {} edges".format(n,m))
    print("Building sparse adjacency and SO(3) connection Laplacian...")
    
    # Adjacency matrix triplets
    a_i    = np.zeros(2*m, dtype=np.int32)
    a_j    = np.zeros(2*m, dtype=np.int32)
    a_data = np.zeros(2*m, dtype=dtype)
    # SO(3) pairwise block matrix triplets
    b_i    = np.zeros(18*m, dtype=np.int32)
    b_j    = np.zeros(18*m, dtype=np.int32)
    b_data = np.zeros(18*m, dtype=dtype)

    a, b = 0, 0
    for (c,t), e in tqdm(edges.items()):
        i = node2idx[c]
        j = node2idx[t]
        a_i[a]      = i
        a_j[a]      = j
        a_i[a+1]    = j
        a_j[a+1]    = i
        a_data[a]   = e['k_r']
        a_data[a+1] = e['k_r']
        
        b_i[b:b+9]       = 3*i + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
        b_j[b:b+9]       = 3*j + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
        b_i[b+9:b+18]    = 3*j + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
        b_j[b+9:b+18]    = 3*i + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
        b_data[b:b+9]    = e['pose'].R().flatten()
        b_data[b+9:b+18] = e['pose'].R().T.flatten()
        
        b += 18
        a += 2

    pairwise_r = csr_matrix((b_data,(b_i,b_j)), shape=(3*n,3*n))
    adj        = csr_matrix((a_data,(a_i,a_j)), shape=(n,n))

    # Initialize lambda (dual variable)
    deg = np.asarray(adj.sum(axis=1)).squeeze()
    lbd_diag = np.zeros(n*3, dtype=dtype)
    for i in range(n):
        lbd_diag[i*3:i*3+3] += deg[i]
    lbd = diags(lbd_diag, 0)

    for it in range(maxiter):
        laplacian = lbd - pairwise_r
        laplacian = 0.5 * (laplacian.T + laplacian) 

        print("Iter {}\n\tComputing eigenvalues...".format(it))
        evals, evecs = eigs(laplacian, k=5, sigma=-1e-6)
        evals = np.real(evals)
        evecs = np.real(evecs)
        print("\tSO(3) Eigenvalues: {}".format(evals))
        print("\tSO(3) Eigengap:    {:1.3e}".format(abs(evals[3]/evals[2])))

        # Update R
        R = evecs[:,:3] @ np.linalg.inv(evecs[:3,:3])

        for i in range(n):
            R[i*3:i*3+3,:] = project_so3(R[i*3:i*3+3,:])
            
        # Update lambda
        RtildeR = pairwise_r @ R
        
        lbd_i    = np.zeros(n*9)
        lbd_j    = np.zeros(n*9)
        lbd_data = np.zeros(n*9)
        
        for i in range(evecs.shape[0] // 3):
            u, s, vt = np.linalg.svd(RtildeR[i*3:i*3+3,:])
            R[i*3:i*3+3,:] = u @ vt
            
            lbd_i[i*9:i*9+9] += 3*i + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
            lbd_j[i*9:i*9+9] += 3*i + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
            lbd_data[i*9:i*9+9] += (u @ np.diag(s) @ u.T).flatten()
            
        lbd = csr_matrix((lbd_data,(lbd_i,lbd_j)), shape=(3*n,3*n))
        
    nodes = np.unique([n for e in edges.keys() for n in e])
    r_est = {}
    for i, node in enumerate(nodes):
        if node[0] == 'c':
            r_est[node[1:]] = R[i*3:i*3+3,:]
        elif  node[0] == 't':
            r_est[node[1:] + '_0'] = R[i*3:i*3+3,:]
    return r_est
    
    

def bipartite_se3sync(src_edges: dict,
                      constraints: dict,
                      noise_model_r: Callable,
                      noise_model_t: Callable,
                      edge_filter: Callable,
                      maxiter: int,
                      dtype=np.float32) -> dict:
    """
        SE(3) synchronization in bipartite graphs
        with edge constraints
    """
    r_est = bipartite_so3sync(src_edges,
                              constraints,
                              noise_model_r,
                              edge_filter,
                              maxiter)
    
    nodes = []
    edges = {}
    for i, (e, v) in enumerate(src_edges.items()):
        if edge_filter(v):
            edges[e] = v
            t, marker_id = e[1].split('_')
            nodes.append(e[0])
            nodes.append(t + '_0')

    nodes = np.unique(nodes)
    node2idx = {n : i for i, n in enumerate(nodes)}
    edge2idx = {e : i for i, e in enumerate(edges.keys())}

    # Stacked measurements
    t_tilde = np.zeros(3*len(edges))

    # Incidence matrix triples
    inc_i    = np.zeros(18*len(edges), dtype=np.int32)
    inc_j    = np.zeros(18*len(edges), dtype=np.int32)
    inc_data = np.zeros(18*len(edges), dtype=np.float32)

    a = 0
    print("Building sparse {}x{} incidence matrix...".format(3*len(edges), 3*len(nodes)))
    for e, v in tqdm(edges.items()):
        c, m = e
        t, marker_id = m.split('_')

        k_t = noise_model_t(v['pose'])

        # Contraint between marker '0' and marker marker_id
        p_0_marker_id = constraints['0'] @ constraints[marker_id].inv()
        r_c_marker_id = r_est[c] @ r_est[t + '_0'].T @ p_0_marker_id.R()

        # \tilde{t}_{c,m^{(t)}} + R_c R_{m^{(t)}}^\top \bar{t}_{m,m_0}
        tilde_c_0 = k_t * (v['pose'].t() + r_c_marker_id @ p_0_marker_id.inv().t())

        ei = edge2idx[e]
        t_tilde[ei*3:ei*3+3] += tilde_c_0

        ni, nj = node2idx[c], node2idx[t + '_0']

        inc_i[a:a+9]       = 3 * ei + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
        inc_j[a:a+9]       = 3 * ni + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
        inc_data[a:a+9]    = k_t * np.eye(3, dtype=np.float32).flatten()

        inc_i[a+9:a+18]    = 3 * ei + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
        inc_j[a+9:a+18]    = 3 * nj + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
        inc_data[a+9:a+18] = -k_t * (r_est[c] @ constraints['0'].R().T).flatten()

        a += 18

    incidence_r = csr_matrix((inc_data,(inc_i,inc_j)), shape=(3*len(edges), 3*len(nodes)))

    print("Solving sparse linear system...")
    t_est = spsolve(incidence_r.T @ incidence_r, incidence_r.T @ t_tilde)

    pose_est = {}
    for i, n in enumerate(nodes):
        idx = node2idx[n]
        pose_est[n] = SE3(R=r_est[n], t=t_est[idx*3:idx*3+3])
    print("Done!")
    return pose_est
    
    
    
    
    
    


class StreamPGO(object):
    def __init__(self, node2idx: dict):
        """
        """
        self.node2idx = node2idx
        self._num_powernodes = len(self.node2idx)

        self._adj_shape  = (self._num_powernodes, self._num_powernodes)
        self._blk_shape  = (3*self._num_powernodes, 3*self._num_powernodes)
        self._pairwise_r = mp.Array('d', np.zeros(self._blk_shape).flatten())
        self._adj        = mp.Array('d', np.zeros(self._adj_shape).flatten())

        self._num_edges  = mp.Value('i', 0)
        self._num_nodes  = mp.Value('i', self._num_powernodes)
        self._updated    = mp.Value('b', False)
        
        self._stop_event = mp.Event()
        self._opt_event  = mp.Event()

        print('StreamPGO started. Call stop() to close.')


    def __getstate__(self):
        state = self.__dict__.copy()

        # Prevent pickling of _update_process process
        state['_update_process'] = None
        return state
    

    def status(self):
        print("Daemon _update PID {}".format(self._update_process.pid))
        print("Graph\n\t{} nodes, {} edges".format(self._num_nodes.value, self._num_edges.value))
        print("Power Graph\n\t{} nodes, {} edges".format(self._num_powernodes, self._num_edges.value))


    def nodes(self):
        """
        """
        out = {n : SE3(R=self._powernodes_r[n], t=self._powernodes_t[n])
               for n in self._powernodes_t.keys()}
        return out


    def run(self, stream):
        """
        """
        self._update_process = mp.Process(target=self._update, args=(stream,), daemon=True)
        self._update_process.start()
        

    def optimize(self):
        """
        """
        #self._optimize_process = mp.Process(target=self._optimize)
        #self._optimize_process.start()
        #self._optimize_process.join()
        return self._optimize()


    def stop(self):
        """
            Stop background all processes
        """
        self._stop_event.set()
        self._update_process.terminate()


    def _update(self, stream: mp.Queue):
        """
            Update matrices with new pairwise measurements
        """
        while not self._stop_event.is_set():
            try:
                edges = stream.get()
            except:
                continue

            powernodes = np.unique([n for e in edges.keys() for n in e if n in self.node2idx.keys()])
            nodes      = np.unique([n for e in edges.keys() for n in e if n not in self.node2idx.keys()])

            if len(powernodes) > 1 and len(nodes) == 1:
                self._num_nodes.value += 1
                nt = nodes[0]

                sum_k_r = np.sum([e['k_r'] for e in edges.values()])

                with self._pairwise_r.get_lock(), self._adj.get_lock(), self._num_edges.get_lock():
                    adj        = np.frombuffer(self._adj.get_obj()).reshape(self._adj_shape)
                    pairwise_r = np.frombuffer(self._pairwise_r.get_obj()).reshape(self._blk_shape)
                    
                    for ni, nj in combinations(powernodes, 2):
                        i, j    = self.node2idx[ni], self.node2idx[nj]
                        r_ij    = edges[ni,nt]['pose'].R() @ edges[nj,nt]['pose'].R().T
                        kij_div = edges[ni,nt]['k_r'] * edges[ni,nt]['k_r'] / sum_k_r

                        # Update adjecency
                        adj[i,j] += kij_div
                        adj[j,i] += kij_div

                        # Update pairwise block matrix
                        pairwise_r[i*3:i*3+3,j*3:j*3+3] += kij_div * r_ij
                        pairwise_r[j*3:j*3+3,i*3:i*3+3] += kij_div * r_ij.T
                        pairwise_r[i*3:i*3+3,i*3:i*3+3] += np.eye(3) / sum_k_r
                        pairwise_r[j*3:j*3+3,j*3:j*3+3] += np.eye(3) / sum_k_r

                        self._num_edges.value += 1

                with self._updated.get_lock():
                    self._updated.value = True


    def _optimize(self):
        """
            Run PGO
        """
        with self._pairwise_r.get_lock(), self._adj.get_lock():
            adj        = np.frombuffer(self._adj.get_obj()).reshape(self._adj_shape)
            pairwise_r = np.frombuffer(self._pairwise_r.get_obj()).reshape(self._blk_shape)
                    
            lbd = 2.0 * np.diag(np.repeat(np.sum(adj, axis=-1), 3))
            evals, evecs = eigs(lbd - pairwise_r, k=3, sigma=-1e-6)

        evals = np.real(evals)
        evecs = np.real(evecs)
        evecs = evecs @ np.linalg.inv(evecs[:3,:3])

        out = {}
        for n,i in self.node2idx.items():
            out[n] = SE3(R=project_so3(evecs[i*3:i*3+3,:]), t=np.zeros(3))
        return out