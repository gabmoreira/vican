"""
    pgo.py
    Gabriel Moreira
    Sep 18, 2023
"""
import time
import numpy as np
import networkx as nx

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs, lsqr, cg, spsolve

from typing import Tuple, Iterable, Callable
from tqdm.auto import tqdm

from linalg import SE3, project_SO3


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
        """
            Vanilla PGO
        """
        self.dtype = dtype

        self.nodes = np.unique([n for e in edges.keys() for n in e])
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
            r_est[node] = project_SO3(evecs[i*3:i*3+3,:])

        # SO(3) ends here SE(3) starts here
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

        print("Done.")
        return pose_est


def bipartite_so3sync(src_edges: dict,
                      constraints: dict,
                      noise_model: Callable, 
                      edge_filter: Callable,
                      maxiter: int,
                      dtype=np.float32) -> dict:
    """
        SO(3) synchronization in bipartite graphs
        with node constraints
    """
    src_nodes = np.unique([n for e in src_edges.keys() for n in e])
    print("Received graph with {} nodes {} edges".format(len(src_nodes), len(src_edges)))
    print("Applying constraints...")
    edges = {}
    for e, v in src_edges.items():
        if edge_filter(v):
            c = 'c' + e[0]               # camera id
            t = 't' + e[1].split('_')[0] # timestamp

            marker_id = e[1].split('_')[1]
            r_m = constraints[marker_id].R()
            r_0 = constraints['0'].R()

            k_r   = noise_model(v)
            kr_c0 = k_r * v['pose'].R() @ r_m @ r_0.T

            if (c,t) in edges.keys():
                edges[c,t]['pose']._R += kr_c0
                edges[c,t]['k_r'] += k_r
            else:
                edges[c,t] = {'pose': SE3(R=kr_c0, t=np.zeros(3)),
                              'k_r' : k_r}

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
    for (c,t), e in edges.items():
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
        r = evecs[:,:3] @ np.linalg.inv(evecs[:3,:3])

        for i in range(n):
            r[i*3:i*3+3,:] = project_SO3(r[i*3:i*3+3,:])
            
        # Update lambda
        RtildeR = pairwise_r @ r
        
        lbd_i    = np.zeros(n*9)
        lbd_j    = np.zeros(n*9)
        lbd_data = np.zeros(n*9)
        
        for i in range(evecs.shape[0] // 3):
            u, s, vt = np.linalg.svd(RtildeR[i*3:i*3+3,:])
            r[i*3:i*3+3,:] = u @ vt
            
            lbd_i[i*9:i*9+9] += 3*i + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
            lbd_j[i*9:i*9+9] += 3*i + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
            lbd_data[i*9:i*9+9] += (u @ np.diag(s) @ u.T).flatten()
            
        lbd = csr_matrix((lbd_data,(lbd_i,lbd_j)), shape=(3*n,3*n))
        
    nodes = np.unique([n for e in edges.keys() for n in e])
    r_est = {}
    for i, node in enumerate(nodes):
        if node[0] == 'c':
            r_est[node[1:]] = r[i*3:i*3+3,:]
        elif  node[0] == 't':
            r_est[node[1:] + '_0'] = r[i*3:i*3+3,:]
    return r_est
    

def large_bipartite_so3sync(src_edges: dict,
                            constraints: dict,
                            noise_model: Callable, 
                            edge_filter: Callable,
                            maxiter: int,
                            dtype=np.float32) -> dict:
    """
        SO(3) synchronization in large bipartite graphs
        with node constraints
    """
    src_nodes = np.unique([n for e in src_edges.keys() for n in e])

    print("Received graph with {} nodes {} edges".format(len(src_nodes),
                                                         len(src_edges)))
    
    print("Applying constraints", end=" ")
    start = time.time()
    # Build new graph by merging measurements from the same time stamp
    edges = {}
    for e, v in src_edges.items():
        if edge_filter(v):
            c = 'c' + e[0]               # camera id
            t = 't' + e[1].split('_')[0] # timestamp

            marker_id = e[1].split('_')[1]
            r_m = constraints[marker_id].R()
            r_0 = constraints['0'].R()

            k_r   = noise_model(v)
            kr_c0 = k_r * v['pose'].R() @ r_m @ r_0.T

            if (c,t) in edges.keys():
                edges[c,t]['pose']._R += kr_c0
                edges[c,t]['k_r'] += k_r
            else:
                edges[c,t] = {'pose': SE3(R=kr_c0, t=np.zeros(3)),
                              'k_r' : k_r}
    print("({:.3f}s).".format(time.time()-start))

    nodes          = np.unique([n for e in edges.keys() for n in e])
    cam_nodes      = [n for n in nodes if n[0] == 'c']
    time_nodes     = [n for n in nodes if n[0] == 't']
    cam_node2idx   = {n : i for i, n in enumerate(cam_nodes)}
    time_node2idx  = {n : i for i, n in enumerate(time_nodes)}
    num_cam_nodes  = len(cam_nodes)
    num_time_nodes = len(time_nodes)
    num_edges      = len(edges)
    print("Bipartite graph: {} cameras, {} timesteps, {} edges.".format(num_cam_nodes,
                                                                        num_time_nodes,
                                                                        num_edges))

    print("Building {}x{} adjacency and {}x{} SO(3) sparse matrices".format(num_cam_nodes,
                                                                            num_time_nodes,
                                                                            3*num_cam_nodes,
                                                                            3*num_time_nodes), end=" ")
    start = time.time()
    # Adjacency matrix triplets
    a_i    = np.zeros(num_edges, dtype=np.int32)
    a_j    = np.zeros(num_edges, dtype=np.int32)
    a_data = np.zeros(num_edges, dtype=dtype)
    # SO(3) pairwise block matrix triplets
    b_i    = np.zeros(9*num_edges, dtype=np.int32)
    b_j    = np.zeros(9*num_edges, dtype=np.int32)
    b_data = np.zeros(9*num_edges, dtype=dtype)

    a, b = 0, 0
    for (c,t), e in edges.items():
        i = cam_node2idx[c]
        j = time_node2idx[t]
        a_i[a]    = i
        a_j[a]    = j
        a_data[a] = e['k_r']
            
        b_i[b:b+9]    = 3*i + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
        b_j[b:b+9]    = 3*j + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
        b_data[b:b+9] = e['pose'].R().flatten()
        b += 9
        a += 1
    print("({:.3f}s).".format(time.time()-start))

    print("Building power graph", end=" ")
    start = time.time()
    pairwise_r_ct = csr_matrix((b_data,(b_i,b_j)), shape=(3*num_cam_nodes, 3*num_time_nodes))
    adj_ct        = csr_matrix((a_data,(a_i,a_j)), shape=(num_cam_nodes, num_time_nodes))
    deg_t         = np.asarray(np.sum(adj_ct, axis=0)).squeeze()

    pairwise_pwr_r = pairwise_r_ct @ diags(1.0 / np.repeat(deg_t, 3), 0) @ pairwise_r_ct.T
    pwr_adj        = adj_ct @ diags(1.0 / deg_t) @ adj_ct.T
    pwr_deg        = np.asarray(np.sum(pwr_adj, axis=-1)).squeeze()
    lbd_c          = diags(np.repeat(pwr_deg, 3), 0)
    print("({:.3f}s).".format(time.time()-start))

    # Main optimization loop
    eigengap = 0
    bar = tqdm(total=maxiter, dynamic_ncols=True, desc='Optimizing') 
    for _ in range(maxiter): 
        if eigengap >= 1e5:
            break
        laplacian = lbd_c - pairwise_pwr_r
        laplacian = 0.5 * (laplacian.T + laplacian) 

        evals, evecs = eigs(laplacian, k=5, sigma=-1e-6)
        evals    = np.real(evals)
        evecs    = np.real(evecs)
        eigengap = abs(evals[3]/evals[2])

        # Update R
        r_c = evecs[:,:3] @ np.linalg.inv(evecs[:3,:3])
        for i in range(num_cam_nodes):
            r_c[i*3:i*3+3,:] = project_SO3(r_c[i*3:i*3+3,:])
                
        # Update lambda
        r_tilde_r = pairwise_pwr_r @ r_c
            
        lbd_c_i    = np.zeros(num_cam_nodes*9, dtype=np.int32)
        lbd_c_j    = np.zeros(num_cam_nodes*9, dtype=np.int32)
        lbd_c_data = np.zeros(num_cam_nodes*9, dtype=dtype)
            
        for i in range(num_cam_nodes):
            u, s, vt = np.linalg.svd(r_tilde_r[i*3:i*3+3,:])
            r_c[i*3:i*3+3,:] = u @ vt
                
            lbd_c_i[i*9:i*9+9] += 3*i + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
            lbd_c_j[i*9:i*9+9] += 3*i + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
            lbd_c_data[i*9:i*9+9] += (u @ np.diag(s) @ u.T).flatten()
            
        lbd_c = csr_matrix((lbd_c_data,(lbd_c_i,lbd_c_j)), shape=(3*num_cam_nodes, 3*num_cam_nodes))

        # Update lambda_T
        r_t = pairwise_r_ct.T @ r_c
        lbd_t_i    = np.zeros(num_time_nodes*9, dtype=np.int32)
        lbd_t_j    = np.zeros(num_time_nodes*9, dtype=np.int32)
        lbd_t_data = np.zeros(num_time_nodes*9, dtype=dtype)

        for i in range(num_time_nodes):
            u, s, vt = np.linalg.svd(r_t[i*3:i*3+3,:])
            r_t[i*3:i*3+3,:] = u @ vt

            lbd_t_i[i*9:i*9+9] += 3*i + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
            lbd_t_j[i*9:i*9+9] += 3*i + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
            lbd_t_data[i*9:i*9+9] += (u @ np.diag(1.0 / s) @ u.T).flatten()

        lbd_t = csr_matrix((lbd_t_data,(lbd_t_i,lbd_t_j)), shape=(3*num_time_nodes,3*num_time_nodes))
        pairwise_pwr_r = pairwise_r_ct @ lbd_t @ pairwise_r_ct.T

        bar.set_postfix(evals0="{:1.3e}".format(evals[0]),
                        evals1="{:1.3e}".format(evals[1]),
                        evals2="{:1.3e}".format(evals[2]),
                        eigengap="{:1.3e}".format(eigengap))
        bar.update()

    bar.close()
    out = {}
    for c, i in cam_node2idx.items():
        out[c[1:]] = r_c[i*3:i*3+3,:]
    for t, i in time_node2idx.items():
        out[t[1:] + '_0'] = r_t[i*3:i*3+3,:]

    return out


def bipartite_se3sync(src_edges: dict,
                      constraints: dict,
                      noise_model_r: Callable,
                      noise_model_t: Callable,
                      edge_filter: Callable,
                      maxiter: int,
                      lsqr_solver: str,
                      dtype=np.float32) -> dict:
    """
        SE(3) synchronization in bipartite graphs
        with node constraints
    """
    r_est = large_bipartite_so3sync(src_edges,
                                    constraints,
                                    noise_model_r,
                                    edge_filter,
                                    maxiter,
                                    dtype)
    
    nodes = []
    edges = {}
    for e, v in src_edges.items():
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
    inc_data = np.zeros(18*len(edges), dtype=dtype)

    a = 0
    print("Building sparse {}x{} incidence matrix".format(3*len(edges),
                                                          3*len(nodes)), end=" ")
    start = time.time()
    for e, v in edges.items():
        c, m = e
        t, marker_id = m.split('_')

        k_t = noise_model_t(v)

        # Contraint between marker '0' and marker marker_id
        p_0_marker_id = constraints['0'] @ constraints[marker_id].inv()
        r_c_marker_id = r_est[c] @ r_est[t + '_0'].T @ p_0_marker_id.R()

        tilde_c_0 = k_t * (v['pose'].t() + r_c_marker_id @ p_0_marker_id.inv().t())

        ei = edge2idx[e]
        t_tilde[ei*3:ei*3+3] += tilde_c_0

        ni, nj = node2idx[c], node2idx[t + '_0']

        inc_i[a:a+9]       = 3 * ei + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
        inc_j[a:a+9]       = 3 * ni + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
        inc_data[a:a+9]    = k_t * np.eye(3, dtype=dtype).flatten()

        inc_i[a+9:a+18]    = 3 * ei + np.array([0,0,0,1,1,1,2,2,2], dtype=np.int32)
        inc_j[a+9:a+18]    = 3 * nj + np.array([0,1,2,0,1,2,0,1,2], dtype=np.int32)
        inc_data[a+9:a+18] = -k_t * (r_est[c] @ constraints['0'].R().T).flatten()

        a += 18

    incidence_r = csr_matrix((inc_data,(inc_i,inc_j)), shape=(3*len(edges), 3*len(nodes)))
    print("({:.3f}s).".format(time.time()-start))

    print("Solving sparse linear system".format(lsqr_solver), end=" ")
    start = time.time()
    if lsqr_solver == "conjugate_gradient":
        t_est, exit_code = cg(incidence_r.T @ incidence_r, incidence_r.T @ t_tilde)
        assert exit_code == 0
    elif lsqr_solver == "direct":
        t_est, istop, itn, normr = lsqr(incidence_r, t_tilde)[:4]
    print("({:.3f}s).".format(time.time()-start))

    # Build solution dictionary
    out = {}
    for n in nodes:
        idx = node2idx[n]
        out[n] = SE3(R=r_est[n], t=t_est[idx*3:idx*3+3])
    print("Done!")

    return out


def object_bipartite_se3sync(src_edges: dict,
                             noise_model_r: Callable,
                             noise_model_t: Callable,
                             edge_filter: Callable,
                             maxiter: int,
                             lsqr_solver: str,
                             dtype=np.float32) -> dict:
    """
        Operates like bipartite_se3sync but for calibrating 
        objects instead of cameras
    """
    edges = {}
    for k, v in src_edges.items():
        t, marker_id = k[1].split('_')
        edges[marker_id, t + '_0'] = {'pose'            : v['pose'].inv(),
                                      'corners'         : v['corners'],
                                      'reprojected_err' : v['reprojected_err'],
                                      'im_filename'     : v['im_filename']}
        
    obj_pose_est = bipartite_se3sync(edges,
                                     constraints={'0' : SE3(pose=np.eye(4))},
                                     noise_model_r=noise_model_r,
                                     noise_model_t=noise_model_t,
                                     edge_filter=edge_filter,
                                     maxiter=maxiter,
                                     lsqr_solver=lsqr_solver,
                                     dtype=dtype)

    # This stores the poses of all the markers in the cube
    obj_pose_est = {k : v for k,v in obj_pose_est.items() if '_' not in k}

    return obj_pose_est
