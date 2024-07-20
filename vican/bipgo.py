"""
    bipgo.py
    Gabriel Moreira
    Sep 18, 2023
"""
import time
import numpy as np

from scipy.sparse import csr_matrix, diags, csgraph
from scipy.sparse.linalg import eigs, lsqr, cg

from typing import Callable
from tqdm.auto import tqdm

from .geometry import SE3, project_SO3


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

    root = str(min(list(constraints.keys())))

    for e, v in src_edges.items():
        if edge_filter(v):
            c = 'c' + e[0]               # camera id
            t = 't' + e[1].split('_')[0] # timestamp

            marker_id = e[1].split('_')[1]
            r_m = constraints[marker_id].R()
            r_0 = constraints[root].R()

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

        Uses primal-dual iteration from VICAN.

        Parameters
        ----------
        src_edges : dict
            Dictionary with camera-object edges.
            Keys should be tuples (str, str) denoting the nodes.
            Example: src_edges["1", "0_5"] = {"pose" : SE3} denotes the edge
            from camera "1" to object node/marker "5", at time 0. Other 
            key-value pairs may be used if needed by noise_model and edge_filter. 
            NOTE: the pose of edge ("1", "0_5") is the pose of object marker "5", 
            at time 0, as seen from the reference frame of camera "1".
        constraints : dict
            Pose constraints to be applied to the object nodes. Keys should 
            be the nodes of the object. Values are SE3 matrices with the 
            corresponding node poses. 
            Example: If the object is a cube with 6 nodes, the constraints dict 
            may have keys "0", "1", "2", "3", "4", "5", and "6". The values are 
            the poses of each of these nodes wrt the world (any reference frame).
        noise_model : Callable
            Function that assigns a scalar to each edge, as the  
            concentration parameter of the Langevin noise model. 
            Example: Let e=src_edges["c", "t_m"]. Then one may use 
            noise_model = lambda : norm(e["pose"].t())
        edge_filter : Callable
            Function that assigns a boolean to each edge, in order to be kept.
            E.g., let e=src_edges["c", "t_m"], edge_filter = 
            lambda : e["reprojected_err"] > threshold
        maxiter : int
            Number of primal-dual iterations
        dtype : type

        Returns
        -------
        out : dict
            Dictionary with camera keys and object node-0 keys containing the 
            rotations of the static cameras and of the object over time wrt
            the world frame, as 3x3 matrices.
    """
    src_nodes = np.unique([n for e in src_edges.keys() for n in e])

    root = str(min(list(constraints.keys())))
    print("Received graph with {} nodes {} edges".format(len(src_nodes),
                                                         len(src_edges)))
    
    print("Applying constraints", end=" ")
    start = time.time()
    edges = {}
    for e, v in src_edges.items():
        if edge_filter(v):
            c = 'c' + e[0]                   # camera id
            t = 't' + e[1].split('_')[0]     # timestamp
            marker_id = e[1].split('_')[1]   # id of the object marker

            r_m = constraints[marker_id].R()
            r_0 = constraints[root].R()

            k_r   = noise_model(v)
            kr_c0 = k_r * v['pose'].R() @ r_m.T @ r_0

            if (c,t) in edges.keys():
                edges[c,t]['pose']._R += kr_c0
                edges[c,t]['k_r'] += k_r
            else:
                edges[c,t] = {'pose': SE3(R=kr_c0,
                                          t=np.zeros(3)),
                              'k_r' : k_r}
                
    print("({:.3f}s).".format(time.time()-start))

    nodes = np.unique([n for e in edges.keys() for n in e])
    nodes2idx = {n : i for i, n in enumerate(nodes)}
    graph = np.zeros((len(nodes), len(nodes)), dtype=np.int32)
    for (c, t) in edges.keys():
        graph[nodes2idx[c], nodes2idx[t]] += 1
        graph[nodes2idx[t], nodes2idx[c]] += 1
    
    print(f"Separated components: {csgraph.connected_components(graph, directed=False)[0]}")
    to_filter = np.where(csgraph.connected_components(graph, directed=False)[1] != 0)[0]
    to_filter = [nodes[i] for i in to_filter]
    print(f"Filtering {len(to_filter)} nodes: {to_filter}")

    for n in to_filter:
        for (c, t) in list(edges.keys()):
            if c == n or t == n:
                del edges[c,t]

    new_src_edges = {}
    for e, v in list(src_edges.items()):
        cam = 'c' + e[0]
        timestep = 't' + e[1].split('_')[0]
        if cam not in to_filter and timestep not in to_filter:
            new_src_edges[e] = v

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
    max_eval = 1.0
    bar = tqdm(total=maxiter, dynamic_ncols=True, desc='Optimizing') 
    for _ in range(maxiter): 
        if max_eval <= 1e-6:
            break
        laplacian = lbd_c - pairwise_pwr_r
        laplacian = 0.5 * (laplacian.T + laplacian) 

        evals, evecs = eigs(laplacian, k=5, sigma=-1e-6)
        evals    = np.real(evals)
        evecs    = np.real(evecs)
        eigengap = abs(evals[3]/evals[2])
        max_eval = np.abs(evals).max()
        
        # Update R
        r_c = evecs[:,:3] @ np.linalg.inv(evecs[:3,:3])
        for i in range(num_cam_nodes):
            r_c[i*3:i*3+3,:] = project_SO3(r_c[i*3:i*3+3,:])
                
        # Update Lambda_C
        r_tilde_r = pairwise_pwr_r @ r_c
            
        lbd_c_i    = np.zeros(num_cam_nodes*9, dtype=np.int32)
        lbd_c_j    = np.zeros(num_cam_nodes*9, dtype=np.int32)
        lbd_c_data = np.zeros(num_cam_nodes*9, dtype=dtype)
            
        for i in range(num_cam_nodes):
            u, s, vt = np.linalg.svd(r_tilde_r[i*3:i*3+3,:])
            r_c[i*3:i*3+3,:] = u @ np.diag([1,1,np.linalg.det(u @ vt)]) @ vt
                
            lbd_c_i[i*9:i*9+9] += 3*i + np.repeat([0,1,2],3)
            lbd_c_j[i*9:i*9+9] += 3*i + np.tile([0,1,2],3)
            lbd_c_data[i*9:i*9+9] += (u @ np.diag(s) @ u.T).flatten()
            
        lbd_c = csr_matrix((lbd_c_data,(lbd_c_i,lbd_c_j)), shape=(3*num_cam_nodes,
                                                                  3*num_cam_nodes))

        # Update Lambda_T
        r_t = pairwise_r_ct.T @ r_c
        lbd_t_i    = np.zeros(num_time_nodes*9, dtype=np.int32)
        lbd_t_j    = np.zeros(num_time_nodes*9, dtype=np.int32)
        lbd_t_data = np.zeros(num_time_nodes*9, dtype=dtype)

        for i in range(num_time_nodes):
            u, s, vt = np.linalg.svd(r_t[i*3:i*3+3,:])
            r_t[i*3:i*3+3,:] = u @ np.diag([1,1,np.linalg.det(u @ vt)]) @ vt

            lbd_t_i[i*9:i*9+9] += 3*i + np.repeat([0,1,2],3)
            lbd_t_j[i*9:i*9+9] += 3*i + np.tile([0,1,2],3)
            lbd_t_data[i*9:i*9+9] += (u @ np.diag(1.0 / s) @ u.T).flatten()

        lbd_t = csr_matrix((lbd_t_data,(lbd_t_i,lbd_t_j)), shape=(3*num_time_nodes,
                                                                  3*num_time_nodes))
        # Update power graph block matrix
        pairwise_pwr_r = pairwise_r_ct @ lbd_t @ pairwise_r_ct.T

        bar.set_postfix(evals0="{:1.3e}".format(evals[0]),
                        evals1="{:1.3e}".format(evals[1]),
                        evals2="{:1.3e}".format(evals[2]),
                        eigengap="{:1.3e}".format(eigengap))
        bar.update()
    bar.close()

    # Invert results so that rotations are wrt the world
    out = {}
    for c, i in cam_node2idx.items():
        out[c[1:]] = r_c[i*3:i*3+3,:].T
    for t, i in time_node2idx.items():
        out[t[1:] + '_0'] = r_t[i*3:i*3+3,:].T

    return out, new_src_edges


def bipartite_se3sync(src_edges: dict,
                      constraints: dict,
                      noise_model_r: Callable,
                      noise_model_t: Callable,
                      edge_filter: Callable,
                      maxiter: int,
                      lsqr_solver: str,
                      dtype=np.float32) -> dict:
    """
        SO(3) synchronization in large bipartite graphs
        with node constraints. Uses large_bipartite_so3sync
        for rotation synchronization and least-squares to solve
        for the translations. Outputs dictionary of poses wrt the world.

        Parameters
        ----------
        src_edges : dict
            Dictionary with camera-object edges. Keys should
            be tuples (str, str) denoting the nodes.
            Example: src_edges["1", "0_5"] = {"pose" : SE3} denotes the edge
            from camera "1" to object node/marker "5", at time 0. Other 
            key-value pairs may be used if needed by noise_model and edge_filter. 
            NOTE: the pose of edge ("1", "0_5") is the pose of object marker "5", 
            at time 0, as seen from the reference frame of camera "1".
        constraints : dict
            Pose constraints to be applied to the object nodes. Keys should  
            be the nodes of the object. Values are SE3 matrices with the  
            corresponding node poses. 
            Example: If the object is a cube with 6 nodes, the constraints dict 
            may have keys "0", "1", "2", "3", "4", "5", and "6". The values are the 
            poses of each of these nodes in any reference frame.
        noise_model_r : Callable
            Function that assigns a scalar to each edge, as the  
            concentration parameter of the Langevin noise model. 
            Example: Let e=src_edges["c", "t_m"]. One may use 
            noise_model = lambda : np.exp(-norm(e["pose"].t())).
        noise_model_t : Callable
            Function that assigns a scalar to each edge, as the precision 
            parameter of the Gaussian noise model. 
            Example: Let e=src_edges["c", "t_m"]. One may use 
            noise_model = lambda : np.exp(-norm(e["pose"].t())).
        edge_filter : Callable
            Function that assigns a boolean to each edge, in order to be kept.
            Example: Let e=src_edges["c", "t_m"]. One may one 
            edge_filter = lambda : e["reprojected_err"] > threshold.
        maxiter : int
            Number of primal-dual iterations.
        lsqr_solver : str
            "direct" for small problems or "conjugate_gradient" for large ones.
        dtype : type

        Returns
        -------
        out : dict
            Dictionary with camera keys and object node-0 keys containing the 
            SE3 poses of the static cameras and of the object over time
            wrt world frame.
    """
    root = str(min(list(constraints.keys())))

    r_est, src_edges = large_bipartite_so3sync(src_edges,
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

        r_0_marker_id = constraints[root].R().T @ constraints[marker_id].R()
        t_marker_id_0 = (constraints[marker_id].inv() @ constraints[root]).t()

        tilde_c_0 = k_t * (r_est[c] @ v['pose'].t() + \
                           r_est[t + '_0'] @ r_0_marker_id @ t_marker_id_0)

        ei = edge2idx[e]
        ni = node2idx[c]
        nj = node2idx[t + '_0']

        t_tilde[ei*3:ei*3+3] += tilde_c_0

        inc_i[a:a+9]       = 3 * ei + np.repeat([0,1,2],3)
        inc_j[a:a+9]       = 3 * ni + np.tile([0,1,2],3)
        inc_data[a:a+9]    = -k_t * np.eye(3, dtype=dtype).flatten()
        inc_i[a+9:a+18]    = 3 * ei + np.repeat([0,1,2],3)
        inc_j[a+9:a+18]    = 3 * nj + np.tile([0,1,2],3)
        inc_data[a+9:a+18] = k_t * np.eye(3, dtype=dtype).flatten()
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

    # Results are poses wrt world frame
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
        objects instead of cameras. Assumes a single object and
        a moving camera. Only outputs poses of object nodes/markers.

        Parameters
        ----------
        src_edges : dict
            Dictionary with camera-object edges. Keys should be
            tuples (str, str) denoting the nodes. Format should be (t, t_m), 
            where t is the timestep and m the object marker/node
            Example: src_edges["3", "3_5"] = {"pose" : SE3} denotes the edge
            at time 3 from the camera to to object node/marker "5". Other 
            key-value pairs may be used if needed by noise_model and edge_filter.
            NOTE: the pose of edge ("3", "3_5") is the pose of object marker "5", 
            at time 3, as seen from the reference frame of the camera.

        Returns
        -------
        out : dict 
            Dictionary of SE3 object poses wrt world frame. Keys
            are the IDs of object nodes / markers.
    """
    edges = {}
    root = str(min([int(e[1].split('_')[1]) for e in src_edges.keys()]))

    for k, v in src_edges.items():
        t, marker_id = k[1].split('_')
        edges[marker_id, t + '_' + root] = {'pose'            : v['pose'].inv(),
                                            'corners'         : v['corners'],
                                            'reprojected_err' : v['reprojected_err'],
                                            'im_filename'     : v['im_filename']}
        
    
    out = bipartite_se3sync(edges,
                            constraints={root : SE3(pose=np.eye(4))},
                            noise_model_r=noise_model_r,
                            noise_model_t=noise_model_t,
                            edge_filter=edge_filter,
                            maxiter=maxiter,
                            lsqr_solver=lsqr_solver,
                            dtype=dtype)

    out = {k : v for k,v in out.items() if '_' not in k}

    return out
