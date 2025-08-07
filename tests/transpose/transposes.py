
# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

import  random
import  numpy                                                  as  np
from    os         import  system                              as  os_system
from    os.path    import  join                                as  pjoin
from    os         import  listdir                             as  os_listdir
from    os.path    import  abspath, dirname, basename, isfile
from    copy       import  deepcopy
from    sys        import  argv

seed = int(argv[2]) if (len(argv)>=3) else 0

random.seed(seed)
np.random.seed(seed)

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

_folder_            =  abspath(dirname(__file__))
lmap                =  lambda f,x: list(map(f,x))
lfilter             =  lambda f,x: list(filter(f,x))
listdir_full        =  lambda x: sorted([pjoin(x,y) for y in os_listdir(x)])
listdir_full_files  =  lambda x: lfilter(isfile, listdir_full(x))

def shuffled(A):
    A = np.array(A)
    np.random.shuffle(A)
    return A

def writer(fname,A):
    with open(fname, 'w') as f:
        for x in A:
            f.write(str(x) + '\n')

def writer_base(fname, D):
    for     irank  in  D:
        for key,A  in  D[irank].items():
            if type(A) != list: A = [str(A)]
            with open(fname(irank,key), 'w') as f:
                for x in A:
                    f.write(str(x) + '\n')

def writer_nd(A, key, start_shape=True):
    if (type(A) == dict): A  =  as_arr(A)
    buffer    =  []
    shape     =  list(np.array(A).shape)
    if start_shape: buffer.append( ' '.join(map(str,shape)))
    if   len(shape) == 1: buffer.extend([f"{i} {A[i]}"               for i in range(len(A))                                                         ])
    elif len(shape) == 2: buffer.extend([f"{i} {j} {A[i][j]}"        for i in range(len(A)) for j in range(len(A[i]))                               ])
    elif len(shape) == 3: buffer.extend([f"{i} {j} {k} {A[i][j][k]}" for i in range(len(A)) for j in range(len(A[i]))  for k in range(len(A[i][j])) ])
    buffer[0] += f' ! {key}'
    return buffer

def padded(A,pads,seed=None):
    if not (seed is None): np.random.seed(seed)
    assert (type(A) == dict) and (len(pads) ==6)
    A                       =  np.array(as_arr(A))
    i0, j0, k0, i1, j1, k1  =  pads
    sA                      =  np.array(A.shape)
    sB                      =  sA + np.array([i0+i1, j0+j1, k0+k1])
    B                       =  (np.random.rand(*sB).astype(A.dtype)-0.5)*200
    B[i0:sB[0]-i1,
      j0:sB[1]-j1,
      k0:sB[2]-k1]          =  A
    return Fort_Array(B)

# ------------------------------------------------------------------------
#                  Utilities
# ------------------------------------------------------------------------

Fort_Array  =  lambda A: {(i,j,k): A[i,j,k] for k in range(A.shape[2])
                                            for j in range(A.shape[1])
                                            for i in range(A.shape[0])}

def get_shape(p):
    shape  =  np.array(list(p.keys()))
    return shape.max(axis=0) - shape.min(axis=0) + 1

def as_arr(p):
    shape  =  get_shape(p)
    A      = (np.zeros(shape)*np.nan).tolist()
    for (i,j,k),val in p.items():
        A[i][j][k] = val
    return A

def mk_shards(p_full, pieces):
    fetch = lambda A,i0,i1,j0,j1,k0,k1: {(i-i0,j-j0,k-k0): A[i,j,k] for k in range(k0,k1)
                                                                    for j in range(j0,j1)
                                                                    for i in range(i0,i1)}
    result    =  {}
    i         =  0
    for         ix,nx in enumerate(pieces[0]):
        j     = 0
        for     iy,ny in enumerate(pieces[1]):
            k = 0
            for iz,nz in enumerate(pieces[2]):
                result[ix,iy,iz]  =  fetch(p_full, i,i+nx, j,j+ny, k,k+nz), [i,j,k], [i+nx-1, j+ny-1, k+nz-1]
                k                +=  nz
            j                    +=  ny
        i                        +=  nx
    return result

def transp_index(A,reorder_send):
    if type(A) == dict:
        D     =  {tuple([key[j] for j in reorder_send]): val for key,val in A.items()}
        keys  =  sorted(D.keys(), key = lambda x: x[::-1])
        return {k:D[k] for k in keys}
    else:
        return [A[j] for j in reorder_send]

def separate_grid(n,d):
    assert d <= n
    B = sorted(shuffled(list(range(n-1)))[:(d-1)]) + [n-1]
    A = [0] + (np.array(B[:-1])+1).tolist()
    pieces = (np.array(B) - np.array(A) + 1).tolist()
    assert (np.array(pieces) > 0).all() and (len(pieces) == d) and (sum(pieces) == n )
    return pieces

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

def get_flat(ranks):
    flat  =  [None for _ in ranks.reshape(-1)]
    for          i in range(ranks.shape[0]):
        for      j in range(ranks.shape[1]):
            for  k in range(ranks.shape[2]):
                flat[ranks[i,j,k]] = [i,j,k]
    return flat

def write_flat(ranks):
    flat  =  get_flat(ranks)
    return [f'{len(flat)} 3',
            *[f'{pos} {i} {j} {k}' for pos,(i,j,k) in enumerate(flat)]]

folder_index = [0]

def main_runner():
    divmax     =   3
    nloc_max   =   8
    nproc_max  =   8
    n_trials   =   int(argv[1])
    for made_trials  in  range(1,n_trials+1):
        use_alltoallv           =  random.randint(0,1)
        mode_api_cans           =  random.randint(0,1)
        mode_use_buf            =  random.randint(0,1) if (not mode_api_cans) else 1
        allow_autotune_reorder  =  random.randint(0,1)
        use_gpu                 =  random.randint(0,1)
        any_to_any              =  int(not use_alltoallv)
        while True:
            ii, jj, kk     =  shuffled(list(range(3)))
            nproc          =  float('inf')
            while nproc > nproc_max:
                div_xyz_ii      =  [random.randint(1,divmax)      for _ in range(3)]
                n_xyz           =  [random.randint(d,d*nloc_max)  for d in div_xyz_ii]
                div_xyz_ii[ii]  =  1
                nproc           =  np.prod(div_xyz_ii)

            if any_to_any:
                div_xyz_jj = [float('inf')]
                while np.prod(div_xyz_jj) != nproc:
                    div_xyz_jj      =  [random.randint(1,divmax)      for _ in range(3)]
                    div_xyz_jj[jj]  =  1
            else:
                div_xyz_jj                      =  deepcopy(div_xyz_ii)
                div_xyz_jj[ii], div_xyz_jj[jj]  =  div_xyz_jj[jj], div_xyz_jj[ii]

            if ((np.array(n_xyz)<np.array(div_xyz_ii)).any() or
                (np.array(n_xyz)<np.array(div_xyz_jj)).any()):
                continue
            else:
                break

        loc_order_ii , loc_order_jj, abs_reorder  =  [shuffled(list(range(3))).tolist()  for _  in range(3)]
        if random.randint(0,1): abs_reorder = [-1,-1,-1]

        pieces_xyz_ii, pieces_xyz_jj  =  [[separate_grid(n,d)  for n,d  in zip(n_xyz, divs)] for divs in [div_xyz_ii, div_xyz_jj]]

        p_full                        =  Fort_Array(20*(np.random.rand(*n_xyz)-0.5))
        p_xyz_ii, p_xyz_jj            =  [{key: lmap(lambda x: transp_index(x,loc_order),vals) for key,vals in mk_shards(p_full, pieces).items()}
                                           for loc_order,pieces in [[loc_order_ii, pieces_xyz_ii],
                                                                    [loc_order_jj, pieces_xyz_jj]]]

        ranks_ii                      =  np.array(shuffled(list(range(nproc)))).reshape(*div_xyz_ii)
        if use_alltoallv and (not any_to_any):
            temp                =  [0,1,2]
            temp[ii], temp[jj]  =  temp[jj], temp[ii]
            ranks_jj            =  ranks_ii.copy().transpose(*temp)
        else:
            ranks_jj            =  np.array(shuffled(list(range(nproc)))).reshape(*div_xyz_jj)

        assert  nproc  ==  np.prod(ranks_ii.shape, dtype=int)  ==  np.prod(ranks_jj.shape, dtype=int)

        trial_folder      =  pjoin(_folder_, 'trials', f'test_{folder_index[0]:06d}')
        folder_index[0]  +=  1
        input_folder      =  pjoin(trial_folder, 'input_transp' )
        output_folder     =  pjoin(trial_folder, 'output_transp')
        os_system(f"mkdir -p {input_folder}")
        os_system(f"mkdir -p {output_folder}")

        padstr        =  lambda s,n: '_'*(n-len(s)) + s
        fname         =  lambda irank,key: pjoin(input_folder, f'input_{irank:06d}{padstr(key,12)}.dat')
        buffer        =  {}
        for key,(px,lo_a,hi_a) in p_xyz_ii.items():
            irank            =  ranks_ii[key]
            (py,lo_b,hi_b),  =  [val for key,val in p_xyz_jj.items() if (ranks_jj[key]==irank)]
            loc_order_a      =  loc_order_ii
            loc_order_b      =  loc_order_jj
            (pads_x      , pads_y,
             pads_x_noise, pads_y_noise)  =  [[random.randint(0,4) for _ in range(6)] for _ in range(4)]
            def fix_pad():
                # pads_? is [h h h h+p h+p h+p]
                # initially is h h h p p p
                for j in range(3,6):
                    pads_x[j] += pads_x[j-3]
                    pads_y[j] += pads_y[j-3]
            fix_pad()
            autotune_opts                         =  lambda: [random.randint(0,1), random.randint(1,2), random.randint(1,6), random.randint(1,6)]
            force_send_autotune, send_autotuned, send_mode_op_simul, send_mode_op_batched = autotune_opts()
            force_recv_autotune, recv_autotuned, recv_mode_op_simul, recv_mode_op_batched = autotune_opts()

            all_modeops     =  [f"{force_send_autotune} {force_recv_autotune} ! force_send_autotune force_recv_autotune",
                                f'{send_autotuned} {recv_autotuned} ! send_autotuned recv_autotuned',
                                f"{send_mode_op_simul} {recv_mode_op_simul} ! send_mode_op_simul recv_mode_op_simul",
                                f"{send_mode_op_batched} {recv_mode_op_batched} ! send_mode_op_batched recv_mode_op_batched"]
            buffer[irank]   =  dict(info  =  [f"{ii} {jj} {kk} {use_alltoallv} {mode_api_cans} {allow_autotune_reorder} {mode_use_buf} ! ii jj kk use_alltoallv mode_api_cans allow_autotune_reorder mode_use_buf"               ,
                                              ' '.join(map(str,loc_order_a))   + ' ! loc_order_a' ,
                                              ' '.join(map(str,loc_order_b))   + ' ! loc_order_b' ,
                                              ' '.join(map(str,abs_reorder))   + ' ! abs_reorder' ,
                                              ' '.join(map(str,lo_a))          + ' ! lo_a'        ,
                                              ' '.join(map(str,lo_b))          + ' ! lo_b'        ,
                                              ' '.join(map(str,hi_a))          + ' ! hi_a'        ,
                                              ' '.join(map(str,hi_b))          + ' ! hi_b'        ,
                                              ' '.join(map(str, pads_x    ))   + ' ! pads_x'      ,
                                              ' '.join(map(str, pads_y    ))   + ' ! pads_y'      ,
                                              ' '.join(map(str, pads_x_noise)) + ' ! pads_x_noise',
                                              ' '.join(map(str, pads_y_noise)) + ' ! pads_y_noise',
                                              *writer_nd(padded(px,pads_x), 'px_padded')       ,
                                              *writer_nd(padded(py,pads_y), 'py_padded')       ,
                                              *all_modeops                                     ])
        print('Writing trial folder: ', trial_folder)
        writer_base(fname, buffer)
        if use_gpu:
            writer(pjoin(trial_folder,'use_gpu.txt'), ['1'])

if __name__ == '__main__':
    os_system(f"rm -rf {pjoin(_folder_, 'trials')}")
    os_system(f"cd {_folder_};rm -f *.mod;rm -f error_*.dat;rm -f main_transp_cpu;rm -f main_transp_gpu;rm -f core")
    main_runner()