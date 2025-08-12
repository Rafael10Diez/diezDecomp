
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
#                  Custom Libraries
# ------------------------------------------------------------------------

try:
    _folder_
except:
    _folder_  =  dirname(abspath(__file__))

# ------------------------------------------------------------------------
#                  Tranposes Module Functions
# ------------------------------------------------------------------------

lmap                =  lambda f,x: list(map(f,x))
lfilter             =  lambda f,x: list(filter(f,x))
listdir_full        =  lambda x: sorted([pjoin(x,y) for y in os_listdir(x)])
listdir_full_files  =  lambda x: lfilter(isfile, listdir_full(x))

def shuffled(A):
    A = np.array(A)
    np.random.shuffle(A)
    return A

def reader(fname):
    with open(fname, 'r') as f:
        return [x.rstrip('\n') for x in f]

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

def clean_borders(A,nh_xyz,ii,is_per,abs_order):
    if is_per:
        nh_xyz =  [(val if (i==ii) else 0) for i,val in enumerate(nh_xyz)]
        i,j,k  =  [nh_xyz[c] for c in abs_order]
        A      =  np.array(as_arr(A))
        if i:
            A[  :i,:,:] = (np.random.rand(*A[  :i,:,:].shape)-0.5)*10
            A[-i: ,:,:] = (np.random.rand(*A[-i: ,:,:].shape)-0.5)*10
        if j:
            A[:,  :j,:] = (np.random.rand(*A[:,  :j,:].shape)-0.5)*10
            A[:,-j: ,:] = (np.random.rand(*A[:,-j: ,:].shape)-0.5)*10
        if k:
            A[:,:,  :k] = (np.random.rand(*A[:,:,  :k].shape)-0.5)*10
            A[:,:,-k: ] = (np.random.rand(*A[:,:,-k: ].shape)-0.5)*10
        A = Fort_Array(A)
    return A

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

Fort_Array  =  lambda A: {(i,j,k): A[i,j,k] for k in range(A.shape[2])
                                            for j in range(A.shape[1])
                                            for i in range(A.shape[0])}

def get_shape(p):
    shape  =  np.array(list(p.keys()))
    return shape.max(axis=0) - shape.min(axis=0) + 1

def transp_index(A,reorder_send):
    D     =  {tuple([key[j] for j in reorder_send]): val for key,val in A.items()}
    keys  =  sorted(D.keys(), key = lambda x: x[::-1])
    return {k:D[k] for k in keys}

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

def as_arr(A):
    B      =  np.zeros(get_shape(A)).astype(float)
    for (i,j,k),val in A.items():
        B[i,j,k] = val
    return B

def mk_rand_shards(nh_xyz, pieces_xyz):
    result    =  {}
    pieces    =  [[(val+2*n) for val in p] for p,n in zip(pieces_xyz, nh_xyz)]
    for         ix,nx in enumerate(pieces[0]):
        for     iy,ny in enumerate(pieces[1]):
            for iz,nz in enumerate(pieces[2]):
                result[ix,iy,iz]  =  20*(np.random.rand(nx,ny,nz)-0.5)
    return result

def quick_halo_propagate(all_p00, nh_xyz, is_per, ii):
    all_p0 =  deepcopy(all_p00)
    all_p  =  deepcopy(all_p00)
    px, py, pz  =  get_shape(all_p)
    per_fix     =  lambda i, p, is_per: (i%p) if is_per else max(0,min(i,p-1))
    for (ix,iy,iz), A in all_p.items():
        for d in [1,-1]:
            i2,j2,k2  =  ix,iy,iz
            if ii == 0: i2 = per_fix(ix + d, px, is_per)
            if ii == 1: j2 = per_fix(iy + d, py, is_per)
            if ii == 2: k2 = per_fix(iz + d, pz, is_per)
            if ((i2,j2,k2) == (ix,iy,iz)) and (not is_per): continue
            B      =  all_p0[i2,j2,k2]
            na,nb  =  nh_xyz[ii],2*nh_xyz[ii]
            if na>0:
                if (ii,d) == (0, 1): A[    -na:  ,:,:] = B[     na: nb,:,:]
                if (ii,d) == (0,-1): A[       :na,:,:] = B[    -nb:-na,:,:]
                if (ii,d) == (1, 1): A[:,  -na:  ,:  ] = B[:,   na: nb,:  ]
                if (ii,d) == (1,-1): A[:,     :na,:  ] = B[:,  -nb:-na,:  ]
                if (ii,d) == (2, 1): A[:,:,-na:      ] = B[:,:, na: nb    ]
                if (ii,d) == (2,-1): A[:,:,   :na    ] = B[:,:,-nb:-na    ]
    return all_p

# ------------------------------------------------------------------------
#                  Utilities
# ------------------------------------------------------------------------

def writer_nd(A,key, start_shape=True):
    if (type(A) == dict): A  =  as_arr(A)
    buffer    =  []
    shape     =  list(np.array(A).shape)
    if start_shape: buffer.append( ' '.join(map(str,shape)))
    if   len(shape) == 1: buffer.extend([f"{i} {A[i]}"               for i in range(len(A))                                                         ])
    elif len(shape) == 2: buffer.extend([f"{i} {j} {A[i][j]}"        for i in range(len(A)) for j in range(len(A[i]))                               ])
    elif len(shape) == 3: buffer.extend([f"{i} {j} {k} {A[i][j][k]}" for i in range(len(A)) for j in range(len(A[i]))  for k in range(len(A[i][j])) ])
    buffer[0] += f' ! {key}'
    return buffer

def mk_mpi_ranks_halo(div_xyz):
    divs       =  list(div_xyz)
    return shuffled(np.array(range(int(np.prod(divs))), dtype=int)).reshape(*divs)

# ------------------------------------------------------------------------
#                  Main Runner
# ------------------------------------------------------------------------
folder_index = [0]
def main_runner():
    divmax              =   3
    nloc_max            =   5
    nproc_max           =   8
    n_trials            =  int(argv[1])
    made_trials         =  0
    max_nh_xyz          =  [float('-inf') for _ in range(3)]
    min_nh_xyz          =  [float('inf')  for _ in range(3)]

    while made_trials < n_trials:
        ii                 =  random.randint(0,2)
        axis               =  random.randint(1,3)
        use_halo_sync      =  random.randint(1,2) if random.randint(0,1) else -1
        pack_type          =  random.randint(2,3) if random.randint(0,1) else -1
        mode_api_cans      =  random.randint(0,1)
        use_gpu            =  random.randint(0,1)
        is_per             =  bool(random.randint(0,1))
        div_xyz            =   [random.randint(1,divmax)   for _ in range(3)]
        pieces_xyz         =  [[random.randint(1,nloc_max) for _ in range(d)] for d in div_xyz]

        nproc                      =  np.prod(div_xyz, dtype=int)
        if nproc > nproc_max: continue

        abs_order     =  shuffled(list(range(3)))
        nh_xyz        =  [random.randint(0,min(p)) for p in pieces_xyz]
        max_nh_xyz    =  [max(n,m) for n,m in zip(nh_xyz,max_nh_xyz)]
        min_nh_xyz    =  [min(n,m) for n,m in zip(nh_xyz,min_nh_xyz)]
        
        def get_ref_incomplete():
            all_p_xyz_incomplete  =  mk_rand_shards(nh_xyz, pieces_xyz)
            all_px_shape          =  {key: [arr[k] for arr in [np.array(val.shape)-2*np.array(nh_xyz)] for k in abs_order]
                                      for key,val in all_p_xyz_incomplete.items()}

            all_p_xyz_ref         =  quick_halo_propagate(all_p_xyz_incomplete, nh_xyz, is_per, ii)
            
            quick_transp          =  lambda D: {key: transp_index(Fort_Array(val),abs_order) for key,val in D.items()}
            all_p_xyz_ref         =  quick_transp(all_p_xyz_ref)
            all_p_xyz_incomplete  =  quick_transp(all_p_xyz_incomplete)

            return all_px_shape, all_p_xyz_ref, all_p_xyz_incomplete
        
        all_px_shape, all_p_xyz_ref, all_p_xyz_incomplete  =  get_ref_incomplete()
        mpi_ranks_ii  =  mk_mpi_ranks_halo(div_xyz)
        saved_fields  =  {}
        for key in all_p_xyz_incomplete:
            irank              =  mpi_ranks_ii[key]
            px                 =  deepcopy(all_p_xyz_incomplete[key])
            px_ref             =  deepcopy(all_p_xyz_ref[key])
            px_shape           =  all_px_shape[key]
            saved_fields[key]  =  dict(px = deepcopy(px), px_ref = deepcopy(px_ref), irank = irank, px_shape = px_shape)
        made_trials   +=  1
        trial_folder   =  pjoin(_folder_, 'trials', f'test_{folder_index[0]:06d}')
        folder_index[0] += 1
        input_folder   =  pjoin(trial_folder, 'input_halo' )
        output_folder  =  pjoin(trial_folder, 'output_halo')
        os_system(f"mkdir -p {input_folder}")
        os_system(f"mkdir -p {output_folder}")
        def mega_writer():
            padstr  =  lambda s,n: '_'*(n-len(s)) + s
            fname   =  lambda irank,key: pjoin(input_folder, f'input_{irank:06d}{padstr(key,12)}.dat')
            buffer  =  {}
            for D in saved_fields.values():
                pads_x, pads_x_noise     =  [[random.randint(0,4) for _ in range(6)] for _ in range(2)]
                buffer[D['irank']]       = dict(info=[f"{ii} {axis} {int(use_halo_sync)} {int(pack_type)} {int(mode_api_cans)} ! ii axis use_halo_sync pack_type mode_api_cans"    ,
                                                      ' '.join(map(str,abs_order)) + ' ! abs_order'                                                     ,
                                                      ' '.join(map(str,nh_xyz))    + ' ! nh_xyz'                                                        ,
                                                      ' '.join(map(str,np.array(np.where(mpi_ranks_ii==D['irank'])).reshape(-1).tolist())) + ' ! lo_xyz',
                                                      str(int(is_per))                                                                     + ' ! is_per',
                                                      ' '.join(map(str, pads_x       )) + ' ! pads_x',
                                                      ' '.join(map(str, pads_x_noise )) + ' ! pads_x_noise',
                                                      *writer_nd(padded(clean_borders(D['px'], nh_xyz,ii,is_per,abs_order),pads_x,seed=seed), 'px_padded'    )                                                     ,
                                                      *writer_nd(padded(D['px_ref'],pads_x,seed=seed), 'px_padded_ref')                                                     ])
            print('Writing trial folder: ', trial_folder)
            writer_base(fname, buffer)
            np.random.seed(random.randint(0,10))
        mega_writer()
        if use_gpu:
            writer(pjoin(trial_folder,'use_gpu.txt'), ['1'])

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    os_system(f"rm -rf {pjoin(_folder_, 'trials')}")
    os_system(f"cd {_folder_};rm -f *.mod;rm -f error_*.dat;rm -f main_halo_cpu;rm -f main_halo_gpu;rm -f core")
    main_runner()
