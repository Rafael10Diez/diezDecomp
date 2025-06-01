
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

seed  =  int(argv[2]) if (len(argv)>=3) else 0

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
tolist              =  lambda x: x.tolist() if hasattr(x,'tolist') else x

def writer(fname,A):
    with open(fname, 'w') as f:
        for x in A:
            f.write(str(x) + '\n')

def prime_decomposition(val):
    x      = val
    result = [1,1,1]
    for i in range(2,x+1):
        while ((x%i) == 0) and x>1:
            result.append(i)
            x //= i
        if x==1: break
    return result

def prod(A):
    x = A[0] + 0
    for i in range(1,len(A)):
        x *= A[i]
    return x

def rand_picks(A,n):
    L      = len(A)
    result = []
    for i in range(n):
        if i==(n-1):
            p = L
        else:
            p = random.randint(1,L-(n-1-i))
        result.append([])
        for _ in range(p):
            result[-1].append(A[L-1])
            L -= 1
    assert (len(A) == sum(map(len,result))) and all(len(p)>0 for p in result) and (L == 0)
    return result

# def rand_picks(A,n):
#     assert len(A) >= n
#     B       =  list(range(len(A)))
#     random.shuffle(B)
#     B       =  [-1,*sorted(B[:n])]
#     B[-1]   =  len(A)-1
#     result  =  [A[(B[i]+1):(B[i+1]+1)] for i in range(n)]
#     assert (len(A) == sum(map(len,result))) and all(len(p)>0 for p in result)
#     return result

def rand_partitions(nproc):
    A      =  prime_decomposition(nproc)
    random.shuffle(A)
    p_xzy =  lmap(prod,rand_picks(A,3))
    assert prod(p_xzy) == prod(A) == nproc
    random.shuffle(p_xzy)
    return p_xzy

def rand_splitter(p,n):
    A = list(range(n))
    result = lmap(len,rand_picks(A,p))
    random.shuffle(result)
    return result

def split_array(A, all_n):
    result = {}
    i,j,k  = 0,0,0
    for          p0,n0 in  enumerate(all_n[0]):
        i += n0
        j  = 0
        k  = 0
        for      p1,n1 in  enumerate(all_n[1]):
            j += n1
            k  = 0
            for  p2,n2 in  enumerate(all_n[2]):
                k += n2
                result[p0,p1,p2] = A[i-n0:i,j-n1:j,k-n2:k]
    return result

def shuffled(A):
    assert type(A) == list
    random.shuffle(A)
    return A

def get_flat_mpi_ranks(mpi_ranks):
    result = [None for _ in range(prod(mpi_ranks.shape))]
    for         i in range(mpi_ranks.shape[0]):
        for     j in range(mpi_ranks.shape[1]):
            for k in range(mpi_ranks.shape[2]):
                result[mpi_ranks[i,j,k]] = [i,j,k]
    return result

# ------------------------------------------------------------------------
#                  Utilities
# ------------------------------------------------------------------------

def main_runner():
    nproc_max            =   8
    n_trials             =  int(argv[1]) if len(argv)>=2 else 1
    for i_trial  in  range(1,n_trials+1):
        nproc            =  random.randint(1,nproc_max)
        ii               =  random.randint(0,2)
        order_halo       =  shuffled(list(range(3)))
        is_per           =  random.randint(0,1)
        p_xyz            =  rand_partitions(nproc)
        n_global         =  [random.randint(p,5*p) for p in p_xyz]
        n_xyz_glob       =  [rand_splitter(p,n) for p,n in zip(p_xyz, n_global)]
        nh_xyz           =  [random.randint(0,min(3,min(all_n))) for all_n in n_xyz_glob]
        
        # print('WARNING: order_halo set to [0,1,2]') ; order_halo = [0,1,2]
        # print('WARNING: nh_xyz     set to [0,0,0]') ; nh_xyz     = [0,0,0] ; nh_xyz[ii] = 0 # [min(1,nh_xyz[0]),min(1,nh_xyz[1]),min(1,nh_xyz[2])]
        # print('WARNING: is_per     set to 0') ; is_per     = 0

        n_xyz_glob       =  [(np.array(A)+2*nh_xyz[i]).tolist() for i,A in enumerate(n_xyz_glob)]
        mpi_ranks        =  list(range(nproc))
        random.shuffle(mpi_ranks)
        mpi_ranks        =  np.array(mpi_ranks ).reshape(*p_xyz)
        flat_mpi_ranks   =  get_flat_mpi_ranks(mpi_ranks)

        def fmt(A):
            A = tolist(A)
            if type(A) in [int,float,str]: return A
            elif type(A) in [list,tuple] : return ' '.join(map(str,A))
            else                         : 1/0
        
        reorder  =  lambda A: [A[i] for i in order_halo ]

        A_write  =  [f"{fmt(nproc)             } ! number of processes"              ,
                     f"{fmt(ii)                } ! halo axis"                        ,
                     f"{fmt(order_halo)        } ! order of local array (xyz -> 012)",
                     f"{fmt(nh_xyz)            } ! halo padding"                     ,
                     f"{fmt(is_per)            } ! periodicity"                      ]
        
        for irank in range(nproc):
            n3       =  [n_xyz_glob [i][j] for i,j in enumerate(flat_mpi_ranks [irank])]
            offset6  =  np.array([random.randint(0,3) for _ in range(6)]).reshape(3,2)
            A_write.append(f"{fmt(flat_mpi_ranks[irank])} ! flat_mpi_ranks ({irank:3d},0:2)")
            A_write.append(f"{fmt(reorder(n3          ))} ! all_n3         ({irank:3d},0:2)")
            A_write.append(f"{fmt(reorder(offset6[:,0]))} ! all_offset6    ({irank:3d},0:2,0)")
            A_write.append(f"{fmt(reorder(offset6[:,1]))} ! all_offset6    ({irank:3d},0:2,1)")
        
        writer(pjoin(_folder_,'trials',f'trial_halo_randomized_{i_trial:04}.dat'),A_write)

if __name__ == '__main__':
    os_system(f"cd {_folder_};rm -f *.mod;rm -f error_*.dat;rm -f example_1_halo_cpu;rm -f example_1_halo_gpu;rm -f core")
    os_system(f"rm   -rf {pjoin(_folder_, 'trials')}")
    os_system(f"mkdir -p {pjoin(_folder_, 'trials')}")
    main_runner()
