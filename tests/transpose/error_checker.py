
# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------

from    os         import  system                                     as  os_system
from    os.path    import  join                                       as  pjoin
from    os         import  listdir                                    as  os_listdir
from    os.path    import  abspath, dirname, basename, isfile, isdir
from    copy       import  deepcopy
from    sys        import  argv

# ------------------------------------------------------------------------
#                  Custom Libraries
# ------------------------------------------------------------------------

try:
    _folder_
except:
    _folder_  =  dirname(abspath(__file__))

# ------------------------------------------------------------------------
#                  Basic Functions
# ------------------------------------------------------------------------

lmap                  =  lambda f,x: list(map(f,x))
lfilter               =  lambda f,x: list(filter(f,x))
listdir_full          =  lambda x: sorted([pjoin(x,y) for y in os_listdir(x)])
listdir_full_files    =  lambda x: lfilter(isfile, listdir_full(x))
listdir_full_folders  =  lambda x: lfilter(isdir , listdir_full(x))

def reader(fname):
    with open(fname, 'r') as f:
        return [x.rstrip('\n') for x in f]

def zero_digits(x):
    for i in range(10):
        x = x.replace(str(i),'0')
    return x

def writer(fname, A):
    with open(fname, 'w') as f:
        for x in A:
            f.write(x + '\n')

# ------------------------------------------------------------------------
#                  Main Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    
    all_trials  =  listdir_full_folders(pjoin(_folder_,'trials'))
    assert all_trials

    print(f"Number of tests: {len(all_trials):8d}")
    final_error = 0
    for folder in all_trials:
        input_folder,   =  lfilter(lambda x: basename(x)[:6] == 'input_', listdir_full_folders(folder))
        input_files     =  listdir_full_files(input_folder)

        nproc           =  len(input_files)//len(set(lmap(zero_digits,input_files)))

        assert nproc == (max([int(basename(x).split('_')[1].lstrip('0') or '0') for x in input_files])+1)

        output_folder,  =  lfilter(lambda x: basename(x)[:7] == 'output_', listdir_full_folders(folder))
        error_files     =  listdir_full_files(output_folder)

        assert len(error_files) == nproc, folder
        final_error     =  max(final_error, max([abs(float(' '.join(reader(x)))) for x in error_files]))
        assert final_error < 1e-10, folder
    
    all_argv  =  ' '.join(argv[1:])
    message   =  f'Done! all tests passed (global_error: {final_error:.3f}) (argv: {all_argv}) ({_folder_})'
    print(message)
    writer(pjoin(_folder_, f'error_total_argv_{all_argv}.dat'), [message])