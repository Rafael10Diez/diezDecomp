program main
  ! ------------------------- Imported Modules -------------------------
#if defined(_OPENACC)
  use openacc
#endif
#if defined(_USE_CUDA)
  use cudafor
#endif
  use mpi
  use, intrinsic :: iso_fortran_env, only: i8 => int64, sp => real32, dp => real64
  use diezdecomp_api_generic
#if defined(_OPENACC)
  use openacc, only: acc_handle_kind
#else
  use, intrinsic :: iso_fortran_env, only: acc_handle_kind => int64
#endif

  implicit none

  ! ------------------------- Variable Declarations -------------------------
#if defined(_DIEZDECOMP_SINGLE)
  integer, parameter :: rp = sp
#else
  integer, parameter :: rp = dp
#endif
  integer                           ::  ii, jj, irank, nproc, mpi_ierr, &
                                        nx_in, ny_in, nz_in, nx_out, ny_out, nz_out, &
                                        order_in(0:2), order_out(0:2), order_intermediate(0:2), &
                                        lo_in(0:2), lo_out(0:2), &
                                        offset6_in(0:2,0:1), offset6_out(0:2,0:1)
  integer(i8)                       ::  wsize
  real(rp)                          ::  error_max
  logical                           ::  allow_alltoallv, allow_autotune_reorder

  real(rp) , allocatable, target    ::  p_out_ref(:,:,:), work(:), p_in_out_buf(:)
  real(rp), pointer                 ::  p_in(:,:,:), p_out(:,:,:)

  type(diezdecomp_props_transp)     ::  tr
  type(diezdecomp_parsed_mpi_ranks) ::  obj_rank_in, obj_rank_out
#if defined(_OPENACC)
  integer(acc_device_kind)          ::  dev_type
#endif

  ! ------------------------- Begin: MPI Setup -------------------------
  call mpi_init     (mpi_ierr)
  call mpi_comm_rank(mpi_comm_world, irank, mpi_ierr)
  call mpi_comm_size(mpi_comm_world, nproc, mpi_ierr)
#if defined(_OPENACC)
  block
    integer           ::  ierr, ndev, mydev, i
    character(len=1)  ::  is_acc
    dev_type  =  acc_get_device_type()
#if defined(_USE_CUDA)
    mpi_ierr  =  cudaGetDeviceCount(ndev)
    mydev     =  mod(irank,ndev)
    ierr      =  ierr + cudaSetDevice(mydev)
#else
    ndev      =  8
    mydev     =  mod(irank,ndev)
#endif
    call acc_set_device_num(mydev,dev_type)
    call acc_init(dev_type)
#if defined(_OPENACC)
    is_acc = 'T'
#else
    is_acc = 'F'
#endif
    do i=0,nproc-1
     call MPI_BARRIER(mpi_comm_world, mpi_ierr)
     if (i==irank) then
       write(6,'(A33,4I6,A12,A1)') '    (irank, nproc, mydev, ndev): ',irank,nproc,mydev,ndev,' , OpenACC: ', is_acc;flush(6)
     end if
     call MPI_BARRIER(mpi_comm_world, mpi_ierr)
    end do
  end block
#endif
  ! ------------------------- End: MPI Setup -------------------------

  ! ------------------------- Begin: Array Setup -------------------------
  block
    integer :: pxy, pz, nx_global, ny_global, nz_global

    ! MPI grid layout is:
    ! start (x-aligned pencils): [1, 2, 4]
    ! end   (y-aligned pencils): [2, 1, 4]

    pxy        =    2
    pz         =    4

    nx_global  =  200
    ny_global  =  320
    nz_global  =  440

    nx_in      =  nx_global
    ny_in      =  ny_global/pxy
    nz_in      =  nz_global/pz

    nx_out     =  nx_global/pxy
    ny_out     =  ny_global
    nz_out     =  nz_global/pz

    ! reference coordinates in n_x/y/z_global
    block
      integer :: i_xy, i_z
      i_xy         =  irank/pz
      i_z          =  mod(irank,pz)
      if (i_xy >= pxy) error stop 'i_xy >= pxy'
      if (i_z  >= pz ) error stop 'i_z  >= pz'
      lo_in        =  [0         , i_xy*ny_in, i_z*nz_in ]
      lo_out       =  [i_xy*nx_out,         0, i_z*nz_out]
    end block

   allocate(p_out_ref(0:nx_out-1, 0:ny_out-1, 0:nz_out-1))
   allocate(p_in_out_buf(0:(nx_in*ny_in*nz_in+nx_out*ny_out*nz_out-1)))

    p_in (0:nx_in -1, 0:ny_in -1, 0:nz_in -1) => p_in_out_buf(0:(nx_in*ny_in*nz_in-1))
    p_out(0:nx_out-1, 0:ny_out-1, 0:nz_out-1) => p_in_out_buf(nx_in*ny_in*nz_in:(nx_in*ny_in*nz_in+nx_out*ny_out*nz_out-1))

    ! define reference input values (p_in)
    block
      integer :: i,j,k,i0,j0,k0
      do     i=0, nx_in-1
        do   j=0, ny_in-1
          do k=0, nz_in-1
            i0           =  lo_in(0) + i
            j0           =  lo_in(1) + j
            k0           =  lo_in(2) + k
            p_in(i,j,k)  =  k0*(ny_global*nx_global) + j0*nx_global + i0 + 1
          end do
        end do
      end do
    end block

    ! define reference output values (p_out_ref)
    block
      integer :: i,j,k,i0,j0,k0
      do     i=0, nx_out-1
        do   j=0, ny_out-1
          do k=0, nz_out-1
            i0                =  lo_out(0) + i
            j0                =  lo_out(1) + j
            k0                =  lo_out(2) + k
            p_out_ref(i,j,k)  =  k0*(ny_global*nx_global) + j0*nx_global + i0 + 1
          end do
        end do
      end do
    end block
    ! define empty output array: p_out
    p_out  =  -1
  end block
  ! ------------------------- End: Array Setup -------------------------

  ! ------------------------- Begin: Transpose Initialization -------------------------
  block
      integer :: sp_in(0:2), sp_out(0:2)
      sp_in  = [nx_in , ny_in , nz_in ]
      sp_out = [nx_out, ny_out, nz_out]

      ii                      =  -1 ! only used if allow_alltoallv
      jj                      =  -1 ! only used if allow_alltoallv
      order_in                =  [0,1,2]
      order_out               =  [0,1,2]
      order_intermediate      =  [0,1,2]
      offset6_in(:,:)         =  0
      offset6_out(:,:)        =  0
      allow_alltoallv         =  .false.
      allow_autotune_reorder  =  .false.

      call diezdecomp_track_mpi_decomp(lo_in , obj_rank_in , irank, nproc, order_in)
      call diezdecomp_track_mpi_decomp(lo_out, obj_rank_out, irank, nproc, order_out)
      call diezdecomp_generic_fill_tr_obj(tr, obj_rank_in, obj_rank_out, sp_in, offset6_in, sp_out, offset6_out, &
                                          order_in, order_out, order_intermediate, allow_alltoallv, ii, jj, wsize, &
                                          allow_autotune_reorder)
      allocate(work(0:(wsize-1)))
      work  =  -1
  end block
  ! ------------------------- End: Transpose Initialization -------------------------

  !$acc wait
  !$acc enter data copyin(p_in_out_buf, work)
  !$acc wait

  ! execute transpose
  call diezdecomp_transp_execute_generic_buf(tr, p_in, p_out, work)

  !$acc wait
  !$acc exit data copyout(p_in_out_buf, work)
  !$acc wait

  error_max = maxval(abs(p_out - p_out_ref))

  block
    real(dp) :: temp_real
    temp_real = error_max
    call MPI_Allreduce(temp_real, error_max, 1, MPI_DOUBLE_PRECISION, mpi_sum, mpi_comm_world, mpi_ierr)
  end block

  if (error_max > 1e-10) then
    error stop 'ERROR: maxval(abs(p_out - p_out_ref)) > 1e-10'
  end if

  if (irank == 0) then
    write(6,'(A67,E12.4)') 'diezDecomp: example_1_transp_xy_8.f90: SUCCESS!!! error_max:  ', error_max ; flush(6)
  end if

  call MPI_BARRIER(mpi_comm_world, mpi_ierr)
  call MPI_BARRIER(mpi_comm_world, mpi_ierr)
  call MPI_finalize(mpi_ierr)

end program


