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
  integer                            ::  ii, irank, nproc, mpi_ierr, ierr, &
                                         nh_xyz(0:2), order_halo(0:2), offset6(0:2,0:1)
  real(rp)                           ::  error_max
  logical                            ::  periodic_xyz(0:2)
  character(len=30)                  ::  fname_30
  real(rp) , allocatable             ::  A(:,:,:), A_ref(:,:,:), work(:)
  integer  , allocatable             ::  flat_mpi_ranks(:,:), all_n3(:,:), all_offset6(:,:,:)
  integer(i8)                        ::  wsize
  type(diezdecomp_props_halo)        ::  hl
  type(diezdecomp_parsed_mpi_ranks)  ::  obj_ranks
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

  ! ------------------------- Begin: Read Input Data -------------------------
  block
    integer            ::  nproc_check, read_int0
    character(len=39)  ::  fname_39
    ! all processes will receive this information, but will only use what they need
    allocate(flat_mpi_ranks(0:nproc-1,0:2),&
                     all_n3(0:nproc-1,0:2),&
                all_offset6(0:nproc-1,0:2,0:1))
    if (irank == 0) then
      call get_command_argument(1, fname_30)
      write(fname_39, '(A9,A30)') './trials/', fname_30
      open(54,file=fname_39)
        read(54,*)  nproc_check
        if (nproc.ne.nproc_check) error stop '(nproc.ne.nproc_check)'
        read(54,*)  ii         ! x/y/z in absolute coordinates
        read(54,*)  order_halo ! custom array order
        read(54,*)  nh_xyz     ! in absolute coordinates
        read(54,*)  read_int0  ! periodic_xyz(ii)
        block
          integer :: i
          do i=0,nproc-1
            read(54,*) flat_mpi_ranks(i,0:2)
            read(54,*)         all_n3(i,0:2)   ! follows order_halo
            read(54,*)    all_offset6(i,0:2,0)
            read(54,*)    all_offset6(i,0:2,1)
          end do
        end block
      close(54)
    end if
    call MPI_BARRIER(mpi_comm_world, mpi_ierr)
    call MPI_BCAST(ii            , 1        , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(order_halo    , 3        , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(nh_xyz        , 3        , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(read_int0     , 1        , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(flat_mpi_ranks, 3*nproc  , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(all_n3        , 3*nproc  , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(all_offset6   , 3*nproc*2, MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BARRIER(mpi_comm_world, mpi_ierr)
    periodic_xyz      =  .false.
    periodic_xyz(ii)  =  (read_int0 == 1)
  end block
  ! -------------------- End: read data --------------------

  ! local padding (offset6)
  offset6  =   all_offset6(irank,:,:)

  ! allocate 3D arrays
  block
    integer :: sx_0, sx_1, sx_2
    ! shape local array [sx_0, sx_1, sx_2]
    sx_0 = all_n3(irank,0) + sum(offset6(0,:))
    sx_1 = all_n3(irank,1) + sum(offset6(1,:))
    sx_2 = all_n3(irank,2) + sum(offset6(2,:))
    allocate(A_ref(0:sx_0-1,0:sx_1-1,0:sx_2-1))
    allocate(A, mold=A_ref)
    A     = -1
    A_ref = -1
  end block

  ! ------------------------- Begin: Verification Array Setup -------------------------
  block
    integer :: lo_loc(0:2), lo_xyz(0:2), n_global(0:2), inv_order_halo(0:2)

    ! reversed order (useful to have)
    block
      integer :: i,j
      do   i=0,2
        do j=0,2
          if (order_halo(j)==i)  inv_order_halo(i)=j  ! formula tested in python
        end do
      end do
    end block

    ! compute n_global (global grid size, without halo cells)
    block
      integer :: i
      n_global(0) = 0
      n_global(1) = 0
      n_global(2) = 0
      do i=0,nproc-1
        if ((flat_mpi_ranks(i,1)==0).and.((flat_mpi_ranks(i,2)==0))) n_global(0)=n_global(0)+all_n3(i,inv_order_halo(0))-2*nh_xyz(0)
        if ((flat_mpi_ranks(i,0)==0).and.((flat_mpi_ranks(i,2)==0))) n_global(1)=n_global(1)+all_n3(i,inv_order_halo(1))-2*nh_xyz(1)
        if ((flat_mpi_ranks(i,0)==0).and.((flat_mpi_ranks(i,1)==0))) n_global(2)=n_global(2)+all_n3(i,inv_order_halo(2))-2*nh_xyz(2)
      end do
    end block

    ! compute lower bounds (lo_xyz) for each MPI task (in n_global)
    block
      integer :: i,j,p0,p1
      lo_xyz = [0,0,0]
      do   j=0,2
        if (j==0) p0=1
        if (j==1) p0=0
        if (j==2) p0=0
        p1  =  3-j-p0
        do i=0,nproc-1
          if ((flat_mpi_ranks(i, j)  < flat_mpi_ranks(irank, j)).and. &
              (flat_mpi_ranks(i,p0) == flat_mpi_ranks(irank,p0)).and. &
              (flat_mpi_ranks(i,p1) == flat_mpi_ranks(irank,p1))) &
           lo_xyz(j) = lo_xyz(j)+all_n3(i,inv_order_halo(j))-2*nh_xyz(j)
        end do
      end do
    end block

    ! lower bounds in local order (lo_loc)
    lo_loc(0) = lo_xyz(order_halo(0))
    lo_loc(1) = lo_xyz(order_halo(1))
    lo_loc(2) = lo_xyz(order_halo(2))

    ! define reference values (A_ref)
    block
      integer :: i,j,k, ijk_loc(0:2), ijk_0(0:2), axis, jj, kk, sz_jj, sz_kk
      logical :: valid

      ! define secondary dimensions (jj,kk) and their expected extent
      if (ii==0) jj = 1
      if (ii==1) jj = 0
      if (ii==2) jj = 0
      kk  =  3 - ii - jj
      sz_jj = all_n3(irank,inv_order_halo(jj)) - 2*nh_xyz(jj) + lo_xyz(jj)
      sz_kk = all_n3(irank,inv_order_halo(kk)) - 2*nh_xyz(kk) + lo_xyz(kk)

      ! define A_ref
      do     i=0, all_n3(irank,0)-1
        do   j=0, all_n3(irank,1)-1
          do k=0, all_n3(irank,2)-1
            ijk_loc  =  [lo_loc(0) + i , &
                         lo_loc(1) + j , &
                         lo_loc(2) + k ]
            ijk_0    =  [ijk_loc(inv_order_halo(0)) - nh_xyz(0) , &
                         ijk_loc(inv_order_halo(1)) - nh_xyz(1) , &
                         ijk_loc(inv_order_halo(2)) - nh_xyz(2) ]
            valid    =  .true.
            if (periodic_xyz(ii)) ijk_0(ii)  =  modulo(ijk_0(ii), n_global(ii))
            valid    =  (ijk_0(ii)>=0).and.(ijk_0(ii)<n_global(ii))
            valid    =  valid.and.(ijk_0(jj)>=lo_xyz(jj)).and.(ijk_0(jj)<sz_jj)
            valid    =  valid.and.(ijk_0(kk)>=lo_xyz(kk)).and.(ijk_0(kk)<sz_kk)
            if (valid) then
               A_ref(i+offset6(0,0), &
                      j+offset6(1,0), &
                      k+offset6(2,0))  =  ijk_0(0)*n_global(1)*n_global(2) + ijk_0(1)*n_global(2) + ijk_0(2) + 1
            end if
          end do
        end do
      end do
    end block
  end block
  ! ------------------------- End: Verification Array Setup -------------------------

  ! Copy inner part of verification array (without halo cells or offset6) to "A"
  block
    integer :: i0,i1,j0,j1,k0,k1
    i0  =               offset6(0,0) + nh_xyz(order_halo(0))
    j0  =               offset6(1,0) + nh_xyz(order_halo(1))
    k0  =               offset6(2,0) + nh_xyz(order_halo(2))
    i1  =  size(A,1) - offset6(0,1) - nh_xyz(order_halo(0)) - 1
    j1  =  size(A,2) - offset6(1,1) - nh_xyz(order_halo(1)) - 1
    k1  =  size(A,3) - offset6(2,1) - nh_xyz(order_halo(2)) - 1
    A(i0:i1,j0:j1,k0:k1) = A_ref(i0:i1,j0:j1,k0:k1)
  end block

  ! initialize diezDecomp
  block
    integer  ::  A_shape(0:2), lo_ref(0:2), use_halo_sync, autotuned_pack
    A_shape        =  [size(A,1), size(A,2), size(A,3)]
    use_halo_sync  = 0
    autotuned_pack = 0
    ! lo_ref: reference [i,j,k] coordinates to track the grid layout for the MPI tasks.
    !     - please note that "lo_xyz" could be used instead of "lo_ref = flat_mpi_ranks(irank,:)"
    !     - both alternatives accomplish the same purpose
    lo_ref         = flat_mpi_ranks(irank,:)
    call diezdecomp_track_mpi_decomp(lo_ref, obj_ranks, irank, nproc)
    call diezdecomp_generic_fill_hl_obj(hl, obj_ranks, A_shape, offset6, ii, nh_xyz, order_halo, periodic_xyz, wsize,&
                                        use_halo_sync, autotuned_pack)
  end block

  ! allocate work buffer
  allocate(work(0:wsize-1))
  work = -1

  !$acc wait
  !$acc enter data copyin(A, work)
  !$acc wait
  call MPI_BARRIER(mpi_comm_world, mpi_ierr)

  ! execute halo exchange
  call diezdecomp_halos_execute_generic(hl, A, work)
  !$acc wait
  !$acc exit data copyout(A)
  !$acc exit data delete(work)
  !$acc wait

  call MPI_BARRIER(mpi_comm_world, mpi_ierr)

  error_max = maxval(abs(A - A_ref))

  block
    real(dp) :: temp_real
    temp_real = error_max
    call MPI_Allreduce(temp_real, error_max, 1, MPI_DOUBLE_PRECISION, mpi_sum, mpi_comm_world, mpi_ierr)
  end block

  if (error_max > 1e-10) then
    error stop 'ERROR: maxval(abs(A - A_ref)) > 1e-10'
  end if

  if (irank == 0) then
    write(6,'(A54,E12.4,A1,A30)') 'diezDecomp: example_1_halo.f90: SUCCESS!!! error_max: ', error_max,&
                                  ' ', fname_30 ; flush(6)
  end if

  call MPI_BARRIER(mpi_comm_world, mpi_ierr)
  call MPI_BARRIER(mpi_comm_world, mpi_ierr)
  call MPI_finalize(mpi_ierr)

end program
