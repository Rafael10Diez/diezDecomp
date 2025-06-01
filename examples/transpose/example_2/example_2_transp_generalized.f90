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
                                        inv_order_in(0:2), inv_order_out(0:2),&
                                        lo_in(0:2), lo_out(0:2), &
                                        offset6_in(0:2,0:1), offset6_out(0:2,0:1)
  integer(i8)                       ::  wsize
  real(rp)                          ::  error_max
  logical                           ::  allow_alltoallv, allow_autotune_reorder

  real(rp) , allocatable, target    ::  p_out_ref(:,:,:), work(:), p_in_out_buf(:)
  integer  , allocatable            ::  all_offset6_in(:,:,:), all_offset6_out(:,:,:), &
                                        all_n3_in(:,:), all_n3_out(:,:), &
                                        flat_mpi_ranks_in(:,:), flat_mpi_ranks_out(:,:)
  real(rp), pointer                 ::  p_in(:,:,:), p_out(:,:,:)
  character(len=36)                 ::  fname_36
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

  ! ------------------------- Begin: Read Input Data -------------------------
  block
    integer            ::  nproc_check
    character(len=45)  ::  fname_45

    ! all processes will receive this information, but will only use what they need
    allocate( flat_mpi_ranks_in(0:nproc-1,0:2),&
             flat_mpi_ranks_out(0:nproc-1,0:2),&
                      all_n3_in(0:nproc-1,0:2),&
                     all_n3_out(0:nproc-1,0:2),&
                 all_offset6_in(0:nproc-1,0:2,0:1),&
                all_offset6_out(0:nproc-1,0:2,0:1))
    if (irank == 0) then
      call get_command_argument(1, fname_36)
      write(fname_45, '(A9,A36)') './trials/', fname_36
      open(54,file=fname_45)
        read(54,*) nproc_check
        if (nproc.ne.nproc_check) error stop '(nproc.ne.nproc_check)'
        read(54,*) order_in
        read(54,*) order_out
        read(54,*) order_intermediate
        block
          integer :: i
          do i=0,nproc-1
            read(54,*) flat_mpi_ranks_in (i,0:2)
            read(54,*) flat_mpi_ranks_out(i,0:2)
            read(54,*) all_n3_in      (i,0:2)
            read(54,*) all_n3_out     (i,0:2)
            read(54,*) all_offset6_in    (i,0:2,0)
            read(54,*) all_offset6_in    (i,0:2,1)
            read(54,*) all_offset6_out   (i,0:2,0)
            read(54,*) all_offset6_out   (i,0:2,1)
          end do
        end block
      close(54)
    end if
    call MPI_BARRIER(mpi_comm_world, mpi_ierr)
    call MPI_BCAST(order_in          , 3        , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(order_out         , 3        , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(order_intermediate, 3        , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(flat_mpi_ranks_in , 3*nproc  , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(flat_mpi_ranks_out, 3*nproc  , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(all_n3_in      , 3*nproc  , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(all_n3_out     , 3*nproc  , MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(all_offset6_in    , 3*nproc*2, MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BCAST(all_offset6_out   , 3*nproc*2, MPI_INTEGER, 0, mpi_comm_world, mpi_ierr)
    call MPI_BARRIER(mpi_comm_world, mpi_ierr)
  end block
  ! ------------------------- End: Read Input Data -------------------------

  offset6_in   =   all_offset6_in(irank,:,:)
  offset6_out  =  all_offset6_out(irank,:,:)
  nx_in        =     all_n3_in(irank,0)
  nx_out       =    all_n3_out(irank,0)
  ny_in        =     all_n3_in(irank,1)
  ny_out       =    all_n3_out(irank,1)
  nz_in        =     all_n3_in(irank,2)
  nz_out       =    all_n3_out(irank,2)

  ! allocate arrays (p_in and p_out buffers)
  block
    integer :: n0_in, n1_in, n2_in, n0_out, n1_out, n2_out
    n0_in  = nx_in  + sum(offset6_in(0,:))
    n1_in  = ny_in  + sum(offset6_in(1,:))
    n2_in  = nz_in  + sum(offset6_in(2,:))
    n0_out = nx_out + sum(offset6_out(0,:))
    n1_out = ny_out + sum(offset6_out(1,:))
    n2_out = nz_out + sum(offset6_out(2,:))
    allocate(p_out_ref(0:n0_out-1, 0:n1_out-1, 0:n2_out-1))
    allocate(p_in_out_buf(0:(n0_in*n1_in*n2_in+n0_out*n1_out*n2_out-1)))
    p_out_ref    = -1
    p_in_out_buf = -1
    p_in (0:n0_in -1, 0:n1_in -1, 0:n2_in -1) => p_in_out_buf(0:(n0_in*n1_in*n2_in-1))
    p_out(0:n0_out-1, 0:n1_out-1, 0:n2_out-1) => p_in_out_buf(n0_in*n1_in*n2_in:(n0_in*n1_in*n2_in+n0_out*n1_out*n2_out-1))
    p_out = -1
  end block

  ! ------------------------- Begin: Verification Array Setup -------------------------
  block
    integer :: nx_global, ny_global, nz_global

    ! calculate inverse order
    block
      integer :: i,j
      do   i=0,2
        do j=0,2
          if (order_in(j)==i)  inv_order_in(i)=j  ! formula tested in python
          if (order_out(j)==i) inv_order_out(i)=j ! formula tested in python
        end do
      end do
    end block

    ! calculate global grid size
    block
      integer :: i
      nx_global = 0
      ny_global = 0
      nz_global = 0
      do i=0,nproc-1
        if ((flat_mpi_ranks_in(i,1)==0).and.((flat_mpi_ranks_in(i,2)==0))) nx_global=nx_global+all_n3_in(i,inv_order_in(0))
        if ((flat_mpi_ranks_in(i,0)==0).and.((flat_mpi_ranks_in(i,2)==0))) ny_global=ny_global+all_n3_in(i,inv_order_in(1))
        if ((flat_mpi_ranks_in(i,0)==0).and.((flat_mpi_ranks_in(i,1)==0))) nz_global=nz_global+all_n3_in(i,inv_order_in(2))
      end do
    end block
    block
      integer :: i,nx2,ny2,nz2
      nx2 = 0
      ny2 = 0
      nz2 = 0
      do i=0,nproc-1
        if ((flat_mpi_ranks_out(i,1)==0).and.((flat_mpi_ranks_out(i,2)==0))) nx2=nx2+all_n3_out(i,inv_order_out(0))
        if ((flat_mpi_ranks_out(i,0)==0).and.((flat_mpi_ranks_out(i,2)==0))) ny2=ny2+all_n3_out(i,inv_order_out(1))
        if ((flat_mpi_ranks_out(i,0)==0).and.((flat_mpi_ranks_out(i,1)==0))) nz2=nz2+all_n3_out(i,inv_order_out(2))
      end do
      if (nx_global.ne.nx2) error stop 'nx_global.ne.nx2'
      if (ny_global.ne.ny2) error stop 'ny_global.ne.ny2'
      if (nz_global.ne.nz2) error stop 'nz_global.ne.nz2'
    end block

    ! calculate lower bounds in global grid size (lo_in and lo_out)

    ! calculate lo_in
    associate(lo             => lo_in             ,&
              order          => order_in          ,&
              inv_order      => inv_order_in      ,&
              all_n_xzy      => all_n3_in      ,&
              flat_mpi_ranks => flat_mpi_ranks_in )
      block
        integer :: i,j,k,p0,p1
        lo = [0,0,0]
        do   j=0,2
          if (j==0) p0=1
          if (j==1) p0=0
          if (j==2) p0=0
          p1 = 3-j-p0
          k = inv_order(j)
          do i=0,nproc-1
            if ((flat_mpi_ranks(i,j)<flat_mpi_ranks(irank,j)).and. &
                (flat_mpi_ranks(i,p0)==flat_mpi_ranks(irank,p0)).and. &
                (flat_mpi_ranks(i,p1)==flat_mpi_ranks(irank,p1))) &
             lo(k) = lo(k)+all_n_xzy(i,k)
          end do
        end do
      end block
    end associate

    ! calculate lo_out
    associate(lo             => lo_out             ,&
              order          => order_out          ,&
              inv_order      => inv_order_out      ,&
              all_n_xzy      => all_n3_out      ,&
              flat_mpi_ranks => flat_mpi_ranks_out )
      block
        integer :: i,j,k,p0,p1
        lo = [0,0,0]
        do   j=0,2
          if (j==0) p0=1
          if (j==1) p0=0
          if (j==2) p0=0
          p1 = 3-j-p0
          k = inv_order(j)
          do i=0,nproc-1
            if ((flat_mpi_ranks(i,j)<flat_mpi_ranks(irank,j)).and. &
                (flat_mpi_ranks(i,p0)==flat_mpi_ranks(irank,p0)).and. &
                (flat_mpi_ranks(i,p1)==flat_mpi_ranks(irank,p1))) &
             lo(k) = lo(k)+all_n_xzy(i,k)
          end do
        end do
      end block
    end associate

    ! define reference input values (p_in)
    block
      integer :: i,j,k,ijk_loc(0:2),i0,j0,k0
      do     i=0, nx_in-1
        do   j=0, ny_in-1
          do k=0, nz_in-1
            ijk_loc =  [lo_in(0) + i , &
                        lo_in(1) + j , &
                        lo_in(2) + k ]
            i0      =  ijk_loc(inv_order_in(0))
            j0      =  ijk_loc(inv_order_in(1))
            k0      =  ijk_loc(inv_order_in(2))
            p_in(i+offset6_in(0,0),&
                 j+offset6_in(1,0),&
                 k+offset6_in(2,0))  =  i0*ny_global*nz_global + j0*nz_global + k0 + 1
          end do
        end do
      end do
    end block

    ! define reference output values (p_out_ref)
    block
      integer :: i,j,k,ijk_loc(0:2),i0,j0,k0
      do     i=0, nx_out-1
        do   j=0, ny_out-1
          do k=0, nz_out-1
            ijk_loc           =  [lo_out(0) + i , &
                                  lo_out(1) + j , &
                                  lo_out(2) + k ]
            i0                =  ijk_loc(inv_order_out(0))
            j0                =  ijk_loc(inv_order_out(1))
            k0                =  ijk_loc(inv_order_out(2))
            p_out_ref(i+offset6_out(0,0),&
                      j+offset6_out(1,0),&
                      k+offset6_out(2,0))  =  i0*ny_global*nz_global + j0*nz_global + k0 + 1
          end do
        end do
      end do
    end block
  end block
  ! ------------------------- End: Verification Array Setup -------------------------

  ! ------------------------- Begin: Transpose Initialization -------------------------
  block
      integer :: sp_in(0:2), sp_out(0:2), lo_ref_in(0:2), lo_ref_out(0:2)
      sp_in   =  [nx_in +sum(offset6_in (0,:)), ny_in +sum(offset6_in (1,:)), nz_in +sum(offset6_in (2,:))]
      sp_out  =  [nx_out+sum(offset6_out(0,:)), ny_out+sum(offset6_out(1,:)), nz_out+sum(offset6_out(2,:))]
      ii                      =  -1 ! only used if allow_alltoallv
      jj                      =  -1 ! only used if allow_alltoallv
      allow_alltoallv         =  .false.
      allow_autotune_reorder  =  .false.
      lo_ref_in               =  flat_mpi_ranks_in(irank,:)
      lo_ref_out              =  flat_mpi_ranks_out(irank,:)
      call diezdecomp_track_mpi_decomp(lo_ref_in , obj_rank_in , irank, nproc)
      call diezdecomp_track_mpi_decomp(lo_ref_out, obj_rank_out, irank, nproc)
      call diezdecomp_generic_fill_tr_obj(tr, obj_rank_in, obj_rank_out, sp_in, offset6_in, sp_out, offset6_out, &
                                          order_in, order_out, &
                                          order_intermediate, allow_alltoallv, ii, jj, wsize, &
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
    write(6,'(A68,E12.4,A1,A36)') 'diezDecomp: example_2_transp_generalized.f90: SUCCESS!!! error_max: ', error_max,&
                                  ' ', fname_36 ; flush(6)
  end if

  call MPI_BARRIER(mpi_comm_world, mpi_ierr)
  call MPI_BARRIER(mpi_comm_world, mpi_ierr)
  call MPI_finalize(mpi_ierr)

end program
