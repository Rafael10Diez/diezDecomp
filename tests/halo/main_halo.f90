program main
#if defined(_OPENACC)
  use openacc
#endif
#if defined(_USE_CUDA)
  use cudafor
#endif
  use mpi
  use diezdecomp_api_cans
  use diezdecomp_api_generic
#if defined(_OPENACC)
  use openacc, only: acc_handle_kind
#else
  use, intrinsic :: iso_fortran_env, only: acc_handle_kind => int64
#endif
  use, intrinsic     :: iso_fortran_env, only: i8 => int64, sp => real32, dp => real64
  implicit none
  integer, parameter :: rp = dp
  integer                            ::  ii,pos, i, j, k, iter, irank, nproc, mpi_ierr, ierr, &
                                         nh_xyz(0:2), order_halo(0:2), axis, offset6(0:2,0:1), &
                                         read_int0, read_int1, numel, lo_xyz(0:2), use_halo_sync, autotuned_pack
  real(rp)                           ::  error_max
  logical                            ::  is_per, periodic(0:2), mode_api_cans
  character(len=60)                  ::  fname_60
  character(len=11)                  ::  fname_11
  real(rp)                           ::  read_real, elapse_time
  real(rp) , allocatable             ::  px(:,:,:), px_ref(:,:,:), buffer(:)
  integer(i8)                        ::  wsize
  integer(acc_handle_kind)           ::  stream = 1
  type(diezdecomp_props_halo)       ::  hl
  type(diezDecompHandle)             ::  ch
  type(diezDecompGridDesc)           ::  gd
  type(diezdecomp_parsed_mpi_ranks)  ::  ox
#if defined(_OPENACC)
  integer(acc_device_kind)          ::  dev_type
#endif

  call mpi_init   (mpi_ierr)
  call mpi_comm_rank(mpi_comm_world, irank, mpi_ierr)
  call mpi_comm_size(mpi_comm_world, nproc, mpi_ierr)
#if defined(_OPENACC)
    block
        integer:: ierr, ndev, mydev,i
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
        do i=0,nproc-1
         call MPI_BARRIER(mpi_comm_world,mpi_ierr)
         if (i==irank) then
           write(6,'(A33,4I6)') '    (irank, nproc, mydev, ndev): ',irank,nproc,mydev,ndev;flush(6)
         end if
         call MPI_BARRIER(mpi_comm_world,mpi_ierr)
        end do
    end block
#endif

  ! -------------------- begin: read data --------------------
  do pos=0,nproc-1
    if (pos==irank) then
      !   get fname_11
      call get_command_argument(1, fname_11)

      !   input_??????__________ii.dat
      write(fname_60, '(A9,A11,A18,I0.6,A16)') './trials/', fname_11, '/input_halo/input_', irank, '________info.dat'
      open(54,file=fname_60)
        read(54,*) ii, axis, use_halo_sync, autotuned_pack,read_int1
        mode_api_cans = (read_int1==1)
        read(54,*)  order_halo
        read(54,*)  nh_xyz
        read(54,*)  lo_xyz
        read(54,*)  read_int0
        is_per = (read_int0==1)
        read(54,*)  offset6
        read(54,*) i,j,k
        allocate(px(0:i-1,0:j-1,0:k-1))
        numel = i*j*k
        do iter=1,numel
          read(54,*) i,j,k,read_real
          px(i,j,k) = read_real
        end do
        read(54,*) i,j,k
        allocate(px_ref(0:i-1,0:j-1,0:k-1))
        numel = i*j*k
        do iter=1,numel
          read(54,*) i,j,k,read_real
          px_ref(i,j,k) = read_real
        end do
      close(54)
    endif
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  enddo
  ! -------------------- end: read data --------------------
  write(6,'(I3,A15,6I3)') irank, ' halo offset6: ', offset6 ; flush(6)
  call diezdecomp_track_mpi_decomp(lo_xyz, ox, irank, nproc)
  periodic      =  .false.
  periodic(ii)  =  is_per
  if (mode_api_cans) then
    block
      integer :: nh_ijk(0:2)
      ! fill grid descriptor (by hand)
      gd%all_ap(axis-1)%internal_order(1:3)  =  order_halo(0:2)
      gd%all_ap(axis-1)%order_halo(1:3)      =  order_halo(0:2)

      do k=0,2
        nh_ijk(k) =  nh_xyz(order_halo(k))
      end do

      gd%all_ap(axis-1)%shape(1:3)           =  [size(px,1) - 2*nh_ijk(0), &
                                                 size(px,2) - 2*nh_ijk(1), &
                                                 size(px,3) - 2*nh_ijk(2)  ]
      gd%irank                               =  irank
      gd%nproc                               =  nproc

      allocate(gd%all_ap(axis-1)%mpi_ranks      , mold=ox%mpi_ranks      )
      allocate(gd%all_ap(axis-1)%flat_mpi_ranks , mold=ox%flat_mpi_ranks )

      gd%all_ap(axis-1)%mpi_ranks        =  ox%mpi_ranks
      gd%all_ap(axis-1)%flat_mpi_ranks   =  ox%flat_mpi_ranks
      gd%all_ap(axis-1)%shape_mpi_ranks  =  ox%shape_mpi_ranks


      ierr = diezDecomp_get_workspace_size_halos(ch,gd,axis,nh_xyz,wsize)
      ! write(6,*) irank,'diezDecomp_get_workspace_size_halos:', wsize, gd%all_ap(axis-1)%shape, size(px), nh_ijk ; flush(6)
      diezdecomp_halo_mpi_mode      = use_halo_sync
      diezdecomp_halo_autotuned_pack = autotuned_pack
    end block
  else
    block
      integer  ::  p_shape(0:2)
      p_shape       =  [size(px,1), size(px,2), size(px,3)]
      call diezdecomp_generic_fill_hl_obj(hl, ox, p_shape, offset6, ii, nh_xyz, order_halo, periodic, wsize,&
                                          use_halo_sync, autotuned_pack)
    end block
  end if
  write(6,*) 'irank, wsize: ', irank, wsize;flush(6)
  allocate(buffer(wsize))

  !$acc wait
  !$acc enter data copyin(px, buffer)
  !$acc wait
#if defined(_OPENACC)
    write(6,*) 'Openacc detected!' ; flush(6)
    px   = -200
#else
    write(6,*) 'Openacc not found!' ; flush(6)
#endif
  call MPI_BARRIER(mpi_comm_world,mpi_ierr)
  write(6,*) 'begin: halos_execute' ; flush(6)
  elapse_time = MPI_Wtime()
  if (mode_api_cans) then
    call diezDecomp_boilerplate_halos(gd, px, buffer, nh_xyz, offset6, periodic, ii+1, axis, stream, .true., .true.)
  else
    call diezdecomp_halos_execute_generic(hl, px, buffer, stream)
    call diezdecomp_summary_halo_autotuning(hl)
  end if
  elapse_time   = MPI_Wtime() - elapse_time
  write(6,*) 'end: halos_execute' ; flush(6)
  !$acc wait
  !$acc exit data copyout(px)
  !$acc exit data delete(buffer)
  !$acc wait

  error_max = maxval(abs(px - px_ref))

  ! -------------------- begin: write error --------------------
  do pos=0,nproc-1
    if (pos==irank) then
      write(fname_60, '(A9,A11,A19,I0.6,A14)') './trials/', fname_11, '/output_halo/error_', irank, '__________.dat'
      open(54,file=fname_60)
        write(54,*) error_max
      close(54)
    endif
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  enddo

  ! -------------------- end: write error --------------------

  if (error_max>1e-10) then
    write(*,'(A22,E12.4,7I4,E12.5)') 'ERROR!! error_max:  ', error_max !, halo_ii%dir_pick, halo_ii%nh_ijk(0), halo_ii%nh_ijk(1), halo_ii%nh_ijk(2), size(px,1),size(px,2),size(px,3), elapse_time
  else
    write(*,'(A22,E12.4,7I4,E12.5)') 'SUCCESS!!! error_max: ', error_max !, halo_ii%dir_pick, halo_ii%nh_ijk(0), halo_ii%nh_ijk(1), halo_ii%nh_ijk(2), size(px,1),size(px,2),size(px,3), elapse_time
  endif
  write(*,*) 'Done! (Fortran)'

  flush(6)
call MPI_finalize(mpi_ierr)
end program


