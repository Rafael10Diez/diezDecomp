program main
#if defined(_OPENACC)
  use openacc
#endif
#if defined(_USE_CUDA)
  use cudafor
#endif
  use mpi
  use, intrinsic :: iso_fortran_env, only: i8 => int64, sp => real32, dp => real64
  use diezdecomp_api_cans
  use diezdecomp_api_generic
#if defined(_OPENACC)
  use openacc, only: acc_handle_kind
#else
  use, intrinsic :: iso_fortran_env, only: acc_handle_kind => int64
#endif
  implicit none
#if defined(_DIEZDECOMP_SINGLE)
  integer, parameter :: rp = sp
#else
  integer, parameter :: rp = dp
#endif
  integer                           ::  ii, jj, kk, pos, i, j, k, iter, irank, nproc, mpi_ierr
  real(rp)                          ::  error_max, error_max_x, error_max_y
  integer                           ::  order_x(0:2), order_y(0:2), order_intermediate(0:2), &
                                        sx0, sx1, sx2, sy0, sy1, sy2, &
                                        lo_a(0:2), lo_b(0:2), hi_a(0:2), hi_b(0:2), &
                                        read_int1, read_int2, read_int3, read_int4, read_int5, &
                                        send_autotuned, recv_autotuned, send_mode_op_simul, recv_mode_op_simul,&
                                        force_send_autotune, force_recv_autotune, offset6_in(0:2,0:1), offset6_out(0:2,0:1), &
                                        offset6_in_bad(0:2,0:1), offset6_out_bad(0:2,0:1), &
                                        send_mode_op_batched, recv_mode_op_batched,i0,i1,j0,j1,k0,k1
  logical                           ::  allow_alltoallv, mode_api_cans, allow_autotune_reorder, mode_use_buf, &
                                        use_cans_change_offsets
  character(len=62)                 ::  fname_62
  character(len=11)                 ::  fname_11
  real(rp)                          ::  read_real, elapse_time
  real(rp), allocatable, target     ::  py_ref(:,:,:), buffer(:), px_orig(:,:,:), py_buf(:), px_buf(:), px_bad(:,:,:), py_bad(:,:,:)
  real(rp), pointer                 ::  py(:,:,:), px(:,:,:)
  integer(i8)                       ::  wsize
  type(diezdecomp_props_transp)     ::  tr
  type(diezDecompHandle)            ::  ch
  type(diezDecompGridDesc)          ::  gd
  type(diezdecomp_parsed_mpi_ranks) ::  ox, oy
  integer(acc_handle_kind)          ::  stream = 1
#if defined(_OPENACC)
  integer(acc_device_kind)          ::  dev_type
#endif

  call mpi_init     (mpi_ierr)
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
      call mpi_barrier(mpi_comm_world,mpi_ierr)
      call mpi_barrier(mpi_comm_world,mpi_ierr)
      call get_command_argument(1, fname_11)

      write(fname_62, '(A9,A11,A20,I0.6,A16)') './trials/', fname_11, '/input_transp/input_', irank, '________info.dat'
      open(54,file=fname_62)
        read(54,*) ii, jj, kk, read_int1, read_int2, read_int3, read_int4
        allow_alltoallv        = (read_int1==1)
        mode_api_cans          = (read_int2==1)
        allow_autotune_reorder = (read_int3==1)
        mode_use_buf           = (read_int4==1)
        read(54,*) order_x
        read(54,*) order_y
        read(54,*) order_intermediate
        read(54,*) lo_a
        read(54,*) lo_b
        read(54,*) hi_a
        read(54,*) hi_b
        read(54,*) offset6_in
        read(54,*) offset6_out
        read(54,*) read_int5
        use_cans_change_offsets = (read_int5==1)
        read(54,*) offset6_in_bad
        read(54,*) offset6_out_bad

        read(54,*) sx0,sx1,sx2
        allocate(px_orig(0:sx0-1,0:sx1-1,0:sx2-1))
        do iter=1,(sx0*sx1*sx2)
          read(54,*) i,j,k,read_real
          px_orig(i,j,k) = read_real
        end do

        read(54,*) sy0,sy1,sy2
        allocate(py_ref(0:sy0-1, 0:sy1-1, 0:sy2-1))
        do iter=1,(sy0*sy1*sy2)
          read(54,*) i,j,k,read_real
          py_ref(i,j,k) = read_real
        end do

        read(54,*) force_send_autotune, force_recv_autotune
        read(54,*) send_autotuned     , recv_autotuned
        read(54,*) send_mode_op_simul , recv_mode_op_simul
        read(54,*) send_mode_op_batched, recv_mode_op_batched
      close(54)
    endif
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  enddo
  ! -------------------- end: read data --------------------

  if (mode_api_cans.and.(.not.mode_use_buf)) error stop 'mode_api_cans.and.(.not.mode_use_buf)'

  if (mode_use_buf) then
   allocate(px_buf(0:(sx0*sx1*sx2-1)))
   allocate(py_buf(0:(sy0*sy1*sy2-1)))
  else
   allocate(px_buf(0:(max(sx0*sx1*sx2,sy0*sy1*sy2) - 1)))
   allocate(py_buf,mold=px_buf)
  end if

  px(0:sx0-1, 0:sx1-1, 0:sx2-1) => px_buf(0:(sx0*sx1*sx2-1))
  py(0:sy0-1, 0:sy1-1, 0:sy2-1) => py_buf(0:(sy0*sy1*sy2-1))
  px = px_orig

  i0                     =        offset6_out(0,0)
  i1                     =  sy0 - offset6_out(0,1)-1
  j0                     =        offset6_out(1,0)
  j1                     =  sy1 - offset6_out(1,1)-1
  k0                     =        offset6_out(2,0)
  k1                     =  sy2 - offset6_out(2,1)-1
  py                     =  py_ref
  py(i0:i1,j0:j1,k0:k1)  =  -1

  if (mode_api_cans) then
    gd%all_ap(ii)%internal_order(1:3)  =  order_x(0:2)
    gd%all_ap(jj)%internal_order(1:3)  =  order_y(0:2)

    gd%all_ap(ii)%lo(1:3)  =  lo_a(0:2) + 1
    gd%all_ap(ii)%hi(1:3)  =  hi_a(0:2) + 1
    gd%all_ap(jj)%lo(1:3)  =  lo_b(0:2) + 1
    gd%all_ap(jj)%hi(1:3)  =  hi_b(0:2) + 1

    if (use_cans_change_offsets) then 
      gd%all_ap(ii)%offset6  =  offset6_in_bad
      gd%all_ap(jj)%offset6  =  offset6_out_bad
    else
      gd%all_ap(ii)%offset6  =  offset6_in
      gd%all_ap(jj)%offset6  =  offset6_out
    end if

    gd%irank                           =  irank
    gd%nproc                           =  nproc

    gd%abs_reorder(:,:,0)  =  order_intermediate(0)
    gd%abs_reorder(:,:,1)  =  order_intermediate(1)
    gd%abs_reorder(:,:,2)  =  order_intermediate(2)

    call diezdecomp_track_mpi_decomp(lo_a, ox, irank, nproc, order_x)
    call diezdecomp_track_mpi_decomp(lo_b, oy, irank, nproc, order_y)

    allocate(gd%all_ap(ii)%flat_mpi_ranks, mold=ox%flat_mpi_ranks)
    gd%all_ap(ii)%flat_mpi_ranks(:,:) = ox%flat_mpi_ranks(:,:)

    allocate(gd%all_ap(jj)%flat_mpi_ranks, mold=oy%flat_mpi_ranks)
    gd%all_ap(jj)%flat_mpi_ranks(:,:) = oy%flat_mpi_ranks(:,:)
    wsize = ( (sx0 - offset6_in(0,0)  - offset6_in(0,1) ) * &
              (sx1 - offset6_in(1,0)  - offset6_in(1,1) ) * &
              (sx2 - offset6_in(2,0)  - offset6_in(2,1) ) + &
              (sy0 - offset6_out(0,0) - offset6_out(0,1)) * &
              (sy1 - offset6_out(1,0) - offset6_out(1,1)) * &
              (sy2 - offset6_out(2,0) - offset6_out(2,1)) )
    diezdecomp_allow_autotune_reorder = allow_autotune_reorder
  else
    block
      integer :: spx(0:2), spy(0:2)
      write(6,*) 'begin: diezdecomp_generic_fill_tr_obj' ; flush(6)
      spx = [sx0, sx1, sx2]
      spy = [sy0, sy1, sy2]

      call diezdecomp_track_mpi_decomp(lo_a, ox, irank, nproc, order_x)
      call diezdecomp_track_mpi_decomp(lo_b, oy, irank, nproc, order_y)
      call diezdecomp_generic_fill_tr_obj(tr, ox, oy, spx, offset6_in, spy, offset6_out, order_x, order_y, &
                                          order_intermediate, allow_alltoallv, ii, jj, wsize, &
                                          allow_autotune_reorder, stream)
      write(6,*) 'end: diezdecomp_generic_fill_tr_obj' ; flush(6)
    end block
  end if
  write(6,*) irank, 'wsize: ', wsize
  allocate(buffer(wsize))
  buffer = -12

  !$acc wait
  !$acc enter data copyin(px_buf, py_buf, buffer)
  !$acc wait


  if (mode_api_cans) then
    block 
      integer :: i_halo(0:2), o_halo(0:2), i_pad(0:2), o_pad(0:2)
      i_halo = -1
      o_halo = -1
      i_pad  = -1
      o_pad  = -1
      if (use_cans_change_offsets) then 
        allocate(px_bad(0:(sx0 -  offset6_in(0,0) -  offset6_in(0,1) +  offset6_in_bad(0,0) +  offset6_in_bad(0,1) )-1,&
                        0:(sx1 -  offset6_in(1,0) -  offset6_in(1,1) +  offset6_in_bad(1,0) +  offset6_in_bad(1,1) )-1,&
                        0:(sx2 -  offset6_in(2,0) -  offset6_in(2,1) +  offset6_in_bad(2,0) +  offset6_in_bad(2,1) )-1),&
                 py_bad(0:(sy0 - offset6_out(0,0) - offset6_out(0,1) + offset6_out_bad(0,0) + offset6_out_bad(0,1) )-1,&
                        0:(sy1 - offset6_out(1,0) - offset6_out(1,1) + offset6_out_bad(1,0) + offset6_out_bad(1,1) )-1,&
                        0:(sy2 - offset6_out(2,0) - offset6_out(2,1) + offset6_out_bad(2,0) + offset6_out_bad(2,1) )-1))
        call diezDecomp_boilerplate_transpose(gd, px_bad, py_bad, buffer, ii, jj,allow_alltoallv, stream, .false., &
                                              i_halo, o_halo, i_pad, o_pad)
      else
        call diezDecomp_boilerplate_transpose(gd, px, py, buffer, ii, jj,allow_alltoallv, stream, .false., &
                                              i_halo, o_halo, i_pad, o_pad)
      end if
    end block 
    !$acc wait
    associate(obj => gd%obj_tr(ii,jj))
      if (force_send_autotune==1) then
        obj%send_autotuned       = send_autotuned
        obj%send_mode_op_batched = send_mode_op_batched
        obj%send_mode_op_simul   = send_mode_op_simul
      end if
      if (force_recv_autotune==1) then
        obj%recv_autotuned       = recv_autotuned
        obj%recv_mode_op_simul   = recv_mode_op_simul
        obj%recv_mode_op_batched = recv_mode_op_batched
      end if
      write(6,*) irank,'mode_api_cans'           , mode_api_cans            ;flush(6)
      write(6,*) irank,'use_cans_change_offsets' , use_cans_change_offsets  ;flush(6)
      write(6,*) irank,'mode_use_buf'            , mode_use_buf             ;flush(6)
      write(6,*) irank,'force_send_autotune'     , force_send_autotune      ;flush(6)
      write(6,*) irank,'obj%send_autotuned'      , obj%send_autotuned       ;flush(6)
      write(6,*) irank,'obj%send_mode_op_simul'  , obj%send_mode_op_simul   ;flush(6)
      write(6,*) irank,'obj%send_mode_op_batched', obj%send_mode_op_batched ;flush(6)
      write(6,*) irank,'force_recv_autotune'     , force_recv_autotune      ;flush(6)
      write(6,*) irank,'obj%recv_autotuned'      , obj%recv_autotuned       ;flush(6)
      write(6,*) irank,'obj%recv_mode_op_simul'  , obj%recv_mode_op_simul   ;flush(6)
      write(6,*) irank,'obj%recv_mode_op_batched', obj%recv_mode_op_batched ;flush(6)
    end associate
  else
    associate(obj => tr)
      if (force_send_autotune==1) then
        obj%send_autotuned       = send_autotuned
        obj%send_mode_op_batched = send_mode_op_batched
        obj%send_mode_op_simul   = send_mode_op_simul
      end if
      if (force_recv_autotune==1) then
        obj%recv_autotuned       = recv_autotuned
        obj%recv_mode_op_simul   = recv_mode_op_simul
        obj%recv_mode_op_batched = recv_mode_op_batched
      end if
      write(6,*) irank,'mode_api_cans'           , mode_api_cans            ;flush(6)
      write(6,*) irank,'use_cans_change_offsets' , use_cans_change_offsets  ;flush(6)
      write(6,*) irank,'mode_use_buf'            , mode_use_buf             ;flush(6)
      write(6,*) irank,'force_send_autotune'     , force_send_autotune      ;flush(6)
      write(6,*) irank,'obj%send_autotuned'      , obj%send_autotuned       ;flush(6)
      write(6,*) irank,'obj%send_mode_op_simul'  , obj%send_mode_op_simul   ;flush(6)
      write(6,*) irank,'obj%send_mode_op_batched', obj%send_mode_op_batched ;flush(6)
      write(6,*) irank,'force_recv_autotune'     , force_recv_autotune      ;flush(6)
      write(6,*) irank,'obj%recv_autotuned'      , obj%recv_autotuned       ;flush(6)
      write(6,*) irank,'obj%recv_mode_op_simul'  , obj%recv_mode_op_simul   ;flush(6)
      write(6,*) irank,'obj%recv_mode_op_batched', obj%recv_mode_op_batched ;flush(6)
    end associate
  end if

#if defined(_OPENACC)
    write(6,*) 'Openacc detected!' ; flush(6)
    px   = -200
#else
    write(6,*) 'Openacc not found!' ; flush(6)
#endif
  write(6,'(I3,A14,6I3)') irank, ' offset6_in : ', offset6_in  ; flush(6)
  write(6,'(I3,A14,6I3)') irank, ' offset6_out: ', offset6_out ; flush(6)
  write(6,*) 'begin: transp_execute' ; flush(6)
  !$acc wait
  elapse_time = MPI_Wtime()
  if (mode_api_cans)  then
    block 
      integer :: i_halo(0:2), o_halo(0:2), i_pad(0:2), o_pad(0:2)
      if (use_cans_change_offsets) then 
        i_pad  = offset6_in(:,1) - offset6_in(:,0)
        i_halo = offset6_in(:,1) - i_pad
        o_pad  = offset6_out(:,1) - offset6_out(:,0)
        o_halo = offset6_out(:,1) - o_pad
        call diezDecomp_boilerplate_transpose(gd, px, py, buffer, ii, jj,allow_alltoallv, stream, .false., &
                                              i_halo, o_halo, i_pad, o_pad)
      else
        i_halo = -1
        o_halo = -1
        i_pad  = -1
        o_pad  = -1
        call diezDecomp_boilerplate_transpose(gd, px, py, buffer, ii, jj,allow_alltoallv, stream, .false., &
                                              i_halo, o_halo, i_pad, o_pad)
      end if
    end block 
  else
    if (mode_use_buf) then
      call diezdecomp_transp_execute_generic_buf(tr, px, py, buffer, stream)
    else
      call diezdecomp_transp_execute_generic_nobuf(tr, px_buf, py_buf, stream)
    end if
  end if
  !$acc wait
  !$acc exit data copyout(px_buf,py_buf)
  !$acc wait
  elapse_time   = MPI_Wtime() - elapse_time
  write(6,*) 'end: transp_execute' ; flush(6)

  if (mode_use_buf) then
    error_max_x = maxval(abs(px - px_orig))
    error_max_y = maxval(abs(py - py_ref ))

  else
    error_max_x = 0
    error_max_y = maxval(abs(py(i0:i1,j0:j1,k0:k1) - py_ref(i0:i1,j0:j1,k0:k1)))
  end if

  error_max = max(error_max_x, error_max_y)

  ! -------------------- begin: write error --------------------
  do pos=0,nproc-1
    if (pos==irank) then
      write(fname_62, '(A9,A11,A21,I0.6,A15)') './trials/', fname_11, '/output_transp/error_', irank, '___________.dat'
      open(54,file=fname_62)
        write(54,*) error_max
      close(54)
    endif
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  call mpi_barrier(mpi_comm_world,mpi_ierr)
  enddo
  ! -------------------- end: write error --------------------

  if (error_max>1e-10) then
    write(*,'(A22,2E12.4)') 'ERROR!! error_max:  ', error_max_x, error_max_y
  else
    write(*,'(A22,E12.4)') 'SUCCESS!!! error_max: ', error_max
  endif
  write(*,*) 'Done! (Fortran)'
  if (mode_api_cans)  then
    call diezdecomp_summary_transp_autotuning(gd%obj_tr(ii,jj))
  else
    call diezdecomp_summary_transp_autotuning(tr)
  end if

call MPI_finalize(mpi_ierr)
end program


