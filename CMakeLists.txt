cmake_minimum_required(VERSION 3.18)
project(NBodySimulation C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall -Wextra")

# ---- Include path -----------------------------------------------------------
include_directories(include)

# ---- Common source ----------------------------------------------------------
set(COMMON_SRC src/common.c)

# ---- Output directory -------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ==============================================================================
# Serial
# ==============================================================================
add_executable(nbody_serial src/serial.c ${COMMON_SRC})
target_link_libraries(nbody_serial m)

# ==============================================================================
# OpenMP
# ==============================================================================
find_package(OpenMP)
if(OpenMP_C_FOUND)
    add_executable(nbody_openmp src/openmp.c ${COMMON_SRC})
    target_link_libraries(nbody_openmp m OpenMP::OpenMP_C)
    message(STATUS "OpenMP found — nbody_openmp will be built.")
else()
    message(WARNING "OpenMP not found — nbody_openmp will NOT be built.")
endif()

# ==============================================================================
# POSIX Threads
# ==============================================================================
find_package(Threads REQUIRED)
add_executable(nbody_pthreads src/pthreads.c ${COMMON_SRC})
target_link_libraries(nbody_pthreads m Threads::Threads)

# ==============================================================================
# MPI
# ==============================================================================
find_package(MPI)
if(MPI_C_FOUND)
    add_executable(nbody_mpi src/mpi.c ${COMMON_SRC})
    target_link_libraries(nbody_mpi m MPI::MPI_C)
    message(STATUS "MPI found — nbody_mpi will be built.")

    # ==========================================================================
    # Hybrid (MPI + OpenMP)
    # ==========================================================================
    if(OpenMP_C_FOUND)
        add_executable(nbody_hybrid src/hybrid.c ${COMMON_SRC})
        target_link_libraries(nbody_hybrid m MPI::MPI_C OpenMP::OpenMP_C)
        message(STATUS "MPI + OpenMP found — nbody_hybrid will be built.")
    else()
        message(WARNING "OpenMP not found — nbody_hybrid will NOT be built.")
    endif()
else()
    message(WARNING "MPI not found — nbody_mpi and nbody_hybrid will NOT be built.")
endif()

# ==============================================================================
# CUDA (optional)
# ==============================================================================
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(nbody_cuda src/cuda_nbody.cu ${COMMON_SRC})
    target_link_libraries(nbody_cuda m)
    set_target_properties(nbody_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    message(STATUS "CUDA found — nbody_cuda will be built.")
else()
    message(WARNING "CUDA not found — nbody_cuda will NOT be built.")
endif()
