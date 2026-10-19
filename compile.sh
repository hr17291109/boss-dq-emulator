GIT_HASH=$(git describe --always --dirty --abbrev=7 2>/dev/null || echo "unknown")
echo "building from ${GIT_HASH}"

# g++ -O3 test.cpp -lfftw3f_omp -lfftw3f -o test.exe -fopenmp -lgsl -lgslcblas -I.
# g++ -O3 -I/home/nishimichi/local/include main.cpp -L/home/nishimichi/local/lib -lfftw3f_omp -lfftw3f -o measure_multipoles.exe -fopenmp -lgsl -lgslcblas -I.
g++ -DGIT_HASH="\"${GIT_HASH}\"" \
    -DZSPACE -O3 -g \
    halos_mcmc.cpp \
    -lyaml-cpp -lfftw3f_omp -lfftw3f -fopenmp -lgsl -lgslcblas \
    -I. -isystem /usr/include/eigen3 \
    -o halos_mcmc.exe
# g++ -DZSPACE -O3 halos_chi2.cpp -lfftw3f_omp -lfftw3f -o halos_chi2.exe -fopenmp -lgsl -lgslcblas -I. -I /usr/include/eigen3
