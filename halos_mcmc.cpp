#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <omp.h>
#include <fftw3.h>
#include <random>
#include <gsl/gsl_rng.h>
#include <time.h>
#include <yaml-cpp/yaml.h>
#include "binneddata.hpp"
#include "fielddata.hpp"
#include "fileloader.hpp"
#include "cosmo.hpp"
#include "util.hpp"
#include "binning.hpp"
#include "param.hpp"
#include "filenames.hpp"
#include "pk_tools.hpp"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return 1;
    }

    YAML::Node config    = YAML::LoadFile(argv[1]);
    std::string FileBase = config["file_base"].as<std::string>();
    int snapnum          = config["snapnum"].as<int>();
    std::string OutBase  = config["output_base"].as<std::string>();
    double v_th          = config["v_th"].as<double>();
    double delta_v       = config["delta_v"].as<double>();
    int NS               = config["ns_type"].as<int>();
    int Nmc              = config["n_mc"].as<int>();
    double fit_kmax      = config["kmax"].as<double>();
    int mcmc_seed = config["mcmc_seed"].as<int>(1);

    Eigen::MatrixXd cov_mat(2, 2);
    if (config["step_covariance"]) {
        YAML::Node cov_node = config["step_covariance"];

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                cov_mat(i, j) = cov_node[i][j].as<double>();
            }
        }
        std::cout << "Loaded Covariance Matrix:\n" << cov_mat << std::endl;
    }

    // Cosmological parameters
    double Omega_cb; // This is Omega_c + Omega_b, not Omega_m
    double Omega_m;  // Omega_m (=Omega_cb + Omega_{massive-neutrino})
    double h0;       // Dimensionless hubble parameter (should be around 0.7)
    double As;       // Amplitude of the primordial fluctuations
    double ns;       // Spectral index
    double k0;       // The "pivot" wavenumber at which As is given
    double Omegam_fid;

    // Simulation parameters
    double Box;      // in Mpc/h
    double redshift; // The output redshift
    int ng(512); // Number of grid points per dim for FFT
    int Npart1d;
    int zBOSS = -1;
    time_t t1 = time(0);

    switch(snapnum){
        case 0:
            redshift = 0.61;
            zBOSS    = 3;
            break;
        case 1:
            redshift = 0.51;
            break;
        case 2:
            redshift = 0.38;
            zBOSS    = 1;
            break;
        default:
            redshift = 0;
            std::cerr << "unsupported snapnum: " << snapnum << std::endl;
            return 1;
    }
    std::cerr << "Redshift: " << redshift << std::endl;

    param::parameter p1(FileBase + "/" + params_dir + "/" + nugenic_param_file);
    Omega_cb = p1.get<double>("Omega");
    Box      = p1.get<double>("Box");
    Npart1d  = (long long int)p1.get<int>("Npart");
    As       = p1.get<double>("As1");
    ns       = p1.get<double>("ns1");
    h0       = p1.get<double>("HubbleParam");
    k0       = p1.get<double>("kpivot") / h0;

    std::cout << "Info from " << FileBase + "/" + params_dir + "/" + nugenic_param_file << std::endl;
    std::cout << "Box: " << Box << std::endl;
    std::cerr << "Npart1d: " << Npart1d << std::endl;
    std::cout << "Omega_cb: " << Omega_cb << std::endl;
    std::cout << "As: " << As << std::endl;
    std::cout << "ns: " << ns << std::endl;
    std::cout << "h: " << h0 << std::endl;
    std::cout << "kpivot: " << k0 << std::endl;

    param::parameter p2(FileBase + "/" + params_dir + "/" + class_param_file);

    Omega_m    = p2.get<double>("Omega_m");
    Omegam_fid = 0.31;
    std::cout << "Info from " << FileBase + "/" + params_dir + "/" + class_param_file << std::endl;
    std::cout << "Omega_m: " << Omega_m << std::endl << std::endl;

    std::string efile = FileBase + "/" + trans_dir + "/" + expansion_file;
    double sfac(get_sfac(redshift,efile));

    // std::cout << "### Check Alcock-Paczynski related scales ###" << std::endl;
    // std::cout << "DA_true,org = " << ZtoComovingD_Q0000(redshift,Omega_m) << ",  DA_fid,org = " << ZtoComovingD_Q0000(redshift,Omegam_fid)
    //     << ",  H_true,org = " << Hz_Q0000(redshift,Omega_m) << ",   H_fid,org = " << Hz_Q0000(redshift,Omegam_fid) << std::endl;

    // std::cout << "DA_true,new = " << ZtoComovingD(redshift,efile) << ",  DA_fid,new = " << ZtoComovingD_LCDM(redshift,Omegam_fid)
    //     << ",  H_true,new = " << Hz(redshift,efile) << ",   H_fid,new = " << Hz_LCDM(redshift,Omegam_fid) << std::endl;

    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());
    std::cout << omp_get_max_threads() << " threads will be used in FFT." << std::endl;
    // fftwf_import_wisdom_from_filename(wisdom_file.c_str());

    // FieldData Df_lin(ng,Box,true);
    // fftwf_export_wisdom_to_filename(wisdom_file.c_str());
    // Df_lin.load(FileBase+"/"+white_dir+"/"+white_file);

    std::cerr << "### Load halos from " << FileBase+"/halo_catalog" << " ###" << std::endl;
    std::vector<myhosthalo_str> halos;
    std::vector<myhosthalo_full_str> halos_full;

    load_halo_full_vmaxthreshold(FileBase, snapnum, halos_full);

    std::string pk_file;
    std::string Mfname;
    std::string Wfname;
    std::string Cfname;

    switch(NS) {
        case 0:
            pk_file = "BOSSmultipoles/ps1D_BOSS_DR12_NGC_z" + itos(zBOSS) + "_COMPnbar_TSC_700_700_700_400_renorm.dat";
            Mfname  = "BOSSmultipoles/M_BOSS_DR12_NGC_z" + itos(zBOSS) + "_V6C_1_1_1_1_1_1200_2000.matrix";
            Wfname  = "BOSSmultipoles/W_BOSS_DR12_NGC_z" + itos(zBOSS) + "_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix";
            Cfname  = "BOSSmultipoles/C_2048_BOSS_DR12_NGC_z" + itos(zBOSS) + "_V6C_1_1_1_1_1_10_200_200_prerecon.matrix";
            break;
        case 1:
            pk_file = "BOSSmultipoles/ps1D_BOSS_DR12_SGC_z" + itos(zBOSS) + "_COMPnbar_TSC_700_700_700_400_renorm.dat";
            Mfname  = "BOSSmultipoles/M_BOSS_DR12_SGC_z" + itos(zBOSS) + "_V6C_1_1_1_1_1_1200_2000.matrix";
            Wfname  = "BOSSmultipoles/W_BOSS_DR12_SGC_z" + itos(zBOSS) + "_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix";
            Cfname  = "BOSSmultipoles/C_2048_BOSS_DR12_SGC_z" + itos(zBOSS) + "_V6C_1_1_1_1_1_10_200_200_prerecon.matrix";
            break;
		default:
			std::cerr << "Invalid ns_type: " << NS << std::endl;
    		return 1;
    }

    Eigen::VectorXd Bpk(120);
    ps_fileload(pk_file, Bpk);

    Eigen::MatrixXd M(2000, 1200);
    Eigen::MatrixXd W(200, 2000);
    Eigen::MatrixXd C(120, 120);
    mwc_fileload(Mfname, Wfname, Cfname, M, W, C);
    Eigen::MatrixXd WM = W * M;

    int k = 0;
    long long int ii;
    double prob;
    double rand_num;

    gsl_rng * rand_halo;
    gsl_rng * rand_mcmc;
    gsl_rng_env_setup();
    const gsl_rng_type * T = gsl_rng_default;
    rand_halo = gsl_rng_alloc(T);
    rand_mcmc = gsl_rng_alloc(T);
    gsl_rng_set(rand_mcmc, mcmc_seed);
    std::cout << "mcmc_seed: " << mcmc_seed << std::endl;

    FieldData Df1(ng, Box, false);
    FieldData Df2(ng, Box, false);
    FieldData halo_overdensity(ng, Box, true);

    double chi2 = 0;

    std::vector<double> chi2list(Nmc);
    std::vector<double> dvlist(Nmc);
    std::vector<double> vthlist(Nmc);
    std::ofstream dfile(OutBase+"_others_cov_chi2.dat");
    dfile << "delta_v" << " " << "Vmax_threshould" << " " << "chi2" << std::endl;
    dfile << std::setprecision(15);
    std::ofstream ofile(OutBase+"_cov_chi2.dat");
    ofile << "# git: " << GIT_HASH << "\n";
    ofile << "# config: " << argv[1] << "\n";
    ofile << "# v_th_init: " << v_th << " delta_v_init: " << delta_v << "\n";
    ofile << "# step_cov: " << cov_mat(0,0) << " " << cov_mat(0,1) << " " << cov_mat(1,1) << "\n";
    ofile << "# mcmc_seed: " << mcmc_seed << "\n";
    ofile << "# fit_kmax: " << fit_kmax << "\n";
    ofile << "# n_mc: " << Nmc << "\n";
    ofile << "delta_v" << " " << "Vmax_threshould" << " " << "chi2" << std::endl;
    ofile << std::setprecision(15);

    for (k = 0; k < Nmc; k++) {
        std::cout << "loop number: " << k << std::endl;
        chi2 = 0;
        gsl_rng_set(rand_halo, 12345);
        ii = 0;
        std::cout << "resize halos ... ";
        halos.resize(halos_full.size());
        std::cout << "done." << std::endl;
        std::cout << "select halos ... ";
    	for (long long int i = 0; i < halos.size(); i++) {
            prob = 0.5 * (1.0 + tanh((halos_full[i].mass - v_th) / delta_v));
            rand_num = gsl_rng_uniform(rand_halo);

            if (prob >= rand_num) {
                halos[ii].mass = halos_full[i].mass;
                for (int j = 0; j < 3; j++) {
                    halos[ii].pos[j] = halos_full[i].pos[j];
                    halos[ii].vel[j] = halos_full[i].vel[j];
                }
                ii++;
            }
        }

        halos.resize(ii);
        std::cout << "done." << std::endl;

        BinnedData pk0(nbins, kmin, kmax, logbin);
        BinnedData pk2(nbins, kmin, kmax, logbin);
        BinnedData pk4(nbins, kmin, kmax, logbin);

        for (int los_dir = 0; los_dir < 3; los_dir++) {

            Df1.clear_elements();
            Df1.change_space(false);
            Df1.assignment(halos, true, false, sfac, los_dir);
            Df1.do_fft();
            Df2.clear_elements();
            Df2.change_space(false);
            Df2.assignment(halos, true, true, sfac, los_dir);
            Df2.do_fft();
            Df2.adjust_grid(); // correct for the phase shift

            halo_overdensity.clear_elements();
            halo_overdensity.change_space(true);
            halo_overdensity.average2fields(Df1, Df2); // merge the 2 fields into one

            // Monopole moment
            int ell = 0;
            halo_overdensity.calc_power(pk0, ell, efile, Omegam_fid, redshift, los_dir);

            // Quadrupole moment
            ell = 2;
            halo_overdensity.calc_power(pk2, ell, efile, Omegam_fid, redshift, los_dir);

            // Hexadecapole moment
            ell = 4;
            halo_overdensity.calc_power(pk4, ell, efile, Omegam_fid, redshift, los_dir);
        }
        std::cout << "check delta_v = " << delta_v << std::endl;
        std::cout << "check v_th = " << v_th << std::endl;
        chi_square(Bpk, WM, C, pk0, pk2, pk4, chi2, fit_kmax);
        //chi_square_hartlap(Bpk, WM, C, pk0, pk2, pk4, chi2, fit_kmax);
        std::cout << "chi2 = " << std::setprecision(16) << chi2 << std::endl;
        mcmc(chi2, delta_v, v_th, chi2list, dvlist, vthlist, rand_mcmc, k, ofile, dfile, cov_mat);

        time_t t2 = time(0);
        std::cout << "finish time: " << t2 - t1 << std::endl;
        std::cout << "##############################################" << std::endl;
    }
    gsl_rng_free(rand_halo);
    gsl_rng_free(rand_mcmc);
    dfile.close();
    ofile.close();

    return 0;
}
