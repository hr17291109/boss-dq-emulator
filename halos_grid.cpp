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
#include "binneddata.hpp"
#include "fielddata.hpp"
#include "fileloader.hpp"
#include "cosmo.hpp"
#include "util.hpp"
#include "binning.hpp"
#include "param.hpp"
#include "filenames.hpp"
#include <gsl/gsl_rng.h>
#include <time.h>
#include "pk_tools.hpp"

int main(int argc, char **argv){
    std::string FileBase = argv[1];
    int snapnum(atoi(argv[2]));
    std::string OutBase = argv[3];
    double v_th = atof(argv[4]);
    double delta_v = atof(argv[5]);
    int NS(atoi(argv[6]));

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
    int ng(500); // Number of grid points per dim for FFT
    int Npart1d;
    int zBOSS;

    switch(snapnum){
        case 0:
            redshift = 0.61;
            zBOSS = 3;
            break;
        case 1:
            redshift = 0.51;
            break;
        case 2:
            redshift = 0.38;
            zBOSS = 1;
            break;
        default:
            redshift = 0;
            break;
    }
    std::cerr << "Redshift: " << redshift << std::endl;

    param::parameter p1(FileBase+"/"+params_dir+"/"+nugenic_param_file);
    Omega_cb = p1.get<double>("Omega");
    Box = p1.get<double>("Box");
    Npart1d = (long long int)p1.get<int>("Npart");
    As = p1.get<double>("As1");
    ns = p1.get<double>("ns1");
    h0 = p1.get<double>("HubbleParam");
    k0 = p1.get<double>("kpivot")/h0;

    std::cout << "Info from " << FileBase+"/"+params_dir+"/"+nugenic_param_file << std::endl;
    std::cout << "Box: " << Box << std::endl;
    std::cerr << "Npart1d: " << Npart1d << std::endl;
    std::cout << "Omega_cb: " << Omega_cb << std::endl;
    std::cout << "As: " << As << std::endl;
    std::cout << "ns: " << ns << std::endl;
    std::cout << "h: " << h0 << std::endl;
    std::cout << "kpivot: " << k0 << std::endl;

    param::parameter p2(FileBase+"/"+params_dir+"/"+class_param_file);

    Omega_m = p2.get<double>("Omega_m");
    Omegam_fid = 0.31;

    std::cout << "Info from " << FileBase+"/"+params_dir+"/"+class_param_file << std::endl;
    std::cout << "Omega_m: " << Omega_m << std::endl << std::endl;

    std::string efile = FileBase+"/"+trans_dir+"/"+expansion_file;
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

    switch(NS){
        case 0:
            pk_file = "BOSSmultipoles/ps1D_BOSS_DR12_NGC_z" + itos(zBOSS) + "_COMPnbar_TSC_700_700_700_400_renorm.dat";
            Mfname = "BOSSmultipoles/M_BOSS_DR12_NGC_z"+ itos(zBOSS) +"_V6C_1_1_1_1_1_1200_2000.matrix";
            Wfname = "BOSSmultipoles/W_BOSS_DR12_NGC_z"+ itos(zBOSS) +"_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix";
            Cfname = "BOSSmultipoles/C_2048_BOSS_DR12_NGC_z"+ itos(zBOSS) +"_V6C_1_1_1_1_1_10_200_200_prerecon.matrix";
            break;
        case 1:
            pk_file = "BOSSmultipoles/ps1D_BOSS_DR12_SGC_z" + itos(zBOSS) + "_COMPnbar_TSC_700_700_700_400_renorm.dat";
            Mfname = "BOSSmultipoles/M_BOSS_DR12_SGC_z"+ itos(zBOSS) +"_V6C_1_1_1_1_1_1200_2000.matrix";
            Wfname = "BOSSmultipoles/W_BOSS_DR12_SGC_z"+ itos(zBOSS) +"_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix";
            Cfname = "BOSSmultipoles/C_2048_BOSS_DR12_SGC_z"+ itos(zBOSS) +"_V6C_1_1_1_1_1_10_200_200_prerecon.matrix";
            break;
    }

    Eigen::VectorXd Bpk(120);
    ps_fileload(pk_file, Bpk);

    Eigen::MatrixXd M(2000, 1200);
    Eigen::MatrixXd W(200, 2000);
    Eigen::MatrixXd C(120, 120);
    mwc_fileload(Mfname, Wfname, Cfname, M, W, C);

    long long int ii;
    double prob;
    double rand_num;

    gsl_rng * rand_ins;
    const gsl_rng_type * T = gsl_rng_default;
    rand_ins = gsl_rng_alloc (T);
    // gsl_rng_set(rand_ins, time(NULL));

    std::stringstream ss1;
    ss1 << std::fixed << std::setprecision(2) << delta_v;
    std::string value_str1 = ss1.str();


    FieldData Df1(ng,Box,false);
    FieldData Df2(ng,Box,false);
    FieldData halo_overdensity(ng,Box,true);

    double chi2 = 0;

    int Ng = 100;
    std::vector<double> chi2list(Ng);
    std::vector<double> dvlist(Ng);
    std::vector<double> vthlist(Ng);

    int p = 0;

    for(double m=delta_v; m < delta_v+ 10; m+=1.0){
        for(double k=v_th; k < v_th+10; k+=1.0){
            chi2 = 0;
            gsl_rng_set(rand_ins, 12345);
            ii = 0;
            halos.resize(halos_full.size());
    	    for(long long int i=0;i<halos.size();i++){
                prob = 0.5 * (1.0 + tanh((halos_full[i].mass - k) / m));
                rand_num = gsl_rng_uniform(rand_ins);

	        if(prob >= rand_num){
            	    halos[ii].mass = halos_full[i].mass;
	    	    for(int j=0;j<3;j++){
	                halos[ii].pos[j] = halos_full[i].pos[j];
	                halos[ii].vel[j] = halos_full[i].vel[j];
	            }
                    ii++;
	        }
            }

            halos.resize(ii);

            BinnedData pk0(nbins,kmin,kmax,logbin);
            BinnedData pk2(nbins,kmin,kmax,logbin);
            BinnedData pk4(nbins,kmin,kmax,logbin);

            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << k;
            std::string value_str = ss.str();

            for(int los_dir = 0; los_dir<3; los_dir++){

                Df1.clear_elements();
                Df1.change_space(false);
                Df1.assignment(halos,true,false,sfac,los_dir);
                Df1.do_fft();
                Df2.clear_elements();
                Df2.change_space(false);
                Df2.assignment(halos,true,true,sfac,los_dir);
                Df2.do_fft();
                Df2.adjust_grid(); // correct for the phase shift

                halo_overdensity.clear_elements();
                halo_overdensity.change_space(true);
                halo_overdensity.average2fields(Df1,Df2); // merge the 2 fields into one


                // Monopole moment
                int ell = 0;
                halo_overdensity.calc_power(pk0, ell, efile, Omegam_fid, redshift, los_dir);
                // BinnedData pk0 = halo_overdensity.calc_power(nbins, kmin, kmax, logbin, ell, efile, Omegam_fid, redshift, los_dir);
                // pk0.dump(OutBase+"_"+value_str+"_"+value_str1+"_pk0.dat");

                // Quadrupole moment
                ell = 2;
                halo_overdensity.calc_power(pk2, ell, efile, Omegam_fid, redshift, los_dir);
                // BinnedData pk2 = halo_overdensity.calc_power(nbins, kmin, kmax, logbin, ell, efile, Omegam_fid, redshift, los_dir);
                // pk2.dump(OutBase+"_"+value_str+"_"+value_str1+"_pk2.dat");

                // Hexadecapole moment
                ell = 4;
                halo_overdensity.calc_power(pk4, ell, efile, Omegam_fid, redshift, los_dir);
                // BinnedData pk4 = halo_overdensity.calc_power(nbins, kmin, kmax, logbin, ell, efile, Omegam_fid, redshift, los_dir);
                // pk4.dump(OutBase+"_"+value_str+"_"+value_str1+"_pk4.dat");

            }
            std::cout << "check delta_v = " << m << std::endl;
            std::cout << "check v_th = " << k << std::endl;
            chi_square(Bpk, M, W, C, pk0, pk2, pk4, chi2, 0.4);
            vthlist[p] = k;
            dvlist[p] = m;
            chi2list[p] = chi2;
            p++;
            std::cout << "chi2 = " << std::setprecision(16) << chi2 << std::endl;
            std::cout << "##############################################" << std::endl;
            pk0.dump(OutBase+"_"+value_str+"_"+value_str1+"_pk0.dat");
            pk2.dump(OutBase+"_"+value_str+"_"+value_str1+"_pk2.dat");
            pk4.dump(OutBase+"_"+value_str+"_"+value_str1+"_pk4.dat");
        }
    }
    gsl_rng_free(rand_ins);
    std::ofstream ofile(OutBase+"_chi2.dat");
    ofile << "delta_v" << " " << "Vmax_threshould" << " " << "chi2" << std::endl;
    ofile << std::setprecision(15);
    for (int i = 0; i < Ng; i++){
        ofile << dvlist[i] << " " << vthlist[i] << " " << chi2list[i] << std::endl;
    }
    exit(0);
}
