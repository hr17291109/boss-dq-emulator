#pragma once
#include "binning.hpp"
#include "binneddata.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <Eigen/Dense>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "binneddata.hpp"

void ps_fileload(std::string fname, Eigen::VectorXd &pk) {
    Eigen::VectorXd k_ps(400);
    Eigen::VectorXd k_eff(400);
    Eigen::VectorXd pk0(400);
    Eigen::VectorXd pk1(400);
    Eigen::VectorXd pk2(400);
    Eigen::VectorXd pk3(400);
    Eigen::VectorXd pk4(400);
    Eigen::VectorXd sig0(400);
    Eigen::VectorXd sig1(400);
    Eigen::VectorXd sig2(400);
    Eigen::VectorXd sig3(400);
    Eigen::VectorXd sig4(400);
    Eigen::VectorXd modes(400);

    int within_header = 0;
    std::ifstream file(fname);
    std::cout << "check pkfilename :" << fname << std::endl;
    std::cout << "load BOSS pk file ... ";
    std::string line;
    int i = 0;

    while (std::getline(file, line)) {
        if (within_header < 2) {
            if (line == "### header ###") {
                within_header += 1;
            }
        } else {
            std::istringstream iss(line);
            iss >> k_ps(i);
            iss >> k_eff(i);
            iss >> pk0(i);
            iss >> sig0(i);
            iss >> pk1(i);
            iss >> sig1(i);
            iss >> pk2(i);
            iss >> sig2(i);
            iss >> pk3(i);
            iss >> sig3(i);
            iss >> pk4(i);
            iss >> sig4(i);
            iss >> modes(i);
            i++;
        }

    }
    file.close();
    std::cout << "done." << std::endl;

    using RowMajorMatrix40x10 = Eigen::Matrix<double, 40, 10, Eigen::RowMajor>;
    Eigen::Map<RowMajorMatrix40x10> weight(modes.data());
    Eigen::Map<RowMajorMatrix40x10> pk0mat(pk0.data());
    Eigen::Map<RowMajorMatrix40x10> pk1mat(pk1.data());
    Eigen::Map<RowMajorMatrix40x10> pk2mat(pk2.data());
    Eigen::Map<RowMajorMatrix40x10> pk3mat(pk3.data());
    Eigen::Map<RowMajorMatrix40x10> pk4mat(pk4.data());
    Eigen::VectorXd wsum = weight.rowwise().sum();
    Eigen::VectorXd wmean0 = pk0mat.cwiseProduct(weight).rowwise().sum().cwiseQuotient(wsum);
    Eigen::VectorXd wmean2 = pk2mat.cwiseProduct(weight).rowwise().sum().cwiseQuotient(wsum);
    Eigen::VectorXd wmean4 = pk4mat.cwiseProduct(weight).rowwise().sum().cwiseQuotient(wsum);
    pk << wmean0, wmean2, wmean4;

}

void mwc_fileload(std::string Mfname, std::string Wfname, std::string Cfname, Eigen::MatrixXd &M, Eigen::MatrixXd &W, Eigen::MatrixXd &C) {
    M.setZero();
    W.setZero();
    C.setZero();

    std::cout <<  "load M file ... ";
    std::ifstream Mfile(Mfname);
    std::string line;
    for (int i = 0; i < 2000; i++) {
        std::getline(Mfile, line);
        std::istringstream issm(line);
        for (int j = 0; j < 1200; j++) {
            issm >> M(i, j);
        }
    }
    Mfile.close();
    std::cout <<  "done." << std::endl;
    std::cout <<  "M rows: " << M.rows() << std::endl;
    std::cout <<  "M cols: " << M.cols() << std::endl;

    std::cout <<  "load W file ... ";
    std::ifstream Wfile(Wfname);
    for (int i = 0; i < 200; i++) {
        std::getline(Wfile, line);
        std::istringstream issw(line);
        for (int j = 0; j < 2000; j++) {
            issw >> W(i, j);
        }
    }
    Wfile.close();
    std::cout <<  "done." << std::endl;
    std::cout <<  "W rows: " << W.rows() << std::endl;
    std::cout <<  "W cols: " << W.cols() << std::endl;

    std::cout <<  "load C file ... ";
    std::ifstream Cfile(Cfname);

    Eigen::VectorXd dummy(25600);

    int ii = 0;
    int jj = 0;
    int k = 0;
    for (int i = 0; i < 200; i++) {
        std::getline(Cfile, line);
        std::istringstream issc(line);
        jj = 0;
        for (int j = 0; j < 200; j++) {
            if ((i >= 40 && i < 80) || (i >= 120 && i < 160) || (j >= 40 && j < 80) || (j >= 120 && j < 160)) {
                issc >> dummy(k);
                k++;
            } else {
                issc >> C(ii, jj);
                jj++;
            }
        }
        if (i < 40 || (i >= 80 && i < 120) || i >= 160) {
            ii++;
        }
    }
    Cfile.close();
    std::cout <<  "done." << std::endl;
    std::cout <<  "C rows: " << C.rows() << std::endl;
    std::cout <<  "C cols: " << C.cols() << std::endl;
}

inline double hartlap_factor(int n_data, int n_mocks = 2048) {
    return double(n_mocks - n_data - 2) / double(n_mocks - 1);
}

void chi_square(const Eigen::VectorXd &Bpk, const Eigen::MatrixXd &WM,
                const Eigen::MatrixXd &C, const BinnedData &pk0,
                const BinnedData &pk2, const BinnedData &pk4,
                double &x2, double fit_kmax, bool use_hartlap) {
    constexpr int kNbinPerMultipole = 40;

    Eigen::VectorXd pk(1200);
    Eigen::VectorXd wmp(200);
    Eigen::VectorXd psim0(pk0.get_nbin());
    Eigen::VectorXd psim2(pk2.get_nbin());
    Eigen::VectorXd psim4(pk4.get_nbin());
    Eigen::VectorXd psim_full(120);

    for (int i = 0; i < pk0.get_nbin(); i++) {
        psim0(i) = pk0.get_ymean(i);
        psim2(i) = pk2.get_ymean(i);
        psim4(i) = pk4.get_ymean(i);
    }

    pk << psim0, psim2, psim4;
    wmp = WM*pk;

    int j = 0;
    for (int i = 0; i < kNbinPerMultipole*5; i++) {
        if (i < kNbinPerMultipole ||
            (i >= kNbinPerMultipole*2 && i < kNbinPerMultipole*3) ||
            i >= kNbinPerMultipole*4) {
            psim_full(j) = wmp(i);
            j++;
        }
    }

    constexpr double kBinWidth = kmax / kNbinPerMultipole;
    int limit = static_cast<int>(std::round(fit_kmax / kBinWidth));
    if (limit < 1 || limit > kNbinPerMultipole) {
        std::cerr << "invalid fit_kmax: " << fit_kmax << std::endl;
        std::exit(1);
    }

    int sub_size = limit * 3;
    Eigen::MatrixXd C_sub(sub_size, sub_size);
    Eigen::VectorXd Bpk_sub(sub_size);
    Eigen::VectorXd psim_sub(sub_size);

    for (int m_row = 0; m_row < 3; m_row++) {
        int orig_row_start = m_row * kNbinPerMultipole;
        int sub_row_start = m_row * limit;

        Bpk_sub.segment(sub_row_start, limit) = Bpk.segment(orig_row_start, limit);
        psim_sub.segment(sub_row_start, limit) = psim_full.segment(orig_row_start, limit);

        for (int m_col = 0; m_col < 3; m_col++) {
            int orig_col_start = m_col * kNbinPerMultipole;
            int sub_col_start = m_col * limit;

            C_sub.block(sub_row_start, sub_col_start, limit, limit) =
                C.block(orig_row_start, orig_col_start, limit, limit);
        }
    }

    Eigen::VectorXd delta = Bpk_sub - psim_sub;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(C_sub);
    x2 = delta.dot(ldlt.solve(delta));
    if (use_hartlap) x2 *= hartlap_factor(sub_size);
    //Eigen::MatrixXd Cinv_sub = C_sub.inverse();
    //x2 = delta.transpose() * Cinv_sub * delta;
}

void mcmc(double chi2, double& delta_v, double& v_th,
          std::vector<double>& chi2list, std::vector<double>& dvlist,
          std::vector<double>& vthlist, gsl_rng* rand_ins, int k,
          std::ofstream& ofs, std::ofstream& dfs, Eigen::Matrix2d cov) {
    double sigma_x = std::sqrt(cov(0, 0));
    double sigma_y = std::sqrt(cov(1, 1));
    double cov_xy  = cov(0, 1);
    double rho     = cov_xy / (sigma_x * sigma_y);
    double step_dv, step_vth;

    if (k == 0 || chi2 < chi2list[k - 1]) {
        chi2list[k] = chi2;
        dvlist[k]   = delta_v;
        vthlist[k]  = v_th;
        ofs << dvlist[k] << " " << vthlist[k] << " " << chi2list[k] << std::endl;

        do {
            gsl_ran_bivariate_gaussian(rand_ins, sigma_x, sigma_y, rho, &step_dv, &step_vth);
            delta_v = dvlist[k] + step_dv;
            v_th    = vthlist[k] + step_vth;
        } while (delta_v < 0 || v_th < 0);

        std::cout << "if number: " << 0 << std::endl;
    } else {
        double r        = std::exp(-(chi2 - chi2list[k - 1]) / 2.0);
        double rand_num = gsl_rng_uniform(rand_ins);

        if (r > rand_num) {
            chi2list[k] = chi2;
            dvlist[k]   = delta_v;
            vthlist[k]  = v_th;
            ofs << dvlist[k] << " " << vthlist[k] << " " << chi2list[k] << std::endl;

            do {
                gsl_ran_bivariate_gaussian(rand_ins, sigma_x, sigma_y, rho, &step_dv, &step_vth);
                delta_v = dvlist[k] + step_dv;
                v_th    = vthlist[k] + step_vth;

            } while (delta_v < 0 || v_th < 0);

            std::cout << "if number: " << 1 << std::endl;
        } else {
            chi2list[k] = chi2list[k - 1];
            dvlist[k]   = dvlist[k - 1];
            vthlist[k]  = vthlist[k - 1];
            ofs << dvlist[k] << " " << vthlist[k] << " " << chi2list[k] << std::endl;
            dfs << delta_v << " " << v_th << " " << chi2 << std::endl;

            do {
                gsl_ran_bivariate_gaussian(rand_ins, sigma_x, sigma_y, rho, &step_dv, &step_vth);
                delta_v = dvlist[k - 1] + step_dv;
                v_th    = vthlist[k - 1] + step_vth;
            } while (delta_v < 0 || v_th < 0);

            std::cout << "if number: " << 2 << std::endl;
        }
    }
}
