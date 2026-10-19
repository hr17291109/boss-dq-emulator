import numpy as np
import pandas as pd
import cosmo_inference as ci
from getdist import plots
from importlib import reload
reload(ci)

def load_design(region, zbin):
    df = pd.DataFrame(ci.combine_data(region, zbin))
    ghc = pd.read_csv(f"design/ghc_{zbin}Z_{region}_marge.csv")
    return df.merge(ghc, on="set_id", validate="one_to_one")

#NHdata = ci.combine_data('NGC', 'HIGH')
#df = pd.DataFrame(NHdata)
df = load_design('NGC', 'HIGH')
Nsteps = 20000
nburnin = 4000
samples1 = ci.GPmcmc(df, Nsteps, nburnin, kernel="rbf", y_col="chi_gauss")
g1 = plots.get_subplot_plotter(subplot_size=2.5)
g1.triangle_plot(
    samples1,
    filled=True,
    contour_colors=["steelblue"],
    title_limit=1,
)
g1.fig.suptitle("Posterior Distribution (GP + MCMC)", fontsize=14, y=1.01)
g1.fig.savefig("gp_output/NGC_HIGHZ_RBFW_posterior_triangle_20000_marge.png", dpi=150, bbox_inches="tight")
#g1.fig.savefig("NGC_HIGHZ_Matern_posterior_triangle_100000.png", dpi=150, bbox_inches="tight")

#NLdata = ci.combine_data('NGC', 'LOW')
#df2 = pd.DataFrame(NLdata)
df2 = load_design('NGC', 'LOW')
Nsteps = 20000
nburnin = 4000
samples2 = ci.GPmcmc(df2, Nsteps, nburnin, kernel="rbf", y_col="chi_gauss")
g2 = plots.get_subplot_plotter(subplot_size=2.5)
g2.triangle_plot(
    samples2,
    filled=True,
    contour_colors=["steelblue"],
    title_limit=1,
)
g2.fig.suptitle("Posterior Distribution (GP + MCMC)", fontsize=14, y=1.01)
g2.fig.savefig("gp_output/NGC_LOWZ_RBFW_posterior_triangle_20000_marge.png", dpi=150, bbox_inches="tight")
#g2.fig.savefig("gp_output/NGC_LOWZ_Matern_posterior_triangle_100000.png", dpi=150, bbox_inches="tight")

#SHdata = ci.combine_data('SGC', 'HIGH')
#df3 = pd.DataFrame(SHdata)
df3 = load_design('SGC', 'HIGH')
Nsteps = 20000
nburnin = 4000
samples3 = ci.GPmcmc(df3, Nsteps, nburnin, kernel="rbf", y_col="chi_gauss")
g3 = plots.get_subplot_plotter(subplot_size=2.5)
g3.triangle_plot(
    samples3,
    filled=True,
    contour_colors=["steelblue"],
    title_limit=1,
)
g3.fig.suptitle("Posterior Distribution (GP + MCMC)", fontsize=14, y=1.01)
g3.fig.savefig("gp_output/SGC_HIGHZ_RBFW_posterior_triangle_20000_marge.png", dpi=150, bbox_inches="tight")
#g3.fig.savefig("gp_output/SGC_HIGHZ_Matern_posterior_triangle_100000.png", dpi=150, bbox_inches="tight")

#SLdata = ci.combine_data('SGC', 'LOW')
#df4 = pd.DataFrame(SLdata)
df4 = load_design('SGC', 'LOW')
Nsteps = 20000
nburnin = 4000
samples4 = ci.GPmcmc(df4, Nsteps, nburnin, kernel="rbf", y_col="chi_gauss")
g4 = plots.get_subplot_plotter(subplot_size=2.5)
g4.triangle_plot(
    samples4,
    filled=True,
    contour_colors=["steelblue"],
    title_limit=1,
)
g4.fig.suptitle("Posterior Distribution (GP + MCMC)", fontsize=14, y=1.01)
g4.fig.savefig("gp_output/SGC_LOWZ_RBFW_posterior_triangle_20000_marge.png", dpi=150, bbox_inches="tight")
#g4.fig.savefig("gp_output/SGC_LOWZ_Matern_posterior_triangle_100000.png", dpi=150, bbox_inches="tight")

samples_prior = ci.GPmcmc(df, Nsteps, nburnin, kernel="rbf", prior_only=True, y_col="chi_gauss")
g = plots.get_subplot_plotter(subplot_size=2.5)
g.triangle_plot(
    [samples1, samples2, samples3, samples4, samples_prior],
    filled=[True, True, True, True, False],
    contour_colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "gray"],
    legend_labels=["NGC HIGH", "NGC LOW", "SGC HIGH", "SGC LOW", "prior only"],
    title_limit=1,
)
g.fig.savefig("gp_output/All_Data_RBFW_posterior_triangle_20000_marge_sample.png", dpi=150, bbox_inches="tight")

g = plots.get_subplot_plotter(subplot_size=2.5)
g.triangle_plot(
    [samples1, samples2, samples3, samples4],
    filled=True,
    contour_colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    legend_labels=["NGC HIGH", "NGC LOW", "SGC HIGH", "SGC LOW"],
    title_limit=1,
)
g.fig.suptitle("Posterior Distribution (GP + MCMC) - All Data", fontsize=14, y=1.01)
g.fig.savefig("gp_output/All_Data_RBFW_posterior_triangle_20000_marge.png", dpi=150, bbox_inches="tight")
#g.fig.savefig("gp_output/All_Data_Matern_posterior_triangle_100000.png", dpi=150, bbox_inches="tight")