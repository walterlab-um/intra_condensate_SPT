


# Intra-Condensate SPT

This repository contains Python scripts to analyze RNA and protein single molecule tracking (SMT) within biomolecular condensates, namely **intra-condensate SMT**.

Please consider to cite: https://doi.org/10.1101/2024.04.01.587651

The article exemplify the use of intra-condensate SMT in optimally tethered Fused-in-Sarcoma (FUS) condensates *in vitro*, where this model RNA-binding  protein was purified in its full-length tag-free form to mimic the native ribonucleoprotein (RNP) granule in cells.

## Installation
 - All prerequisites to run the scripts in this repository are specified in conda_environment-spt.yml
 - A *spt* conda environment can be installed from the yml file by:

    conda env create -f environment.yml

- Activate the *spt* conda environment before running any scripts

    conda activate spt

## A basic pipeline to analyze diffusion

 1. [optional] If dual-color SPT is needed, channel registration should be performed to align two videos from each channel, and scripts in folder "Camera_Registration" can be used.
 2. Filter out non-single molecule signals in the SMT videos using "bandpass_filter.py"
 3. Extract SMT trajectories using TrackMate (https://imagej.net/plugins/trackmate/)
 4. Export trajectories as csv files using "Export Tracks" or "Export Spots" function in TrackMate
 5. Reformat the csv file using "Reformat-TrackMate-tracks.py" or "Reformat-TrackMate-spots.py"
 6. Calculate all classical diffusion metrics from every single trajectory and save as an all-in-one (AIO) format csv file using "calculate_SPT_all_in_one.py". The AIO format includes:
	 - trackID
	 - list of time (s)
	 - list of x positions (pixel)
	 - list of y positions (pixel)
	 - total trajectory length (number of steps)
	 - displacement (nm)
	 - mean step size (nm)
	 - maximum distance between any two positions in a trajectory (nm)
	 - centroid x location of the trajectory (pixel)
	 - centroid y location of the trajectory (pixel)
	 - maximum mean fluorescence intensity of each spot in a trajectory
	 - list of mean square displacement (MSD, um^2)
	 - list of lag time (tau, second)
	 - Fitting of MSD-tau in linear scale using an optimized formula (doi:10.1103/PhysRevE.85.061916):
		 - slope
		 - R2
		 - localization error sigma (um)
		 - apparent diffusion coefficient D (um^2/s)
		 - log10D
	- Fitting of MSD-tau in log-log scale using a classical formula:
		- R2
		- log10D
		- anomalous diffusion component alpha
	- list of all angles between steps
	- histogram of angle distribution within a single trajectory (which is not reliable most of the time due to short trajectory length and thus limited datapoints to yield a meaningful histogram)
 7. Pooling all datasets from different dates of experiments under the same condition using "concat_SPT_AIO_files.py"
 8. Classification of different diffusion types and the distribution plots of all above diffusion metrics can be done with your own scripts or plotting program. Some examples can be found in folder "plot_paper_figure" or folder "plot_SPT_metrices".
 9. To analyze the normal diffusion fraction of trajectories and characterize diffusive states within the fraction, Spot-ON (doi:10.7554/eLife.33125) or saSPT (doi:10.7554/eLife.70169) could be used and scripts "calculate_SpotON_pooled_one_condition.py" and "calculate_saSPT_pooled_one_condition.py" feed the standard AIO format to each program.
 10. To analyze the spatial correlation between single molecule locations, Pair Correlation Function (a.k.a., Radial Distribution Function) could be calculated with careful boundary correction and normalization using scripts "calculate_Pair_Correlation_per_condensate_perLoc.py", "calculate_Pair_Correlation_per_condensate_perTrack.py", or "calculate_Pair_Correlation_per_condensate_perDomain.py". 
 11. Points Accumulation for Imaging in Nanoscale Topography (PAINT) reconstructions for SMT-PAINT analysis could be performed using script "ALEX_spots2PAINT_split_condensates.py" for all condensates at once from each microscope field of view (FOV). Moreover, a time-lapse version of SMT-PAINT to show dynamics of underlying structures that confine diffusion can be performed using "ALEX_spots2PAINT_timelapse_single_condensate.py"
 12. To analyze the spatial distribution of trajectories within a condensate, "calculate_normalized_distance2center.py" could be used.


