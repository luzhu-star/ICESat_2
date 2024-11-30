# coding: utf-8

import argparse
import pandas as pd
import os
import json
import scipy
import skimage
import time
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import shapely

from sliderule import icesat2, earthdata
from datetime import datetime, timedelta

url = "slideruleearth.io"
icesat2.init(url, verbose=False)
asset = "icesat2"




# modify the following lines for custom tracks identified from Open Altimetry

parser = argparse.ArgumentParser(description='Preprocess ICESat-2 data.')
parser.add_argument('--target_date', type=str, help='Target date (YYYY-MM-DD)')
parser.add_argument('--target_rgt', type=int, help='Target reference ground track (RGT)')
parser.add_argument('--bbox_coords', type=str, help='Bounding box coordinates')
args = parser.parse_args()

target_date = args.target_date
target_rgt = args.target_rgt
bbox_coords = json.loads(args.bbox_coords)
beam_types = ['strong', 'weak']
track_nums = [1, 2, 3]

# 打印调试信息，查看解析后的类型和内容
print(f"Type of bbox_coords: {type(bbox_coords)}")
print(f"Bounding box coordinates: {bbox_coords}")
# Reformatting bounding box for query purposes
start_time = time.time()
poly = [{'lat': point[1], 'lon': point[0]} for point in bbox_coords]

# sliderule parameters for querying photon data
parms = {
    
    "len": 20,
    "pass_invalid": True, # include segments which dont pass all checks
    "cnf": -2, # returns all photons
    
    #  try other signal confidence scores! SRT_LAND, SRT_OCEAN, SRT_SEA_ICE, SRT_LAND_ICE, SRT_INLAND_WATER
    "srt": icesat2.SRT_LAND, 
    
    # density based signal finding
    "yapc": dict(knn=0, win_h=6, win_x=11, min_ph=4, score=0), 
    
    # extra fields from ATL03 gtxx/geolocation or gtxx/geophys_corr channels
    "atl03_geo_fields": ["ref_azimuth", "ref_elev", "geoid"],
    
    # extra fields via ATL03 "gtxx/heights" (see atl03 data dictionary)
    "atl03_ph_fields": [], # 
    
    # add query polygon
    "poly": poly,

}

# Formatting datetime query for demo purposes - 1 day buffer
day_window = 1 # days on either side to query
target_dt = datetime.strptime(target_date, '%Y-%m-%d')
time_start = target_dt - timedelta(days=day_window)
time_end = target_dt + timedelta(days=day_window)

# find granule for each region of interest
granules_list = earthdata.cmr(short_name='ATL03', 
                              polygon=poly, 
                              time_start=time_start.strftime('%Y-%m-%d'), 
                              time_end=time_end.strftime('%Y-%m-%d'), 
                              version='006')


# create an empty geodataframe
df_0 = icesat2.atl03sp(parms, resources=granules_list)
end_time = time.time()
df_0.head()

def reduce_dataframe(gdf, RGT=None, GT=None, track=None, pair=None, cycle=None, beam='', crs=4326): 
    # convert coordinate reference system
    df = gdf.to_crs(crs)
    
    # reduce to reference ground track
    if RGT is not None:
        df = df[df["rgt"] == RGT]
        
    # reduce to ground track (gt[123][lr]), track ([123]), or pair (l=0, r=1) 
    gtlookup = {icesat2.GT1L: 1, icesat2.GT1R: 1, icesat2.GT2L: 2, icesat2.GT2R: 2, icesat2.GT3L: 3, icesat2.GT3R: 3}
    pairlookup = {icesat2.GT1L: 0, icesat2.GT1R: 1, icesat2.GT2L: 0, icesat2.GT2R: 1, icesat2.GT3L: 0, icesat2.GT3R: 1}
    
    if GT is not None:
        df = df[(df["track"] == gtlookup[GT]) & (df["pair"] == pairlookup[GT])]
    if track is not None:
        df = df[df["track"] == track]
    if pair is not None:
        df = df[df["pair"] == pair]
        
    # reduce to weak or strong beams if specified
    if beam is not None:
        if (beam == 'strong'):
            df = df[df['sc_orient'] == df['pair']]
        elif (beam == 'weak'):
            df = df[df['sc_orient'] != df['pair']]
        
    # reduce to cycle
    if cycle is not None:
        df = df[df["cycle"] == cycle]
        
    # otherwise, return both beams
    return df

df_list = []

for track_num in track_nums:
    for beam_type in beam_types: 
        # Reduce dataframe based on the specified track number and beam type, and transform to EPSG:4326
        df = reduce_dataframe(df_0, 
                            RGT=target_rgt, 
                            track=track_num, 
                            beam=beam_type, 
                            crs='EPSG:4326')
        
        # Add a column for along-track distance
        df['along_track_meters'] = df['segment_dist'] + df['x_atc'] - np.min(df['segment_dist'])
        
        # Compute orthometric heights using the onboard geoid model (EGM08)
        df['height_ortho'] = df['height'] - df['geoid']
        
        # Add columns for track number and beam type
        df['track_num'] = track_num
        df['beam_type'] = beam_type

        # Elevation range for plotting
        ylim = [-25, 10]

        # Filter very shallow / deep data to refine bathymetry
        min_bathy_depth = 0.5 
        max_bathy_depth = 30

        # Calculate Otsu threshold
        yapc_threshold = skimage.filters.threshold_otsu(np.array(df['yapc_score']))

        # Get photons with a minimum YAPC signal score 
        signal_photons_yapc = df['yapc_score'] > yapc_threshold

        # Define histogram bin edges for height data
        bin_edges_1 = np.arange(-50, 50, 0.1)
        hist_1 = plt.hist(df['height_ortho'][signal_photons_yapc], bins=bin_edges_1)[0]

        # Determine trimming height above which photons will be removed (buffer for surface width / waves)
        trim_photons_above = bin_edges_1[np.argmax(hist_1)] + 1

        # Filter to include only the signal photons within the height threshold
        water_zone_photons = df['height_ortho'] < trim_photons_above
        df_cleaned = df.loc[water_zone_photons & signal_photons_yapc]

        # Adaptive resolution: adjust resolution based on photon density
        photon_density = np.histogram(df_cleaned['height_ortho'], bins=bin_edges_1)[0]
        res_z = 0.1 if np.max(photon_density) > 100 else 0.2  # Adjust vertical resolution based on photon density
        res_at = 20  # Along-track resolution

        # Define along-track bin sizing
        bin_edges_z = np.arange(ylim[0], ylim[1], res_z)
        range_at = [0, df_cleaned.along_track_meters.max() + res_at]
        bin_edges_at = np.arange(range_at[0], range_at[1], res_at)

        bin_centers_z = bin_edges_z[:-1] + np.diff(bin_edges_z)[0] / 2
        bin_centers_at = bin_edges_at[:-1] + np.diff(bin_edges_at)[0] / 2

        # **Adaptive filtering**: Compute local variance and dynamically adjust filter parameters
        window_size = 50  # Data window size for local variance calculation
        local_variance = df_cleaned['height_ortho'].rolling(window=window_size, min_periods=1).var()

        # Adjust Gaussian filter standard deviation based on local variance
        sigma = np.sqrt(local_variance)  

        # Dynamically adjust filter parameters based on local variance
        filtered_data = np.zeros_like(df_cleaned['height_ortho'])
        for i in range(len(df_cleaned)):
            window_sigma = sigma.iloc[i] if i < len(sigma) else 0.5  # Adjust based on local variance
            # Apply Gaussian filter to data
            filtered_data[i] = scipy.ndimage.gaussian_filter1d(df_cleaned['height_ortho'][i], window_sigma)

        # Create a 2D histogram for plotting
        hist_2 = np.histogram2d(df_cleaned.along_track_meters, df_cleaned.height_ortho,
                                bins = [bin_edges_at, bin_edges_z])[0]

        hist_2 = hist_2.T  # Transpose for more intuitive orientation

        # Apply adaptive Gaussian filtering to 2D histogram
        for i in range(hist_2.shape[0]):
            window_sigma = np.sqrt(local_variance.iloc[i])  # Use local variance as standard deviation
            hist_2[i, :] = scipy.ndimage.gaussian_filter1d(hist_2[i, :], window_sigma)

        n_inspection_points = 6  # Automatically choose equally spaced locations to inspect

        # Calculate inspection points along the track
        inspect_along_track_locations = np.linspace(bin_centers_at[0], bin_centers_at[-1], 
                                                    n_inspection_points + 2,  # Adjust for edges
                                                    dtype=np.int64)[1:-1] 

        # Round inspection points to the nearest 100 meters
        inspect_along_track_locations = np.round(inspect_along_track_locations, -3)

        # Set kernel size for smoothing (in meters)
        kernel_size_meters = 0.5
        kernel_size = np.int64(kernel_size_meters / res_z)

        # Smooth waveform data at each inspection point
        for i, location in enumerate(inspect_along_track_locations):
            # Extract waveform data at the current location
            waveform = hist_2[:, np.int64(location / res_at)]
            # Apply convolution to smooth the waveform data
            waveform_smoothed = np.convolve(waveform, np.ones(kernel_size) / kernel_size, mode='same')



        def extract_bathymetry(waveform, min_height=.25, min_prominence=.25):
            """
            Extracts bathymetry information from a histogram of photon heights and simple assumptions of topographic features.

            Parameters:
            waveform (1D array-like): Input histogram of photon heights.
            min_height (float, optional): Minimum height of peaks to be considered for bathymetry or water surface.
                                        Defaults to 0.25.
            min_prominence (float, optional): Minimum prominence of peaks to be considered for bathymetry or water surface.
                                            Defaults to 0.25.

            Returns:
            water_surface_peak (int or None): Index of the peak representing the water surface, or None if no valid peaks are found.
            bathy_peak (int or None): Index of the peak representing the subsurface bathymetry, or None if no valid peaks are found.
            """
            peaks, peak_info_dict = scipy.signal.find_peaks(waveform, 
                                                            height=min_height, 
                                                            prominence=min_prominence)

            if len(peaks) == 0:
                return None, None

            # topmost return is the water surface
            water_surface_peak = peaks[-1]

            # all other peaks are possible bathymetry
            bathy_candidate_peaks = peaks[:-1]
            bathy_candidate_peak_heights = peak_info_dict['peak_heights'][:-1]
            bathy_candidate_peak_prominences = peak_info_dict['prominences'][:-1]

            if len(bathy_candidate_peaks) == 0:
                return water_surface_peak, None

            # get the most prominent subsurface peak
            bathy_peak = bathy_candidate_peaks[np.argmax(bathy_candidate_peak_prominences)]

            return water_surface_peak, bathy_peak


        def photon_refraction(W, Z, ref_az, ref_el, n1=1.00029, n2=1.34116):
            """
            Refraction correction for photon depth data based on Parrish et al. (2019).

            Parameters:
            W : float, or nx1 array of float
                Elevation of the water surface.
            Z : nx1 array of float
                Elevation of seabed photon data (use geoid heights).
            ref_az : nx1 array of float
                Photon azimuth data.
            ref_el : nx1 array of float
                Photon elevation data.
            n1 : float, optional
                Refractive index of air (default 1.00029).
            n2 : float, optional
                Refractive index of water (default 1.34116).

            Returns:
            dE : nx1 array of float
                Easting offset of seabed photons.
            dN : nx1 array of float
                Northing offset of seabed photons.
            dZ : nx1 array of float
                Vertical offset of seabed photons.
            """
            # compute uncorrected depths
            D = W - Z
            H = 496  # mean orbital altitude of IS2, km
            Re = 6371  # mean radius of Earth, km

            # angle of incidence
            theta_1_ = (np.pi / 2) - ref_el
            theta_1 = theta_1_  # ignore curvature correction here

            # angle of refraction
            theta_2 = np.arcsin(n1 * np.sin(theta_1) / n2)

            phi = theta_1 - theta_2

            # uncorrected slant range
            S = D / np.cos(theta_1)

            # corrected slant range
            R = S * n1 / n2

            P = np.sqrt(R**2 + S**2 - 2*R*S*np.cos(theta_1 - theta_2))

            gamma = (np.pi / 2) - theta_1

            alpha = np.arcsin(R * np.sin(phi) / P)

            beta = gamma - alpha

            # cross-track offset
            dY = P * np.cos(beta)

            # vertical offset
            dZ = P * np.sin(beta)

            kappa = ref_az

            # UTM offsets
            dE = dY * np.sin(kappa)
            dN = dY * np.cos(kappa)

            return dE, dN, dZ


        # Processing Bathymetry Data
        water_surface = np.nan * np.ones_like(bin_centers_at)
        bathymetry_uncorrected = np.nan * np.ones_like(bin_centers_at)
        bathymetry_corrected = np.nan * np.ones_like(bin_centers_at)
        bin_centroid = np.empty_like(bin_centers_at, dtype=shapely.Point)
        df_cleaned['dz_refraction'] = 0

        for i_at in tqdm(range(len(bin_centers_at))):
            waveform_i = hist_2[:, i_at]

            # 1. Estimate Water/Seafloor Surfaces
            surface_idx, bathy_idx = extract_bathymetry(waveform_i, min_height=0.1, min_prominence=0.1)

            if surface_idx is None:
                continue

            water_surface_z = bin_centers_z[surface_idx]
            water_surface[i_at] = water_surface_z

            # 2. Refraction Correct Photons
            ph_refr_i = (df_cleaned.along_track_meters.values >= bin_centers_at[i_at] - res_at / 2) & \
                        (df_cleaned.along_track_meters.values <= bin_centers_at[i_at] + res_at / 2) & \
                        (df_cleaned.height_ortho.values <= water_surface_z)

            if np.sum(ph_refr_i) == 0:
                continue

            z_ph_i = df_cleaned.height_ortho.values[ph_refr_i]
            ref_az_ph_i = df_cleaned.ref_azimuth.values[ph_refr_i]
            ref_elev_ph_i = df_cleaned.ref_elev.values[ph_refr_i]

            # Compute refraction corrections for all subsurface photons
            _, _, dz_ph_i = photon_refraction(water_surface_z, z_ph_i, ref_az_ph_i, ref_elev_ph_i)

            df_cleaned.loc[ph_refr_i, 'dz_refraction'] = dz_ph_i

            # Corrected Bathymetry
            if bathy_idx is None:
                continue

            bathymetry_uncorrected_i = bin_centers_z[bathy_idx]
            ref_az_bin_i = np.mean(ref_az_ph_i)
            ref_elev_bin_i = np.mean(ref_elev_ph_i)

            _, _, dz_bin_i = photon_refraction(water_surface_z, bathymetry_uncorrected_i, ref_az_bin_i, ref_elev_bin_i)

            bathymetry_corrected_i = bathymetry_uncorrected_i + dz_bin_i

            if (water_surface_z - bathymetry_corrected_i) < min_bathy_depth or (water_surface_z - bathymetry_corrected_i) > max_bathy_depth:
                continue

            bathymetry_uncorrected[i_at] = bathymetry_uncorrected_i
            bathymetry_corrected[i_at] = bathymetry_corrected_i

            # Geolocate this bin
            bin_centroid[i_at] = df_cleaned.loc[ph_refr_i, 'geometry'].unary_union.centroid


        good_data = ~np.isnan(bathymetry_corrected)

        # Merge water surface and corrected bathymetry depth into a single GeoDataFrame
        df_out = gpd.GeoDataFrame(np.vstack([water_surface[good_data], 
                                            bathymetry_corrected[good_data]]).T, 
                                columns=['water_surface_z', 'bathymetry_corr_z'], 
                                geometry=bin_centroid[good_data], 
                                crs=df_cleaned.crs)

        # Compute the depth and add it to the DataFrame
        df_out['depth'] = df_out.water_surface_z - df_out.bathymetry_corr_z

        # Append the result to the list of DataFrames
        df_list.append(df_out)

        # Merge all data into one DataFrame
        df_all = pd.concat(df_list, ignore_index=True)


# Get the directory path of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(current_directory, 'Data')

# Get the unique granule filename and generate the output filename
granule_name = granules_list[0]
base_name = os.path.splitext(granule_name)[0]  # Remove the file extension

output_xlsx = os.path.join(output_directory, f'{base_name}.xlsx')

# Create the output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

# Save the DataFrame as an Excel file
df_all.to_excel(output_xlsx, index=False)

# Print the CSV format string of the DataFrame
print(df_all.to_csv(index=False))

# Extract longitude and latitude into new columns
df_all['longitude'] = df_all.geometry.x
df_all['latitude'] = df_all.geometry.y

# Create a new DataFrame containing only longitude, latitude, and depth information
df_geo = df_all[['longitude', 'latitude', 'depth']]

# Convert to a GeoDataFrame and set the CRS (Coordinate Reference System)
gdf = gpd.GeoDataFrame(df_geo, geometry=gpd.points_from_xy(df_geo.longitude, df_geo.latitude), crs=df_cleaned.crs)

# Define the output path for the Shapefile
output_shp = os.path.join(output_directory, f'{base_name}.shp')

# Save the GeoDataFrame as a Shapefile
gdf.to_file(output_shp)


