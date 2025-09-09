import random
import csv
import glob
import os
import torch
import torch.cuda

import SimpleITK as sitk
import numpy as np
import functools

from collections import namedtuple
from util import XYZTuple, xyz_to_irc
from torch.utils.data import Dataset
from disk import getCache

from logconf import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache('segmentation')

CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'is_nodule, has_annotation, is_malignant, diameter_mm, series_uid, center_xyz')

# A single CTScan Instance based on series_uid attribute
class CTScan:
    def __init__(self, series_uid):
        self.series_uid = series_uid

        mhd_path = glob.glob(f"data/subset*/subset*/{series_uid}.mhd")[0] # should return list of one element (the path), like ['1.45.553262.66'], and you just grab the string directly
        ct_scan = sitk.ReadImage(mhd_path) # reads the .mhd (metadata) path file and automatically reads the associated .raw CTScan file
        self.ct_HU_voxels = np.array(sitk.GetArrayFromImage(ct_scan), dtype=np.float32) # CONTAINS ALL THE VOXELS of the CTScan, with each voxel having an HU value. This will be I x 512 x 512

        self.direction = np.array(ct_scan.GetDirection()).reshape(3,3)
        self.origin_xyz = XYZTuple(ct_scan.GetOrigin()[0], ct_scan.GetOrigin()[1], ct_scan.GetOrigin()[2])
        self.voxel_size_xyz = XYZTuple(ct_scan.GetSpacing()[0], ct_scan.GetSpacing()[1], ct_scan.GetSpacing()[2])

        candidateInfoList_series_uid = getCandidateInfoDictionary()[self.series_uid] # Gets all the Candidate Nodules/Non-Nodules from this CTScan (series_uid)
        self.nodule_candidateInfoList = [cand_tup for cand_tup in candidateInfoList_series_uid if cand_tup.is_nodule] # For this series_uid CT Scan, get all Nodule candidate info list tuples

        self.nodule_mask = self.buildAnnotationMask(self.nodule_candidateInfoList) # For the entire Index x 512 x 512 voxel CT scan, builds a mask of True and False indicating at each voxel whether there is Nodule or Non-Nodule
        self.nodule_indexes_z = self.nodule_mask.sum(axis=(1, 2)).nonzero()[0].tolist() # Returns something like [0, 5, 6] representing the Index Axis' that have nodule voxels


    def buildAnnotationMask(self, nodule_candidateInfoList, threshold_HU = -700):
        bounding_box = np.zeros_like(self.ct_HU_voxels, dtype=np.bool_)
        
        for nodule_candidate_tuple in nodule_candidateInfoList:
            nodule_center_irc = xyz_to_irc(nodule_candidate_tuple.center_xyz, self.origin_xyz, self.voxel_size_xyz, self.direction) # Nodule Center in voxel IRC format
            c_index = nodule_center_irc.index
            c_row = nodule_center_irc.row
            c_col = nodule_center_irc.col

            # Index Dimension
            index_rad = 2
            try:
                while self.ct_HU_voxels[c_index - index_rad, c_row, c_col] > threshold_HU and self.ct_HU_voxels[c_index + index_rad, c_row, c_col] > threshold_HU:
                    index_rad += 1
            except IndexError:
                index_rad -= 1

            # Row Dimension
            row_rad = 2
            try:
                while self.ct_HU_voxels[c_index, c_row - row_rad, c_col] > threshold_HU and self.ct_HU_voxels[c_index, c_row + row_rad, c_col] > threshold_HU:
                    row_rad += 1
            except IndexError:
                row_rad -= 1

            # Column Dimension
            col_rad = 2
            try:
                while self.ct_HU_voxels[c_index, c_row, c_col - col_rad] > threshold_HU and self.ct_HU_voxels[c_index, c_row, c_col + col_rad] > threshold_HU:
                    col_rad += 1
            except IndexError:
                col_rad -= 1

            bounding_box[c_index - index_rad: c_index + index_rad + 1, c_row - row_rad: c_row + row_rad + 1, c_col - col_rad: c_col + col_rad + 1] = True

        final_nodule_mask = bounding_box & (self.ct_HU_voxels > threshold_HU) # Some voxels WITHIN bounding box might not be nodule (the RAD variables only identify one axis at a time as it goes in straightline)

        return final_nodule_mask
    

    # Extracts a chunk of the CTScan self.ct_HU_voxels
    def extractChunk(self, center_xyz, width_irc):
        center_irc = xyz_to_irc(center_xyz, self.origin_xyz, self.voxel_size_xyz, self.direction)
        
        slice_list = []
        for axis, coord_val in enumerate(center_irc):
            start_ind = int(round(coord_val - width_irc[axis] / 2))
            end_ind = int(start_ind + width_irc[axis])

            # In case start_ind or end_ind go out of bounds (we don't want IndexError)
            if start_ind < 0:
                start_ind = 0
                end_ind = int(width_irc[axis])
            
            if end_ind > self.ct_HU_voxels.shape[axis]:
                end_ind = self.ct_HU_voxels.shape[axis]
                start_ind = int(end_ind - width_irc[axis])

            slice_list.append(slice(start_ind, end_ind))

        ct_chunk = self.ct_HU_voxels[slice_list[0], slice_list[1], slice_list[2]]
        nodule_mask_chunk = self.nodule_mask[slice_list[0], slice_list[1], slice_list[2]]

        return ct_chunk, nodule_mask_chunk, center_irc


@functools.lru_cache(1)
def getCandidateInfoList():
    mhd_list = glob.glob("data/subset*/subset*/*.mhd")
    series_uid_present_disk = {os.path.split(file)[-1][:-4] for file in mhd_list} # Since we only use a subset, some series_uid in csv files might not be in our disk subsets

    candidateInfoList = []

    # annotations.csv contains all true positive nodules (malignant or benign) across ALL CT scans
    with open('data/annotations_with_malignancy.csv', "r") as f:
        csv_reader = list(csv.reader(f))[1:] # don't include row index
        for row in csv_reader:
            series_uid = row[0] # 1.2.3.4.6426.7474747

            if series_uid not in series_uid_present_disk:
                continue

            annot_center_coords_xyz = tuple([float(c) for c in row[1:4]]) # (-1.75, 35.565, 13.6843)
            diameter_mm = float(row[4])
            is_malignant = {'True': True, 'False': False}[row[5]]

            # is_nodule, has_annotation, is_malignant, nodule diameter, ct scan series_uid, nodule center xyz
            candidateInfoList.append(CandidateInfoTuple(
                True, True, is_malignant, diameter_mm, series_uid, annot_center_coords_xyz))
    
    # candidates.csv will be filtered below to contain non-nodules (negative)
    with open('data/candidates.csv', "r") as f:
        csv_reader = list(csv.reader(f))[1:]
        for row in csv_reader:
            series_uid = row[0] # 1.2.3.4.6426.7474747

            if series_uid not in series_uid_present_disk:
                continue
            
            cand_center_coords_xyz = tuple([float(c) for c in row[1:4]])
            is_nodule = bool(row[4])

            if not is_nodule:
                candidateInfoList.append(CandidateInfoTuple(
                    False, False, False, 0.0, series_uid, cand_center_coords_xyz
                ))
    
    return candidateInfoList


# Stores all Nodule and Non-Nodule Locations from annotations_with_malignancy.csv and candidate.csv by Key (series_uid)
@functools.lru_cache(1)
def getCandidateInfoDictionary():
    candidateInfoList = getCandidateInfoList()
    
    candidateInfoDict = {}
    for candidateInfoTuple in candidateInfoList:
        if candidateInfoTuple.series_uid not in candidateInfoDict:
            candidateInfoDict[candidateInfoTuple.series_uid] = []
        candidateInfoDict[candidateInfoTuple.series_uid].append(candidateInfoTuple)
    
    return candidateInfoDict

# Caches this CTScan Instance by series_uid
@functools.lru_cache(1, typed=True)
def getCT(series_uid):
    return CTScan(series_uid)


@raw_cache.memoize(typed=True)
def extractMemoizedChunk(series_uid, center_xyz, width_irc): # Extracts cached CTScan by series_uid first
    ct = getCT(series_uid)
    ct_chunk, nodule_mask_chunk, center_irc = ct.extractChunk(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk) # We clip values to this range to normalize the range for traininga and reduce noise and outliers
    return ct_chunk, nodule_mask_chunk, center_irc


@raw_cache.memoize(typed=True)
def getCTIndexAxisSize(series_uid):
    ct = getCT(series_uid)
    return int(ct.ct_HU_voxels.shape[0]), ct.nodule_indexes_z # Index Dimension Length and Index Slices that have Nodules


# For Validation. This will contain full 7 x 512 x 512 slices of CT Scans that contain nodules
class Luna2DSegmentationDataset(Dataset):
    def __init__(self, val_stride, is_val_set, context_slices_count=3, fullCTSlices=False):
        self.context_slices_count = context_slices_count
        self.fullCTSlices = fullCTSlices

        self.series_list = sorted(getCandidateInfoDictionary().keys()) # Gets all CT Scans (series_uid as List)
        
        # Series_list will be based on whether this is training or validation set
        if is_val_set:
            self.series_list = self.series_list[::val_stride]
        else: # val_stride > 0
            del self.series_list[::val_stride]
        
        self.slice_list = []
        for series_uid in self.series_list:
            index_dim_size, nodule_indexes_z = getCTIndexAxisSize(series_uid)

            # For each CTScan (series_uid), we only keep slices along index axis where positive nodules exist.
            # The nodule_indexes_z from getCTIndexAxisSize() tells us for the CTScan which specific Index Axis' (Z) have positive nodules.
            # So this will have across all series_uid CTScans, all their index/Z axis slice numbers with nodules.
            # fullCTSlices = True when you want to evaludate on the full CT scan and not just slices with nodules
            # fullCTSlices = False to save computations and focus on nodule slices to balance the skewed dataset with more positives
            # Just set it to False most of the time. Self.slice_list is used for Validation.
            if self.fullCTSlices:
                self.slice_list += [(series_uid, z_slice_ind) for z_slice_ind in range(index_dim_size)]
            else:
                self.slice_list += [(series_uid, z_slice_ind) for z_slice_ind in nodule_indexes_z]

        # For Training. In the Candidate Nodules List (contains potential nodules and also non-nodules), we only take nodules for nodule_info_list. We extract 7x96x96 chunks around nodules
        self.candidate_info_list = getCandidateInfoList()
        self.nodule_info_list = [candidate_tuple for candidate_tuple in self.candidate_info_list if candidate_tuple.is_nodule]


    def __len__(self):
        return len(self.slice_list)


    def __getitem__(self, ndx):
        series_uid, ct_scan_indexAxisSlice = self.slice_list[ndx % len(self.slice_list)] # This % prevents overflow IndexError. This effectively makes this an infinite Dataset list
        return self.getitem_full_ct_slice(series_uid, ct_scan_indexAxisSlice)


    # For a CT Scan, gets 7 FULL slices (based on context_slices_count) along index/Z axis. So, this will be 7 x 512 x 512
    def getitem_full_ct_slice(self, series_uid, ct_scan_indexAxisSlice):
        ct = getCT(series_uid)
        ct_slices = torch.zeros((self.context_slices_count * 2 + 1, 512, 512)) # For each slice index axis slice you pass in, it will take 2 slices behind and 2 slices ahead

        start_index = ct_scan_indexAxisSlice - self.context_slices_count
        end_index = ct_scan_indexAxisSlice + self.context_slices_count + 1

        for i, slice_ind in enumerate(range(start_index, end_index)):
            slice_ind = max(0, slice_ind) # Prevent slice_ind along index/z-axis out of bounds
            slice_ind = min(ct.ct_HU_voxels.shape[0] - 1, slice_ind)
            ct_slices[i] = torch.from_numpy(ct.ct_HU_voxels[slice_ind].astype(np.float32)) # Each of these is 512 x 512

        ct_slices.clamp_(-1000, 1000)

        # Extracts ground truth label for this SINGLE CENTER SLICE (Not including 2 below and 2 above like in input label)
        # Uses unsqueeze to get into shape [1, 512, 512]
        ground_truth_mask = torch.from_numpy(ct.nodule_mask[ct_scan_indexAxisSlice]).unsqueeze(0)

        # ct_slices is like the input ([7, 512, 512]) with voxels and ground_truth_mask is the correct ground truth label ([1, 512, 512]) with boolean masks indicating nodules or non nodule areas
        return ct_slices, ground_truth_mask, ct.series_uid, ct_scan_indexAxisSlice


# This is used for Training. It returns random 7x64x64 crops and 1x64x64 ground truth labels with nodules
class TrainingLuna2DSegmentationDataset(Luna2DSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __len__(self):
        return 300000


    def __getitem__(self, ndx):
        nodule_candidate_info_tup = self.nodule_info_list[ndx % len(self.nodule_info_list)] # This % prevents IndexError
        return self.getitem_crop_ct_slice(nodule_candidate_info_tup)
    

    # We first extract a 7x96x96 chunk from the CTScan based on the center coordinates of the inputted tuple which contains a nodule
    # Then we extract a random 7x64x64 chunk from it. These are voxels remember.
    def getitem_crop_ct_slice(self, nodule_candidate_info_tup):
        ct_chunk, nodule_mask_chunk, center_irc = extractMemoizedChunk(
            nodule_candidate_info_tup.series_uid, nodule_candidate_info_tup.center_xyz, (7, 96, 96))
        
        # Was a 7x96x96 boolean mask. We only want the center slice for our ground truth labels
        # Now becomes 1x96x96 boolean mask
        nodule_mask_chunk = nodule_mask_chunk[3:4] 

        random_row_offset = random.randrange(0, 32)
        random_col_offset = random.randrange(0, 32)

        # Extracting random 7x64x64 chunk from the 7x96x96. This acts as a form of data augmentation
        cropped_ct_chunk = torch.from_numpy(ct_chunk[:, random_row_offset: random_row_offset + 64, random_col_offset: random_col_offset + 64]).to(torch.float32)
        
        # Similarly here. Extract 1x64x64 from 1x96x96 boolean mask
        cropped_label_mask_chunk = torch.from_numpy(nodule_mask_chunk[:, random_row_offset: random_row_offset + 64, random_col_offset: random_col_offset + 64]).to(torch.long)
        
        # cropped_ct_chunk is shape [7, 64, 64] and cropped_label_mask_chunk is shape [1, 64, 64]
        return cropped_ct_chunk, cropped_label_mask_chunk, nodule_candidate_info_tup.series_uid, center_irc.index


    def shuffle_samples(self):
        random.shuffle(self.nodule_info_list)
