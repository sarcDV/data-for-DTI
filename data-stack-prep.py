import os
import nibabel as nib
import pandas as pd
import h5py as h5
import numpy as np
import itertools
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')
# ----------------------------------
def normalize_slice_by_slice(input_vol_):
    vol_ = np.zeros((input_vol_.shape))
    for ii in range(0,vol_.shape[2]):
        tmp_ = input_vol_[:,:,ii]/input_vol_[:,:,ii].max()
        where_are_nan = np.isnan(tmp_)
        tmp_[where_are_nan] = 0
        vol_[:,:,ii] = tmp_
    
    return vol_ 
# ----------------------------------
    
table_input = pd.read_csv('list-input-info.csv', header=None, sep=' ')
# table_in_mask = pd.read_csv('list-input-mask-info.csv', header=None, sep=' ')
table_dwi = pd.read_csv('list-out-dwi-info.csv', header=None, sep=' ')
table_tensor = pd.read_csv('list-out-tensor-info.csv', header=None, sep=' ')

input_ = pd.DataFrame(table_input.values, columns =["subID","dim1","dim2","dim3","dim4"])
# mask_ = pd.DataFrame(table_in_mask.values, columns =["subID","dim1","dim2","dim3","dim4"])
dwi_ = pd.DataFrame(table_dwi.values, columns =["subID","dim1","dim2","dim3","dim4"])
tensor_ = pd.DataFrame(table_tensor.values, columns =["subID","dim1","dim2","dim3","dim4"])

### split list ####
## train/test 85/15
fact_ = 0.85
train_input_ = input_[0:int(len(input_)*fact_ )]
test_input_ = input_[int(len(input_)*fact_ ):]

# train_mask_ = mask_[0:int(len(mask_)*fact_ )]
# test_mask_ = mask_[int(len(mask_)*fact_ ):]

train_dwi_ = dwi_[0:int(len(dwi_)*fact_ )]
test_dwi_ = dwi_[int(len(dwi_)*fact_ ):]

train_tensor_ = tensor_[0:int(len(tensor_)*fact_ )]
test_tensor_ = tensor_[int(len(tensor_)*fact_ ):]

# -------------------------------------------------
## Input
tmp_train_input = np.zeros((128,128,1))
for row in train_input_.iterrows():
	print(row[1]['subID'])
	img = nib.load('./Input/'+row[1]["subID"]).get_fdata()
	img = normalize_slice_by_slice(img)
	tmp_train_input = np.concatenate((tmp_train_input, img), axis=2)
	# print(img.shape)

tmp_test_input = np.zeros((128,128,1))
for row in test_input_.iterrows():
	print(row[1]['subID'])
	img = nib.load('./Input/'+row[1]["subID"]).get_fdata()
	img = normalize_slice_by_slice(img)
	tmp_test_input = np.concatenate((tmp_test_input, img), axis=2)

tmp_train_input = tmp_train_input[:,:,1:]
tmp_test_input = tmp_test_input[:,:,1:]

slices_train = train_input_['dim3'].sum()
slices_test = test_input_['dim3'].sum()

print(slices_train, slices_test)
print(tmp_train_input.shape, tmp_test_input.shape)


img_train_input = nib.Nifti1Image(tmp_train_input, np.eye(4))
nib.save(img_train_input, 'train-4D-input.nii.gz')

img_test_input = nib.Nifti1Image(tmp_test_input, np.eye(4))
nib.save(img_test_input, 'test-4D-input.nii.gz')

# -------------------------------------------------
## Mask
"""
tmp_train_mask = np.zeros((128,128,1))
for row in train_mask_.iterrows():
	print(row[1]['subID'])
	img = nib.load('./Mask/'+row[1]["subID"]).get_fdata()
	img = normalize_slice_by_slice(img)
	tmp_train_mask = np.concatenate((tmp_train_mask, img), axis=2)
	# print(img.shape)

tmp_test_mask = np.zeros((128,128,1))
for row in test_mask_.iterrows():
	print(row[1]['subID'])
	img = nib.load('./Mask/'+row[1]["subID"]).get_fdata()
	img = normalize_slice_by_slice(img)
	tmp_test_mask = np.concatenate((tmp_test_mask, img), axis=2)

tmp_train_mask = tmp_train_mask[:,:,1:]
tmp_test_mask = tmp_test_mask[:,:,1:]

slices_train = train_mask_['dim3'].sum()
slices_test = test_mask_['dim3'].sum()

print(slices_train, slices_test)
print(tmp_train_mask.shape, tmp_test_mask.shape)


img_train_mask = nib.Nifti1Image(tmp_train_mask, np.eye(4))
nib.save(img_train_mask, 'train-4D-mask.nii.gz')

img_test_mask = nib.Nifti1Image(tmp_test_mask, np.eye(4))
nib.save(img_test_mask, 'test-4D-mask.nii.gz')

"""
# -------------------------------------------------
tmp_train_dwi = np.zeros((128,128,1))
for row in train_dwi_.iterrows():
	print(row[1]['subID'])
	img = nib.load('./Out-dwi/'+row[1]["subID"]).get_fdata()
	img = normalize_slice_by_slice(img)
	tmp_train_dwi = np.concatenate((tmp_train_dwi, img), axis=2)

tmp_test_dwi = np.zeros((128,128,1))
for row in test_dwi_.iterrows():
	print(row[1]['subID'])
	img = nib.load('./Out-dwi/'+row[1]["subID"]).get_fdata()
	img = normalize_slice_by_slice(img)
	tmp_test_dwi = np.concatenate((tmp_test_dwi, img), axis=2)

tmp_train_dwi = tmp_train_dwi[:,:,1:]
tmp_test_dwi = tmp_test_dwi[:,:,1:]

slices_train = train_dwi_['dim3'].sum()
slices_test = test_dwi_['dim3'].sum()

print(slices_train, slices_test)
print(tmp_train_dwi.shape, tmp_test_dwi.shape)


img_train_dwi = nib.Nifti1Image(tmp_train_dwi, np.eye(4))
nib.save(img_train_dwi, 'train-4D-dwi.nii.gz')

img_test_dwi = nib.Nifti1Image(tmp_test_dwi, np.eye(4))
nib.save(img_test_dwi, 'test-4D-dwi.nii.gz')

# -------------------------------------------------
tmp_train_tensor = np.zeros((128,128,1,6))
for row in train_tensor_.iterrows():
	print(row[1]['subID'])
	img = nib.load('./Out-tensor/'+row[1]["subID"]).get_fdata()
	tmp_train_tensor = np.concatenate((tmp_train_tensor, img), axis=2)

tmp_test_tensor = np.zeros((128,128,1,6))
for row in test_tensor_.iterrows():
	print(row[1]['subID'])
	img = nib.load('./Out-tensor/'+row[1]["subID"]).get_fdata()
	tmp_test_tensor = np.concatenate((tmp_test_tensor, img), axis=2)

tmp_train_tensor = tmp_train_tensor[:,:,1:,:]
tmp_test_tensor = tmp_test_tensor[:,:,1:,:]

slices_train = train_tensor_['dim3'].sum()
slices_test = test_tensor_['dim3'].sum()

print(slices_train, slices_test)
print(tmp_train_tensor.shape, tmp_test_tensor.shape)


img_train_tensor = nib.Nifti1Image(tmp_train_tensor, np.eye(4))
nib.save(img_train_tensor, 'train-5D-tensor.nii.gz')

img_test_tensor = nib.Nifti1Image(tmp_test_tensor, np.eye(4))
nib.save(img_test_tensor, 'test-5D-tensor.nii.gz')