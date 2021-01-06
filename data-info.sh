#!/bin/bash
# bash data-info.sh > list-input-info.csv
# bash data-info.sh > list-out-dwi-info.csv
# bash data-info.sh > list-out-tensor-info.csv


# cd Input/
# cd Out-dwi/
cd Out-tensor/
# cd Mask/

for file in *.nii.gz;
do
	
	info_=`fslinfo "$file"`
	dim1_=`echo $info_ | cut -d' ' -f4`
	dim2_=`echo $info_ | cut -d' ' -f6`
	dim3_=`echo $info_ | cut -d' ' -f8`
	dim4_=`echo $info_ | cut -d' ' -f10`
	echo "$file" "$dim1_" "$dim2_" "$dim3_" "$dim4_"

done