## Dependent Libraries ##
  * python 3.8 (experimented with version 3.8.10)
  * keras 2.5.0
  * tensorflow 2.2 or above (experimented with version 2.5.0)

## Run ##
`dlbac_alpha_resnet.py` file contain all the source code related to the DLBAC_alpha ResNet.

This python script takes two parameters- train and test file in following order.

**python3 dlbac_alpha_resnet.py <train_data_file_path> <test_data_file_path>**

For example, following command will build a trained dlbac_alpha network for the `u4k-r4k-auth11k` dataset.  
python3 dlbac_alpha_resnet.py ../dataset/train_u4k-r4k-auth11k.sample ../dataset/test_u4k-r4k-auth11k.sample 

## Output ##
The **output** will be a *trained dlbac_alpha network* and saved in dlbac_alpha.hdf5 file. 

The training history and training results will be stored in history_dlbac_aplha and results.txt files, respectively. 

All the trained network and all other files will be exported in the `results/` directory.
