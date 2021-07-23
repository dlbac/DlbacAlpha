## Dependent Libraries ##
  * python 3.8 (experimented with version 3.8.10)
  * keras 2.5.0
  * tensorflow (experimented with version 2.5.0)
  * numpy (experimented with version 1.19.5)

## Run ##
dlbac_alpha_resnet.py file contain all the source code related to the DLBAC_alpha ResNet.

This python script takes two parameters- train and test file in following order.

python3 dlbac_alpha_resnet.py <train_data_file_path> <test_data_file_path>

For example,
python3 dlbac_alpha_resnet.py ../dataset/train_u5k-r5k-auth19k.sample ../dataset/test_u5k-r5k-auth19k.sample 

## Output ##
The **output** will be a *trained dlbac_alpha network* and saved in dlbac_alpha.hdf5 file. 

The training history, and accuracy related results will be stored in history_dlbac_aplha and results.txt files, respectively. 

All the output and result files will be exported in the /results directory.
