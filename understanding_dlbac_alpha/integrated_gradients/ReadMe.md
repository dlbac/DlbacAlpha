_For IngratedGradients experiments, we used **captum** library. The library works only for the models implemented in **pytorch**. Also, we provide understanding for a specific operation. E.g. Alice has op1 access on projectA resource. Hence, to keep it simple, we implement DLBAC_alpha for a single operation (as opposed to four operation in our DLBAC_alpha network experimentation section) using pytorch that is compatible with captum library._

## Dependent Libraries ##
  * python 3.8 (experimented with version 3.8.10)
  * Pytorch 1.5.0
  * tensorflow (experimented with version 2.2.0)
  * numpy (experimented with version 1.20.3)
  * captum 0.3.1

# Run #
## Model Training ##

_We implement a DLBAC_alpha in pytorch for Op1 operation. We only need to train DLBAC_alpha once. The same trained network can be used for both local and global interpretation._

resnet.py file contain pytorch implementation of ResNet.
dataloader.py is the utility file for processing data.


dlbac_alpha_training.py file contain all the source code related to the DLBAC_alpha training.

This python script has two required parameters. --train_data (training dataset file path) and --test_data (test dataset file path).

There are also three other optional parameters.

--depth (type: int. Determine the layers of the ResNet network, default value is 8.)
--epochs (type: int. Number of epochs, default value is 60).
--debug (type: bool. Display the detailed logs, default False). 

For example,
python3 dlbac_alpha_training.py --train_data dataset/train_u4k-r4k-auth11k.sample --test_data dataset/test_u4k-r4k-auth11k.sample

### Output ###
The **output** will be a *trained dlbac_alpha network* and saved in model_state.pth path. 

System keeps track of all the configurations and stored in config.json file.

The output file and configuration will be exported in the /result directory.


## Local Interpretation ##

Local interpretation works based on a trained DLBAC_alpha network. For simplicity, we assume the trained network is stored in /result directory.

local_interpretation.py file contain all the source code related to local interpretation (understanding decision of a single sample).

This python script has a required parameters. --dataset (dataset file path).

There are also three other optional parameters.

--depth (type: int. Determine the layers of the ResNet network, default value is 8. The depth of the network has to be same as the stored network).
--index (type: int. Index of sample in the dataset, default value is 1). For the simplicity we take index of the sample as the input, which equivalent to taking both uid of a user and rid a resource as input. Based on the index, internally system determines uid and rid. Then, it retrieves their access to the corresoponding operation.
--debug (type: bool. Display the detailed logs, default False). 

For example, python3 local_interpretation.py --dataset dataset/train_UR_0_4500_4500.sample --index 1


### Output ###
The script outputs attribution information for corresponding sample. 

The output file will be exported in the result/local_interpret_result.txt file.


## Global Interpretation ##

Global interpretation also works based on a trained DLBAC_alpha network. For simplicity, we assume the trained network is stored in /result directory.

global_interpretation.py file contain all the source code related to global interpretation (understanding decision of a batch of samples together). We evaluated the global interpretation for a batch of maximum 50 samples.

This python script has a required parameters. --data (dataset file path).

There are also three other optional parameters.

--depth (type: int. Determine the layers of the ResNet network, default value is 8. The depth of the network has to be same as the stored network).
--batch_size (type: int. The size of batch, default value is 50). We experimented for maximum batch size of 50.
--debug (type: bool. Display the detailed logs, default False). 

For example, python3 global_interpretation.py --data dataset/train_UR_0_4500_4500.sample --batch_size 20


### Output ###
The script outputs attribution information for corresponding batch. 

The output file will be exported in the result/global_interpret_result.txt file.

