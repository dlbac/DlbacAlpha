_For IngratedGradients experiments, we used **captum** library. The library works only for the models implemented in **PyTorch**. Also, we provide understanding for a specific operation. E.g., Alice has op1 access to projectA resource. Hence, to keep it simple, we implement DLBAC_alpha for a single operation (as opposed to four operations in our DLBAC_alpha network experimentation section) using PyTorch compatible with the captum library._

## Dependent Libraries ##
  * python 3.6 or above (experimented with version 3.8.10)
  * Pytorch 1.5.0 (https://pytorch.org/get-started/previous-versions/#linux-and-windows-11  
For CPU: 
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html)
  * tensorflow  2.2 or above (experimented with version 2.5.0)
  * numpy (experimented with version 1.20.3)
  * captum 0.3.1

# Run #
## _Training DLBAC_alpha Network_ ##

_We implement DLBAC_alpha in PyTorch for a single operation (we experimented with op1). We need to train DLBAC_alpha once only. The same trained network can be used for both local and global interpretation. A trained DLBAC_alpha (dlbac_alpha.pth) is added in the `neural_network/` directory._


`resnet.py` file contains pytorch implementation of ResNet.
`dataloader.py` is the utility file for processing data.


`dlbac_alpha_training.py` file contains all the source code related to the DLBAC_alpha training.

This python script has two required parameters. --train_data (training dataset file path) and --test_data (test dataset file path).

There are also three other optional parameters.

--depth (type: int. Determine the layers of the ResNet network, the default value is 8.)  
--epochs (type: int. Number of epochs, the default value is 60).  
--debug (type: bool. Display the detailed logs, default False).  

For example,
python3 dlbac_alpha_training.py --train_data dataset/u4k-r4k-auth11k/train_u4k-r4k-auth11k.sample --test_data dataset/u4k-r4k-auth11k/test_u4k-r4k-auth11k.sample

### Output ###
The **output** will be a *trained dlbac_alpha network* and saved in `neural_network/dlbac_alpha.pth` path.  
The system keeps track of all the configurations and stores them in a `result/config.json` file.  

**Required Time:** _CPU: About 1 hour_  


## _Local Interpretation_ ##

Local interpretation works based on a trained DLBAC_alpha network (dlbac_alpha.pth). We assume the trained network is stored in `neural_network/` directory.

`local_interpretation.py` file contains all the source code related to local interpretation (understanding decision of a single sample).

This python script has three required parameters. `--data` (dataset file path), `--uid` (unique id of a user), and `--rid` (unique id of a resource). As this experiment is for _op1_, based on _uid_ and _rid_, the system determines the user's access to the corresponding resource. 

There are also two other optional parameters.

--depth (type: int. Determine the layers of the ResNet network, the default value is 8. The depth of the network has to be the same as the stored network).  
--debug (type: bool. Display the detailed logs, default False).  

For example, following command determines the local interpretation of a tuple with uid=3660, rid=3878, and `grant` access to op1 operation.  
python3 local_interpretation.py --data dataset/u4k-r4k-auth11k/u4k-r4k-auth11k.sample --uid 3660 --rid 3878 

### Output ###
The script outputs attribution information (local interpretation) for the corresponding tuple.  
The output file will be exported in the `result/local_interpret_result.txt` file.


## _Global Interpretation_ ##

Global interpretation also works based on a trained DLBAC_alpha network (dlbac_alpha.pth). We assume the trained network is stored in `neural_network/` directory.

`global_interpretation.py` file includes all the source code related to global interpretation (understanding decision of a batch of samples together). We evaluated the global interpretation for a batch of a maximum of 50 samples.

This python script has the required parameters. `--data` (dataset file path).

There are also three other optional parameters.

--depth (type: int. Determine the layers of the ResNet network, the default value is 8. The depth of the network has to be the same as the stored network).  
--batch_size (type: int. The size of the batch, the default value is 50). We experimented for a maximum batch size of 50.  
--debug (type: bool. Display the detailed logs, default False).  

For example, following command determines the global interpretation for tuples with grant access.  
python3 global_interpretation.py --data dataset/u4k-r4k-auth11k/train_u4k-r4k-auth11k_grant.sample  

### Output ###
The script outputs attribution information for the corresponding batch.  
The output file will be exported in the `result/global_interpret_result.txt` file.


## _Application of Integrated Gradients based Understanding_ ##

This experimentation also works based on a trained DLBAC_alpha network (dlbac_alpha.pth). We assume the trained network is stored in `neural_network/` directory.

`application_integrated_gradients.py` file contains all the source code related to _Integrated Gradients based understanding application_ experimentation (Section 5.1.1 in the paper).  

This python script has the required parameters, --data (dataset file path). For this experiment, we split our train_u4k-r4k-auth11k.sample file into two different files based on the op1 access information. We create a train_u4k-r4k-auth11k_grant.sample that contains all the samples with grant access on op1 operation. For the rest of the samples, we create a train_u4k-r4k-auth11k_deny.sample file. Essentially, this **train_u4k-r4k-auth11k_deny.sample** is the input to the script. We provide both files in **dataset/synthetic/u4k-r4k-auth11k/** directory.  

We randomly select a tuple (uid:4246, rid:4435) with grant access on op1 and apply its corresponding metadata values for changed value. We change metadata values in order of their significance level in the global interpretation (Figure 10 in the paper). E.g., first, we change the value of _rmeta2_ metadata.  

There are also two other optional parameters.  
--depth (type: int. Determine the layers of the ResNet network, the default value is 8. The depth of the network has to be the same as the stored network).  
--debug (type: bool. Display the detailed logs, default False).  

Following is a sample command for running this experiment.   
python3 application_integrated_gradients.py --data dataset/u4k-r4k-auth11k/train_u4k-r4k-auth11k_deny.sample


### Output ###
We evaluate for all the samples in train_u4k-r4k-auth11k_deny.sample.
As all the samples (tuples) in train_u4k-r4k-auth11k_deny.sample datasets are with _deny_ access. Ideally, without any change, the accuracy should be as close as 100%. However, with the evolution of a different number of metadata values, this accuracy decreases, indicating the tuples are receiving grant access.  
As such, we measure what percentage of tuples still have _deny_ access. We also measure what percentage of tuples are receiving _**grant**_ access with the change of their metadata values.

The output file will be exported in the `result/application_integrated_gradients_result.txt` file.
