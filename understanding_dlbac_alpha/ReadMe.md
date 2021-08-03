## Dependent Libraries ##
  * python 3.8 (experimented with version 3.8.10)
  * Pytorch 1.5.0
  * tensorflow (experimented with version 2.2.0)
  * numpy (experimented with version 1.20.3)

## Run ##
resnet.py file contain all the source code related to ResNet.
dataloader.py is the utility file for processing data.


dlbac_alpha_training.py file contain all the source code related to the DLBAC_alpha training.

This python script has two required parameters. --train_data (training dataset file path) and --test_data (test dataset file path).

There are also three other optional parameters.

--depth (type: int. Determine the layers of the ResNet network, default value is 8.)
--epochs (type: int. Number of epochs, default value is 60).
--debug (type: bool. Display the detailed logs, default False). 

For example,
python3 dlbac_alpha_training.py --train_data dataset/train_u4k-r4k-auth11k.sample --test_data dataset/test_u4k-r4k-auth11k.sample

## Output ##
The **output** will be a *trained dlbac_alpha network* and saved in model_state.pth path. 

System keeps track of all the configurations and stored in config.json file.

The output file and configuration will be exported in the /results directory.
