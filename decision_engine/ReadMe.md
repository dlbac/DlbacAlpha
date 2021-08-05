## Dependent Libraries ##
  * python 3.8 (experimented with version 3.8.10)
  * keras 2.5.0
  * tensorflow (experimented with version 2.5.0)
  * numpy (experimented with version 1.19.5)

# Run #

_The Decision Engine determine whether a user has requested access to a requested resource. It first load a trained DLBAC_alpha network (dlbac_alpha.hdf5). 
We assume the network is stored in the `neural_network/` directory._

`decision_engine.py` file contains all the source code related to _Decision Engine_.

This python script has three required parameters. --uid (unique id of a user), --rid (unique id of a resource) and --operation (name of the operation for 
which user requested access). For the sake of simplicity, we assume the uid and rid will be choosen from the tuples already available in the dataset.

There are also two other optional parameters.

--data (type: String. The path of the dataset, the default value is `dataset/u4k-r4k-auth11k.sample`). Given network is trained for this dataset (u4k-r4k-auth11k).
Hence, we kept all the default configuration for this dataset. However, if the network is trained for other dataset, then corresponding dataset can be communicated
through this parameter.

--debug (type: bool. Display the detailed logs, default False).  

For example, to know the access of user with uid=3704 and resource with rid=3856, we execute following command.
This tuple is 9'th tuple in the dataset (`dataset/u4k-r4k-auth11k.sample`).  
python3 decision_engine.py --uid 3704 --rid 3856 --operation op3

### Output ###
The **output** is displayed in the command line. If the user has access to given resource for the given operation, 
then the "_Access Granted!_" message will be displayed. Otherwise, it will show "_Access Denied!_"
