## Dependent Libraries ##
  * python 3.6 or above (experimented with version 3.8.10)
  * keras 2.5.0
  * tensorflow 2.2 or above (experimented with version 2.5.0)

# Run #

_The Decision Engine determines whether a user has requested access to a requested resource. It first load a trained DLBAC_alpha network (dlbac_alpha.hdf5). 
We assume the network is stored in the `neural_network/` directory._

`decision_engine.py` file contains all the source code related to _Decision Engine_.

This python script has three required parameters. --uid (unique id of a user), --rid (unique id of a resource), and --operation (name of the operation `op1, op2, op3, or op4` for which user requested access). For the sake of simplicity, we assume the uid and rid will be chosen from the tuples already available in the dataset.

There are also two other optional parameters.

--data (type: String. The path of the dataset, the default value is `dataset/u4k-r4k-auth11k.sample`). Given network is trained based on training data (train_u4k-r4k-auth11k.sample) of u4k-r4k-auth11k dataset.
However, if the network is trained for other datasets, the corresponding dataset can be provided through this parameter.

--debug (type: bool. Display the detailed logs, default False).  

For example, we execute the following command to know the access of a user with uid=3704 and a resource with rid=3856.
This tuple is 9'th sample (selected arbitrarily) in the dataset (`dataset/u4k-r4k-auth11k.sample`).  
python3 decision_engine.py --uid 3704 --rid 3856 --operation op3

### Output ###
The **output** is displayed in the command line. If the user has access to a given resource for the requested operation, 
the "_Access Granted!_" message will be displayed. Otherwise, it will show "_Access Denied!_."
