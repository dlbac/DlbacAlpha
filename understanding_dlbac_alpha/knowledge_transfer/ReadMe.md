
## Dependent Libraries ##
  * python 3.8 (experimented with version 3.8.10)
  * keras 2.5.0
  * tensorflow (experimented with version 2.5.0)
  * numpy (experimented with version 1.19.5)
  * scikit-learn (experimented with version 0.24.2)

  * IPython (experimented with version 7.26.0)
  * pydotplus (experimented with version 2.0.2)
  * graphviz (required to install with commad _`sudo apt-get install graphviz`_)

# Run #

_Knowledge Transferring technique first load a trained DLBAC_alpha network. We assume the network will be stored in the `result/` directory._

`knowledge_transfer.py` file contains all the source code related to _Knowledge Transferring_ experiment.

This python script has two required parameters. --train_data (training dataset file path) and --test_data (test dataset file path).

There are also three other optional parameters.

--max_depth (type: int. The maximum depth of the Decision Tree, the default value is 8). 
It is an optional parameter for the Decision Tree to limit the way the tree grows.
We experimented with various depth. For better visualization and moderate sized tree, we use max_depth=8.  
--model_name (type: String. The name of the trained DLBAC_alpha network, the default name is dlbac_alpha.hdf5).  
--debug (type: bool. Display the detailed logs, default False).  

For example,
python3 knowledge_transfer.py --train_data dataset/synthetic/u4k-r4k-auth11k/train_u4k-r4k-auth11k.sample --test_data dataset/synthetic/u4k-r4k-auth11k/test_u4k-r4k-auth11k.sample

### Output ###
The **output** will be a *Decision Tree* and saved in `result/dlbac_alpha_decision_tree.png` path.  

