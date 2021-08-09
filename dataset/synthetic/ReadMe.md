# Syntax of the Dataset

There are eight different synthetic datasets, and each dataset is stored in a separate directory.
For example, a dataset named _u4k-r4k-auth11k_ includes in the `/u4k-r4k-auth11k` directory with their separate training and test files.

Each dataset has a varied number of users, resources, user-resource metadata. However, all the datasets have four operations (op1, op2, op3, and op4).

As described in the paper (Section 4.2.1), a dataset comprises a set of authorization tuples.  
The syntax of an authorization tuple is:  
unique id of a user _u_ | unique id of a resource _r_ | metadata values of all the user metadata of user _u_ |
metadata values of all the resource metadata of resource _r_ | access information of all the four operations

A sample authorization tuple is shown below.  
**3434 3410** `32 84 32 23 56 109 15 39 32 84 65 40 56 109 3 25` **_0 0 1 1_**

This authorization tuple can be read as -  
* A user with uid _3434_ has eight metadata, and their corresponding values are `32 84 32 23 56 109 15 39`.
* A resource with rid _3410_ has eight metadata, and their corresponding values are `32 84 65 40 56 109 3 25`.
* The user has _op3_ and _op4_ access to the resource as their corresponding binary digits are `1`.
