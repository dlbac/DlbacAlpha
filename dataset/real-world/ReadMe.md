# Syntax of the Dataset
We experiment on Amazon Access Control dataset released for a Kaggle competition (a challenge to the community to build a machine learning model to determine the employees' accesses). 
The detailed task description and the dataset are available at https://www.kaggle.com/c/amazon-employee-access-challenge/.  

The dataset consists of historical access data of Amazon, where employees were manually allowed or denied access to resources over time.
Each element (authorization tuple) in the dataset specifies eight user metadata that illustrates a user's properties, a resource id to identify the resource, 
and a binary flag to indicate whether the user has access to the resource or not. We name this dataset amazon1.

However, as described in the paper (Section 4.1), we augment the dataset to create two other instances named amazon2 and amazon3. 
All these dataset instances have an equal number of users and an equal number of resources.  


The syntax of an authorization tuple:  
unique id of a user _u_ | unique id of a resource _r_ | metadata values of all the user metadata of user _u_ |
metadata values of all the resource metadata of resource _r_ | access information (1=allow, 0=deny)  

A sample authorization tuple from amazon1 instance is shown below.  
**6888 5624** `4698 117961 117962 118352 117905 117906 290919 117908 79092` **_1_**   

This authorization tuple can be read as -  
* A user with uid _6888_ has eight metadata, and their corresponding values are `4698 117961 117962 118352 117905 117906 290919 117908`.
* A resource with rid _5624_ has one metadata, and its corresponding value is `79092`.
* The user is allowed to access the resource as the respective binary flag is `1`.


