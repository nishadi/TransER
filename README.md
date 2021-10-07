# TransER

This is the source code for the `TransER` framework proposed in the paper, 
'TransER: Homogeneous transfer learning for entity resolution'.

#### Data
Data sets are located at 'data/' directory. We provide the pickle  files 
that contains the feature vectors of each data set pair. For each data set we
 use three different blocking samples. For further details on data sets, 
 please refer to the paper.
 
#### Quick Start
 
 1. Define the source and target data sets and the linking entity types
 
    ```
    src_dataset = 'dblp-acm1'
    src_link = 'A-A'
    tgt_dataset = 'dblp-scholar1'
    tgt_link = 'A-A'
    ```
    
    We currently support the following data sets (followed by the link);
    * dblp-acm1 A-A
    * dblp-scholar1 A-A
    * MB S-S
    * MSD S-S
    * IOS Bp-Dp
    * KIL Bp-Dp
    * IOS Bp-Bp
    * KIL Bp-Bp
    
 2. Define the hyper parameters
 
    ```
    k = 7       # Neighbourhood size
    t_c = 0.9   # Threshold for instance confidence similarity
    t_l = 0.9   # Threshold for instance structural similarity
    t_p =  0.99 # Threshold for pseudo label confidence
    ```
 
 3. Match with transfer learning

    ```
    model.predict(src_dataset, src_link, tgt_dataset, tgt_link, k, t_c, t_l, t_p)
    ``` 
 
    
    
#### Dependencies

The `TransER` package requires the following python packages to be installed:
- [Python 3](https://www.python.org)
- [Pickle](https://docs.python.org/3/library/pickle.html)
- [Numpy](https://www.numpy.org)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)

    
#### Contact
Contact the author of the package: nishadi.kirielle@anu.edu.au
    