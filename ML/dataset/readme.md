# Dataset

## Data Structure

**dataset**/

​	--**input**:/ Cellular geometry parameters(before Abaqus Processing)

​	--**model_data**/: dataset for training

​		--**dataset_test**/:  temporary dataset for debugging 

​			--**CVAE_test.csv**

​		--**data_process.ipynb**: dataset construction and preview of data relationship

​		--**forward_data_type0.csv**: Dataset for forward network of type 0 structure

​		--**forward_data_type3.csv**: Dataset for forward network of type 3 structure

​		--**model_data.csv**:  Dataset embodying all information processed from odb

​	--**get_dataset.py** : Generate the model_data file