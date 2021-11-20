from sklearn.preprocessing import MinMaxScaler
import numpy as np
def transform(DataMatrix):
    """
    Parameters:
        DataMatrix: A two dimentional matrix 
    
    Returns:
        Data_Matrix_transformed: Transformed mix max matrix 
    """
    DataMatrix = np.array(DataMatrix)

    Scaler = MinMaxScaler()
    Scaler.fit(DataMatrix)

    Data_matrix_transformed = Scaler.transform(DataMatrix)
    return Data_matrix_transformed

#testing 
#data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
#print(transform(data))