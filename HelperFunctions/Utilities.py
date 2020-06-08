'''
This contains a collection of commonly used functions I have written to perform
menial tasks not directly related to the extraction of relevant information
'''

def listToTxt(data, dir, **kwargs):
    # Converts a list of information into a txt folder with the inputted name
    # Inputs:   (data), the list to be saved
    #           (dir), the exact name and path which this list will be saved as
    #           (*args), inputs that appear at the top of the saved file
    # Outputs:  (), txt file saved in directory 

    # txt layout
    '''
    ArgumentNo_[number of addition arguments to read in]
    Arg_1_[some data]
    Arg_2_[some data]
    ...
    Rows_[X number of rows]
    Cols_[Y number of columns]
    [x0y0],[x0y1],[x0y2],[x0yY],...
    [x1y0],[x1y1],[x1y2],[x1yY],...
    [xXyY],...
    EndData
    '''

    f = open(dir, 'w')

    # declar
    f.write("ArgumnetNo_" + str(len(kwargs)) + "\n")

    # write the arguments at the top of the file
    for arg in kwargs.values():
        f.write("Arg_" + str(arg) + "\n")        

    X, Y = data.shape
    f.write("Rows_" + str(X) + "\n")
    f.write("Cols_" + str(Y) + "\n")

    for x in range(X):
        for y in range(Y):
            f.write(str(data[x, y]))
            if (y+1)%Y:
                f.write(",")
            else:
                f.write("\n")

    f.write("EndData")

    f.close()

def txtToList(dir):
    # Reads in a text file which was saved with the listToTxt function
    # Inputs:   (dir), the name/s of the file/s
    # Outputs:  (dataMain), a list containing the data
    #           (dataArgs), a list containing the argument data

    pass
