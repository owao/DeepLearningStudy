def shape(csv):
    """
    check the shape of csv data
    """
    print("data's count of row and column is", csv.shape, "...")


def nameofcolumn(csv):
    """
    check the name of columns
    """
    print(csv.columns)

def kinds(csv, labelname):
    """
    count the kinds of labels
    """
    print("kind of %s: %d" %(labelname, csv[labelname].nunique()))
    # z
    '''
    if csv[labelname].nunique() < csv[labelname].shape():
        print("There are %d duplication" %(csv[labelname].shape() - csv[labelname].nunique()))
    '''

def missingValue(csv):
    print(csv.isnull().sum())