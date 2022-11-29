import os


def route():
    """
    check the file route
    """
    print("Your file route is \"%s\"..." %(os.getcwd()))

    
def csvRoad(csv, count):
    """
    check roading of csv data
    """
    print(csv.head(count))
