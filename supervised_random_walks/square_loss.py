# Input y and function
# Returns square loss

def square_loss(y, func):
    return (1 - (y*func()))**2
