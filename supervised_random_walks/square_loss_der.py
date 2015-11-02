# Input y, square loss value
# Returns derivative of square loss

def square_loss_der(y, loss):
    return 2*y*((y*loss)-1)
