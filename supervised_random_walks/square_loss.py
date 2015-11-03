# Input y and function
# Returns square loss

def square_loss(y, delta):
    if delta < 0:
        return 0
    return (1 - (y*delta))**2

# Input y, square loss value
# Returns derivative of square loss

def square_loss_der(y, delta):
    return 2*y*((y*delta)-1)
