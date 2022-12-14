# DeepNeuroevolution paper optimizer
def update_genome(old_genome, beta = 0.05):
    updated_genome = old_genome + beta * np.random.normal(scale=1,loc=0,size=old_genome.shape)
    return updated_genome

#def apply_output(preds, output_shape):
#    np.reshape


"""
def apply_output(preds, output):
    try:
        output = output.squeeze()
    except:
        pass
    try:
        preds = preds.squeeze()
    except: 
        pass

    if len(preds.shape) == len(output.shape):
        try:
            output[:] = preds
        except:
            output = preds
    elif len(preds.shape) == 1 and len(output.shape) == 3:
        output[0,0] = preds
    elif len(preds.shape) == 1 and len(output.shape) == 2:
        output[0] = preds
    elif len(preds.shape) == 2 and len(output.shape) == 3:
        output[0] = preds
    elif len(preds.shape) == 2 and len(output.shape) == 1:
        output = preds[0]
    elif len(preds.shape) == 3 and len(output.shape) == 1:
        output = preds[0,0]
    elif len(preds.shape) == 3 and len(output.shape) == 2:
        output = preds[0,0]
    return output
"""