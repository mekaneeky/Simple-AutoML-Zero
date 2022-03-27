def _rescale_gaussian(gauss_arr, min_val = -100, max_val=100, normal_max=4, normal_min=-4):
    gauss_arr = (( (gauss_arr-normal_min)     \
                 /(normal_max-normal_min ))  \
                * (max_val- (min_val))) + min_val
    return gauss_arr

def update_genome(old_genome, beta = 0.05):
    updated_genome = old_genome + beta * np.random.normal(scale=1,loc=0,size=old_genome.shape)
    return updated_genome

