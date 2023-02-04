from scipy.stats import geom

def geometric_distribution(fp, fn, prior, alpha):
    A = ( (1 - fn) * prior ) / ( (1 - fn) * prior + fp * (1 - prior) )
    B = 1 - A
    print("Probability of same person given a positive test = ", A)
    i = 1
    while True:
        prob = geom.cdf(i, A)
        if i % 10 == 0:
            print("Probability with ", i, " samples = ", prob)
        if prob > alpha:
            print("It will take ", i - 1, "false positives to get the first true positive with ", alpha * 100, "% certainty")
            return i
        i += 1
    #print("It will take more than ", i, " false positives to get the first true positive with ", alpha * 100, "% certainty")
    #return i


if __name__ == "__main__":
    # false positive rate
    fp = 0.05
    # false negative rate
    fn = 0.09
    # prevalence is the prior of the proportion of fingerprints in the dataset belonging to the same person
    prevalence = 0.001
    # confidence level
    alpha = 0.95
    output = geometric_distribution(fp, fn, prevalence, alpha)
