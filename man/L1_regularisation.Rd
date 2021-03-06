% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/regularisation.R
\name{L1_regularisation}
\alias{L1_regularisation}
\title{L1_regularisation function}
\usage{
L1_regularisation(alpha)
}
\arguments{
\item{alpha}{parameter to weight the relative contribution of the regulariser}
}
\value{
list containing functions to evaluate the cost modifier and grandient modifier
}
\description{
A function to return the L1 regularisation strategy for a network object.
}
\examples{

# Example in context: NOTE the value of 1 used here is arbitrary,
# to get this to work well, you'll have to experiment.

net <- network( dims = c(784,16,16,10),
                regulariser = L1_regularisation(1),
                activ=list(ReLU(),logistic(),softmax()))

}
\references{
\enumerate{
    \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
    \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
    \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
    \item{http://neuralnetworksanddeeplearning.com/}
}
}
\seealso{
\link{network}, \link{train}, \link{L2_regularisation}, \link{no_regularisation}
}
