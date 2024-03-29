% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/regularisation.R
\name{L2_regularisation}
\alias{L2_regularisation}
\title{L2_regularisation function}
\usage{
L2_regularisation(alpha)
}
\arguments{
\item{alpha}{parameter to weight the relative contribution of the regulariser}
}
\value{
list containing functions to evaluate the cost modifier and grandient modifier
}
\description{
A function to return the L2 regularisation strategy for a network object.
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
\link{network}, \link{train}, \link{L1_regularisation}, \link{no_regularisation}
}
