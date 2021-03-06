% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/activations.R
\name{ReLU}
\alias{ReLU}
\title{ReLU function}
\usage{
ReLU()
}
\value{
a list of functions used to compute the activation function, the derivative and cost derivative.
}
\description{
A function to evaluate the ReLU activation function, the derivative and cost derivative to be used in defining a neural network.
}
\examples{

# Example in context

net <- network( dims = c(100,50,20,2),
                activ=list(ReLU(),ReLU(),softmax()))

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
\link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
\link{logistic}, \link{smoothReLU}, \link{ident}, \link{softmax}
}
