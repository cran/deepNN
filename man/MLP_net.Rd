% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/MLP.R
\name{MLP_net}
\alias{MLP_net}
\title{MLP_net function}
\usage{
MLP_net(input, weights, bias, dims, nlayers, activ, back = TRUE, regulariser)
}
\arguments{
\item{input}{input data, a list of vectors (i.e. ragged array)}

\item{weights}{a list object containing weights for the forward pass, see ?weights2list}

\item{bias}{a list object containing biases for the forward pass, see ?bias2list}

\item{dims}{the dimensions of the network as stored from a call to the function network, see ?network}

\item{nlayers}{number of layers as stored from a call to the function network, see ?network}

\item{activ}{list of activation functions as stored from a call to the function network, see ?network}

\item{back}{logical, whether to compute quantities for backpropagation (set to FALSE for feed-forward use only)}

\item{regulariser}{type of regularisation strategy to, see ?train, ?no_regularisation ?L1_regularisation, ?L2_regularisation}
}
\value{
a list object containing the evaluated forward pass and also, if selected, quantities for backpropagation.
}
\description{
A function to define a multilayer perceptron and compute quantities for backpropagation, if needed.
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
\link{logistic}, \link{ReLU}, \link{smoothReLU}, \link{ident}, \link{softmax}, \link{Qloss}, \link{multinomial},
\link{NNgrad_test}, \link{weights2list}, \link{bias2list}, \link{biasInit}, \link{memInit}, \link{gradInit},
\link{addGrad}, \link{nnetpar}, \link{nbiaspar}, \link{addList}, \link{no_regularisation}, \link{L1_regularisation},
\link{L2_regularisation}
}
