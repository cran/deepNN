##' NNgrad_test function
##'
##' A function to test gradient evaluation of a neural network by comparing it with central finite differencing.
##'
##' @param net an object of class network, see ?network
##' @param loss a loss function to compute, see ?Qloss, ?multinomial
##' @param eps small value used in the computation of the finite differencing. Default value is 0.00001
##' @return the exact (computed via backpropagation) and approximate (via central finite differencing) gradients and also a plot of one
##' against the other.
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{logistic}, \link{ReLU}, \link{smoothReLU}, \link{ident}, \link{softmax}, \link{Qloss}, \link{multinomial},
##' \link{NNgrad_test}, \link{weights2list}, \link{bias2list}, \link{biasInit}, \link{memInit}, \link{gradInit},
##' \link{addGrad}, \link{nnetpar}, \link{nbiaspar}, \link{addList}, \link{no_regularisation}, \link{L1_regularisation},
##' \link{L2_regularisation}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' net <- network( dims = c(5,10,2),
##'                 activ=list(ReLU(),softmax()))
##' NNgrad_test(net)
##'
##' @export

NNgrad_test <- function(net,loss=Qloss(),eps=0.00001){

    npar <- nnetpar(net) + nbiaspar(net)

    parameters <- runif(npar,-1,1)

    weights <- weights2list(parameters[1:nnetpar(net)],net$dims)
    bias <- bias2list(parameters[(nnetpar(net)+1):length(parameters)],net$dims)

    inputs <- runif(net$input_length)
    truth <- runif(net$output_length)

    calculated <- net$forward_pass( inputs,
                                    weights=weights,
                                    bias=bias,
                                    dims=net$dims,
                                    nlayers=net$nlayers,
                                    activ=net$activ,
                                    back=TRUE,
                                    regulariser=net$regulariser)

    exact <- net$backward_pass(calculated,loss=loss,truth=truth)
    exact <- c(unlist(exact$dCost_dW),unlist(exact$dCost_db))

    approx <- c()

    for(i in 1:length(parameters)){
        newparameters <- parameters
        newparameters[i] <- parameters[i] + eps # see line newparameters[i] <- newparameters[i] - eps below

        weights <- weights2list(newparameters[1:nnetpar(net)],net$dims)
        bias <- bias2list(newparameters[(nnetpar(net)+1):length(newparameters)],net$dims)

        feval <- net$forward_pass(  inputs,
                                    weights=weights,
                                    bias=bias,
                                    dims=net$dims,
                                    nlayers=net$nlayers,
                                    activ=net$activ,
                                    back=TRUE,
                                    regulariser=net$regulariser)


        beval_plus <- net$backward_pass(feval,loss=loss,truth=truth)

        newparameters <- parameters
        newparameters[i] <- parameters[i] - eps

        weights <- weights2list(newparameters[1:nnetpar(net)],net$dims)
        bias <- bias2list(newparameters[(nnetpar(net)+1):length(newparameters)],net$dims)

        feval <- net$forward_pass(  inputs,
                                    weights=weights,
                                    bias=bias,
                                    dims=net$dims,
                                    nlayers=net$nlayers,
                                    activ=net$activ,
                                    back=TRUE,
                                    regulariser=net$regulariser)

        beval_minus <- net$backward_pass(feval,loss=loss,truth=truth)


        approx[i] <- (beval_plus$cost - beval_minus$cost) / (2*eps)
    }

    plot(exact,approx,col=c(rep("black",nnetpar(net)),rep("red",nbiaspar(net))))
    abline(0,1)

    return(list(exact=exact,approx=approx))
}
