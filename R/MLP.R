##' MLP_net function
##'
##' A function to define a multilayer perceptron and compute quantities for backpropagation, if needed.
##'
##' @param input input data, a list of vectors (i.e. ragged array)
##' @param weights a list object containing weights for the forward pass, see ?weights2list
##' @param bias a list object containing biases for the forward pass, see ?bias2list
##' @param dims the dimensions of the network as stored from a call to the function network, see ?network
##' @param nlayers number of layers as stored from a call to the function network, see ?network
##' @param activ list of activation functions as stored from a call to the function network, see ?network
##' @param back logical, whether to compute quantities for backpropagation (set to FALSE for feed-forward use only)
##' @param regulariser type of regularisation strategy to, see ?train, ?no_regularisation ?L1_regularisation, ?L2_regularisation
##' @return a list object containing the evaluated forward pass and also, if selected, quantities for backpropagation.
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



MLP_net <- function(input,weights,bias,dims,nlayers,activ,back=TRUE,regulariser){
    out <- input

    if(back){
        A <- memInit(dims[-1])
        diffA <- A
        Z <- A
    }

    for(i in 1:nlayers){
        z <- weights[[i]] %*% out + bias[[i]]
        if(inherits(activ$activ,"function")){
            a <- activ$activ(z)
        }
        else{
            a <- activ[[i]]$activ(z)
        }

        out <- a

        if(back){
            Z[[i]] <- as.vector(z)
            if(inherits(activ$activ,"function")){
                diffA[[i]] <- activ$deriv_activ(z)
            }
            else{
                diffA[[i]] <- activ[[i]]$deriv_activ(z)
            }
            diffA[[i]] <- diffA[[i]]
            A[[i]] <- as.vector(a)

        }
    }

    retlist <- list()

    retlist$input <- input
    retlist$W <- weights
    retlist$bias <- bias
    retlist$dims <- dims
    retlist$nlayers <- nlayers
    retlist$output <- as.vector(out)
    retlist$Z <- NULL
    retlist$diffA <- NULL
    retlist$A <- NULL
    retlist$regulariser <- regulariser
    retlist$activ <- activ

    if(back){
        retlist$Z <- Z
        retlist$diffA <- diffA
        retlist$A <- A
    }

    return(retlist)
}

##' backpropagation_MLP function
##'
##' A function to perform backpropagation for a multilayer perceptron.
##'
##' @param MLPNet output from the function MLP_net, as applied to some data with given parameters
##' @param loss the loss function, see ?Qloss and ?multinomial
##' @param truth the truth, a list of vectors to compare with output from the feed-forward network
##' @return a list object containing the cost and the gradient with respect to each of the model parameters
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



backpropagation_MLP <- function(MLPNet,loss,truth){

    force(MLPNet)

    activ <- MLPNet$activ

    if(!is.null(MLPNet$regulariser)){
        W <- unlist(MLPNet$W)
        b <- unlist(MLPNet$bias)
        params <- c(W,b)
    }

    L <- MLPNet$nlayers

    cost <- loss$loss(truth=truth,output=MLPNet$output)
    grad_cost <- loss$grad_loss(truth=truth,output=MLPNet$output)

    if(!is.null(MLPNet$regulariser)){
        cost <- cost + MLPNet$regulariser$cost_modifier(params)
    }

    delta <- list()
    deltaB <- list()

    if(inherits(activ$activ,"function")){
        deltaCurrent <- activ$deltaL_eval(x=MLPNet$diffA[[L]],C=grad_cost)
    }
    else{
        deltaCurrent <- activ[[L]]$deltaL_eval(x=MLPNet$diffA[[L]],C=grad_cost)
    }

    delta[[L]] <- deltaCurrent
    deltaB[[L]] <- deltaCurrent

    for(l in (L-1):1){
        delta[[l]] <- as.vector((t(MLPNet$W[[l+1]]) %*% deltaCurrent)) * as.vector(MLPNet$diffA[[l]])
        deltaCurrent <- delta[[l]]
        deltaB[[l]] <- delta[[l]]
    }

    Aminus1 <- MLPNet$A
    Aminus1 <- Aminus1[-L]
    Aminus1 <- c(list(MLPNet$input),Aminus1)

    dCost_dW <- mapply(outer,delta,Aminus1)
    dCost_db <- deltaB

    if(!is.null(MLPNet$regulariser)){
        cost_mod <- MLPNet$regulariser$grad_modifier(params)

        dCost_dW <- addList(dCost_dW,weights2list(cost_mod[1:length(W)],MLPNet$dims))
        dCost_db <- addList(dCost_db,bias2list(cost_mod[(length(W)+1):length(cost_mod)],MLPNet$dims))
    }

    return(list(cost=cost,dCost_db=dCost_db,dCost_dW=dCost_dW))
}
