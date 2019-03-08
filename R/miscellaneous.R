##' weights2list function
##'
##' A function to convert a vector of weights into a ragged array (coded here a list of vectors)
##'
##' @param weights a vector of weights
##' @param dims the dimensions of the network as stored from a call to the function network, see ?network
##' @return a list object with appropriate structures for compatibility with the functions network, train, MLP_net and backpropagation_MLP
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

weights2list <- function(weights,dims){
    dims <- c(dims,0)
    wlist <- list()
    start <- 1
    for(i in 1:(length(dims)-2)){
        end <- start + dims[i]*dims[i+1] - 1
        wlist[[i]] <- matrix(weights[start:end],dims[i+1],dims[i])
        start <- start + dims[i]*dims[i+1]
    }
    return(wlist)
}



##' bias2list function
##'
##' A function to convert a vector of biases into a ragged array (coded here a list of vectors)
##'
##' @param bias a vector of biases
##' @param dims the dimensions of the network as stored from a call to the function network, see ?network
##' @return a list object with appropriate structures for compatibility with the functions network, train, MLP_net and backpropagation_MLP
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

bias2list <- function(bias,dims){
    blist <- list()
    start <- 1
    end <- dims[2]
    blist[[1]] <- bias[start:end]
    for(i in 2:(length(dims)-1)){
        start <- end + 1
        end <- start + dims[i+1] -1
        blist[[i]] <- bias[start:end]
    }
    return(blist)
}

##' biasInit function
##'
##' A function to inialise memory space for bias parameters. Now redundant.
##'
##' @param dims the dimensions of the network as stored from a call to the function network, see ?network
##' @return memory space for biases
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

biasInit <- function(dims){
    blist <- list()
    for(i in 2:(length(dims))){
        blist[[i-1]] <- rep(0,dims[i])
    }
    return(blist)
}

##' memInit function
##'
##' A function to initialise memory space. Likely this will become deprecated in future versions.
##'
##' @param dim the dimensions of the network as stored from a call to the function network, see ?network
##' @return memory space, only really of internal use
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

memInit <- function(dim){
    return(sapply(dim,function(x){rep(NA,x)}))
}

##' gradInit function
##'
##' A function to initialise memory for the gradient.
##'
##' @param dim the dimensions of the network as stored from a call to the function network, see ?network
##' @return memory space and structure for the gradient, initialised as zeros
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

gradInit <- function(dim){
    l <- list()
    l[[1]] <- 0
    l[[2]] <- list()
    l[[3]] <- list()
    for(i in 1:(length(dim)-1)){
        l[[2]][[i]] <- 0
        l[[3]][[i]] <- 0
    }
    names(l) <- c("cost","dCost_db","dCost_dW")
    return(l)
}


##' addGrad function
##'
##' A function to add two gradients together, gradients expressed as nested lists.
##'
##' @param x a gradient list object, as used in network training via backpropagation
##' @param y a gradient list object, as used in network training via backpropagation
##' @return another gradient object
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

addGrad <- function(x,y){
    x[[1]] <- x[[1]] + y[[1]]
    for(i in 1:length(x[[2]])){
        x[[2]][[i]] <- x[[2]][[i]] + y[[2]][[i]]
        x[[3]][[i]] <- x[[3]][[i]] + y[[3]][[i]]
    }
    return(x)
}

##' nnetpar function
##'
##' A function to calculate the number of weight parameters in a neural network, see ?network
##'
##' @param net an object of class network, see ?network
##' @return an integer, the number of weight parameters in a neural network
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
##' nnetpar(net)
##'
##' @export
nnetpar <- function(net){
    np <- 0
    calc <- c()
    for(i in 1:(length(net$dims)-1)){
        np <- np + net$dims[i]*net$dims[i+1]
        calc[i] <- paste("(",net$dims[i]," x ",net$dims[i+1],")",sep="")
    }
    attr(np,"calc") <- paste(calc,collapse=" + ")
    return(np)
}

##' nbiaspar function
##'
##' A function to calculate the number of bias parameters in a neural network, see ?network
##'
##' @param net an object of class network, see ?network
##' @return an integer, the number of bias parameters in a neural network
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
##' nbiaspar(net)
##'
##' @export

nbiaspar <- function(net){
    return(sum(net$dims[-1]))
}


##' addList function
##'
##' A function to add two lists together
##'
##' @param x a list
##' @param y a list
##' @return a list, the elements of which are the sums of the elements of the arguments x and y.
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


addList <- function(x,y){
    return(mapply("+",x,y))
}

##' download_mnist function
##'
##' A function to download mnist data in .RData format. File includes objects train_set, truth, test_set and test_truth
##'
##' @param fn the name of the file to save as
##' @return a list, the elements of which are the sums of the elements of the arguments x and y.
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##'     \item{Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998}
##'     \item{http://yann.lecun.com/exdb/mnist/}
##' }
##' @examples
##'
##' \donttest{
##' download_mnist("mnist.RData")
##' }
##'
##' @export

download_mnist <- function(fn){
    download.file("https://www.lancaster.ac.uk/staff/taylorb1/mnist.RData",fn)
}
