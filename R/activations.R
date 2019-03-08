##' logistic function
##'
##' A function to evaluate the logistic activation function, the derivative and cost derivative to be used in defining a neural network.
##'
##' @return a list of functions used to compute the activation function, the derivative and cost derivative.
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{ReLU}, \link{smoothReLU}, \link{ident}, \link{softmax}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context
##'
##' net <- network( dims = c(100,50,20,2),
##'                 activ=list(logistic(),ReLU(),softmax()))
##'
##' @export

logistic <- function(){
    retlist <- list()
    retlist$activ <- function(x){return(1/(1+exp(-x)))}
    retlist$deriv_activ <- function(x){return(exp(-x)/(1+exp(-x))^2)}
    retlist$deltaL_eval <- function(x,C){as.vector(x*C)}
    return(retlist)
}

##' hyptan function
##'
##' A function to evaluate the hyperbolic tanget activation function, the derivative and cost derivative to be used in defining a neural network.
##'
##' @return a list of functions used to compute the activation function, the derivative and cost derivative.
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{ReLU}, \link{smoothReLU}, \link{ident}, \link{softmax}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context
##'
##' net <- network( dims = c(100,50,20,2),
##'                 activ=list(hyptan(),ReLU(),softmax()))
##'
##' @export

hyptan <- function(){
    retlist <- list()
    retlist$activ <- function(x){return((exp(x)-exp(-x))/(exp(x)+exp(-x)))}
    retlist$deriv_activ <- function(x){return(4/(exp(x)+exp(-x))^2)}
    retlist$deltaL_eval <- function(x,C){as.vector(x*C)}
    return(retlist)
}

##' smoothReLU function
##'
##' A function to evaluate the smooth ReLU (AKA softplus) activation function, the derivative and cost derivative to be used in defining a neural network.
##'
##' @return a list of functions used to compute the activation function, the derivative and cost derivative.
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{logistic}, \link{ReLU}, \link{ident}, \link{softmax}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context
##'
##' net <- network( dims = c(100,50,20,2),
##'                 activ=list(smoothReLU(),ReLU(),softmax()))
##'
##' @export

smoothReLU <- function(){
    retlist <- list()
    retlist$activ <- function(x){return(log(1+exp(x)))}
    retlist$deriv_activ <- function(x){return(1/(1+exp(-x)))}
    retlist$deltaL_eval <- function(x,C){as.vector(x*C)}
    return(retlist)
}

##' ReLU function
##'
##' A function to evaluate the ReLU activation function, the derivative and cost derivative to be used in defining a neural network.
##'
##' @return a list of functions used to compute the activation function, the derivative and cost derivative.
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{logistic}, \link{smoothReLU}, \link{ident}, \link{softmax}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context
##'
##' net <- network( dims = c(100,50,20,2),
##'                 activ=list(ReLU(),ReLU(),softmax()))
##'
##' @export

ReLU <- function(){
    retlist <- list()
    retlist$activ <- function(x){return(pmax(0,x))}
    drv <- function(x){
        ans <- rep(0,length(x))
        ans[x>=0] <- 1
        return(ans)
    }
    retlist$deriv_activ <- drv
    retlist$deltaL_eval <- function(x,C){as.vector(x*C)}
    return(retlist)
}

##' ident function
##'
##' A function to evaluate the identity (linear) activation function, the derivative and cost derivative to be used in defining a neural network.
##'
##' @return a list of functions used to compute the activation function, the derivative and cost derivative.
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{logistic}, \link{ReLU}, \link{smoothReLU}, \link{softmax}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context
##'
##' net <- network( dims = c(100,50,20,2),
##'                 activ=list(ident(),ReLU(),softmax()))
##'
##' @export

ident <- function(){
    retlist <- list()
    retlist$activ <- function(x){return(x)}
    retlist$deriv_activ <- function(x){return(rep(1,length(x)))}
    retlist$deltaL_eval <- function(x,C){as.vector(x*C)}
    return(retlist)
}

##' softmax function
##'
##' A function to evaluate the softmax activation function, the derivative and cost derivative to be used in defining a neural network. Note that at present, this unit can only be used as an output unit.
##'
##' @return a list of functions used to compute the activation function, the derivative and cost derivative.
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{logistic}, \link{ReLU}, \link{smoothReLU}, \link{ident}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context
##'
##' net <- network( dims = c(100,50,20,2),
##'                 activ=list(logistic(),ReLU(),softmax()))
##'
##' @export

softmax <- function(){
    retlist <- list()
    retlist$activ <- function(x){return(exp(x)/(sum(exp(x))))}
    retlist$deriv_activ <- function(x){
        expx <- exp(as.vector(x))
        s <- sum(expx)^2
        l <- length(x)
        ans <- (-1) * outer(expx,expx) / s
        diag(ans) <- sapply(1:l,function(i){return(expx[i]*sum(expx[-i]))}) / s
        return(ans)
    }
    retlist$deltaL_eval <- function(x,C){return(as.vector(x%*%C))}
    return(retlist)
}
