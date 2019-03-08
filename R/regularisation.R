##' no_regularisation function
##'
##' A function to return the no regularisation strategy for a network object.
##'
##' @return list containing functions to evaluate the cost modifier and grandient modifier
##' @seealso \link{network}, \link{train}, \link{L1_regularisation}, \link{L2_regularisation}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context: NOTE with the network function
##' # no_regularisation() is the default, so this argument
##' # actually need not be included
##'
##' net <- network( dims = c(784,16,16,10),
##'                 regulariser = no_regularisation(),
##'                 activ=list(ReLU(),logistic(),softmax()))
##'
##' @export

no_regularisation <- function(){
    retlist <- list()
    retlist$cost_modifier <- function(params){return(0)}
    retlist$grad_modifier <- function(params){return(rep(0,length(params)))}
    return(retlist)
}

##' L1_regularisation function
##'
##' A function to return the L1 regularisation strategy for a network object.
##'
##' @param alpha parameter to weight the relative contribution of the regulariser
##' @return list containing functions to evaluate the cost modifier and grandient modifier
##' @seealso \link{network}, \link{train}, \link{L2_regularisation}, \link{no_regularisation}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context: NOTE the value of 1 used here is arbitrary,
##' # to get this to work well, you'll have to experiment.
##'
##' net <- network( dims = c(784,16,16,10),
##'                 regulariser = L1_regularisation(1),
##'                 activ=list(ReLU(),logistic(),softmax()))
##'
##' @export

L1_regularisation <- function(alpha){
    retlist <- list()
    retlist$cost_modifier <- function(params){return(alpha*sum(abs(params)))}
    retlist$grad_modifier <- function(params){return(alpha*sign(params))}
    return(retlist)
}




##' L2_regularisation function
##'
##' A function to return the L2 regularisation strategy for a network object.
##'
##' @param alpha parameter to weight the relative contribution of the regulariser
##' @return list containing functions to evaluate the cost modifier and grandient modifier
##' @seealso \link{network}, \link{train}, \link{L1_regularisation}, \link{no_regularisation}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context: NOTE the value of 1 used here is arbitrary,
##' # to get this to work well, you'll have to experiment.
##'
##' net <- network( dims = c(784,16,16,10),
##'                 regulariser = L2_regularisation(1),
##'                 activ=list(ReLU(),logistic(),softmax()))
##'
##' @export

L2_regularisation <- function(alpha){
    retlist <- list()
    retlist$cost_modifier <- function(params){return(alpha*sum(params^2))}
    retlist$grad_modifier <- function(params){return(alpha*params)}
    return(retlist)
}
