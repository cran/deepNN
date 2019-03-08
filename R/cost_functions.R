##' Qloss function
##'
##' A function to evaluate the quadratic loss function and the derivative of this function to be used when training a neural network.
##'
##' @return a list object with elements that are functions, evaluating the loss and the derivative
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{multinomial}, \link{no_regularisation}, \link{L1_regularisation}, \link{L2_regularisation}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context:
##'
##' \dontrun{
##' netwts <- train(dat=train_set,
##'                 truth=truth,
##'                 net=net,
##'                 eps=0.001,
##'                 tol=0.95,
##'                 loss=Qloss(), # note Qloss is actually the default
##'                 batchsize=100)
##' }
##'
##' @export

Qloss <- function(){
    retlist <- list()
    retlist$loss <- function(truth,output){return(sum((truth-output)^2))}
    retlist$grad_loss <- function(truth,output){return(-2*(truth-output))}
    return(retlist)
}

##' wQloss function
##'
##' A function to evaluate the weighted quadratic loss function and the derivative of this function to be used when training a neural network.
##'
##' @param w a vector of weights, adding up to 1, whose length is equalt to the output length of the net
##' @return a list object with elements that are functions, evaluating the loss and the derivative
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{multinomial}, \link{no_regularisation}, \link{L1_regularisation}, \link{L2_regularisation}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' # Example in context:
##'
##' \dontrun{
##' netwts <- train(dat=train_set,
##'                 truth=truth,
##'                 net=net,
##'                 eps=0.001,
##'                 tol=0.95,
##'                 loss=wQloss(c(10,5,6,9)), # here assuming output of length 4
##'                 batchsize=100)
##' }
##'
##' @export

wQloss <- function(w){
    retlist <- list()
    retlist$loss <- function(truth,output){return(sum(w*(truth-output)^2))}
    retlist$grad_loss <- function(truth,output){return(-2*w*(truth-output))}
    return(retlist)
}

##' multinomial function
##'
##' A function to evaluate the multinomial loss function and the derivative of this function to be used when training a neural network.
##'
##' @return a list object with elements that are functions, evaluating the loss and the derivative
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{Qloss}, \link{no_regularisation}, \link{L1_regularisation}, \link{L2_regularisation}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' \dontrun{
##' netwts <- train(dat=train_set,
##'                 truth=truth,
##'                 net=net,
##'                 eps=0.001,
##'                 tol=0.95,
##'                 loss=multinomial(),
##'                 batchsize=100)
##' }
##'
##' @export

multinomial <- function(){
    retlist <- list()
    retlist$loss <- function(truth,output){return(-sum(truth*log(output)))}
    retlist$grad_loss <- function(truth,output){return(-truth/output)}
    return(retlist)
}

##' wmultinomial function
##'
##' A function to evaluate the weighted multinomial loss function and the derivative of this function
##' to be used when training a neural network. This is eqivalent to a multinomial cost function
##' employing a Dirichlet prior on the probabilities. Its effect is to regularise the estimation so
##' that in the case where we apriori expect more of one particular category compared to another
##' then this can be included in the objective.
##'
##' @param w a vector of weights, adding up whose length is equal to the output length of the net
##' @param batchsize of batch used in inference WARNING: ensure this matches with actual batchsize used!
##' @return a list object with elements that are functions, evaluating the loss and the derivative
##' @seealso \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
##' \link{Qloss}, \link{no_regularisation}, \link{L1_regularisation}, \link{L2_regularisation}
##' @references
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' @examples
##'
##' \dontrun{
##' netwts <- train(dat=train_set,
##'                 truth=truth,
##'                 net=net,
##'                 eps=0.001,
##'                 tol=0.95,
##'                 loss=wmultinomial(c(10,5,6,9)), # here assuming output of length 4
##'                 batchsize=100)
##' }
##'
##' @export

wmultinomial <- function(w,batchsize){
    retlist <- list()
    retlist$loss <- function(truth,output){return(-sum((truth+(w-1)/batchsize)*log(output)))}
    retlist$grad_loss <- function(truth,output){return(-(truth+(w-1)/batchsize)/output)}
    return(retlist)
}
