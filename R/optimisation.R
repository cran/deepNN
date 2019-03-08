##' dropoutProbs function
##'
##' A function to specify dropout for a neural network.
##'
##' @param input inclusion rate for input parameters
##' @param hidden inclusion rate for hidden parameters
##' @return returns these probabilities in an appropriate format for interaction with the network and train functions, see ?network and ?train
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
##' \dontrun{
##'
##' netwts <- train( dat=d,
##'                  truth=truth,
##'                  net=net,
##'                  eps=0.01,
##'                  tol=0.95,           # run for 100 iterations
##'                  batchsize=10,       # note this is not enough
##'                  loss=multinomial(), # for convergence
##'                  dropout=dropoutProbs(input=0.8,hidden=0.5))
##' }
##'
##' @export

dropoutProbs <- function(input=1,hidden=1){
    if(input > 1 | input <= 0){
        stop("input a probability in (0,1]")
    }
    if(hidden > 1 | hidden <= 0){
        stop("hidden a probability in (0,1]")
    }

    if(input==1 & hidden==1){
        return(NULL)
    }
    else{
        return(list(input=input,hidden=hidden))
    }
}
