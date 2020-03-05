##' network function
##'
##' A function to set up a neural network structure.
##'
## NO LONGER USED @param netfun a function for evaluating the forward pass of a neural network, see for example ?MLP_Net
## NO LONGER USED @param backprop a function for evaluating backpropagation for netfun, see for example ?backpropagation_MLP
##' @param dims a vector giving the dimensions of the network. The first and last elements are respectively the input and output lengths and the intermediate elements are the dimensions of the hidden layers
##' @param activ either a single function or a list of activation functions, one each for the hidden layers and one for the output layer. See for example ?ReLU, ?softmax etc.
##' @param regulariser optional regularisation strategy, see for example ?no_regularisation (the default) ?L1_regularisation, ?L2_regularisation
##' @return a list object with all information to train the network
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
##'
##' net <- network( dims = c(100,50,50,20),
##'                 activ=list(ReLU(),ReLU(),softmax()),
##'                 regulariser=L1_regularisation())
##'
##' @export

network <- function(dims,activ=logistic(),regulariser=NULL){

    netfun <- MLP_net
    backprop <- backpropagation_MLP

    if(dims[1]<=2){
        stop("dims[1] must be bigger than 2")
    }

    n <- length(dims)

    retlist <- list()

    retlist$forward_pass <- netfun
    retlist$backward_pass <- backprop

    retlist$input_length <- dims[1]
    retlist$output_length <- rev(dims)[1]
    retlist$nlayers <- n-1
    retlist$ncount <- dims[-c(1,n)]
    retlist$activ <- activ
    retlist$dims <- dims
    retlist$regulariser <- regulariser

    class(retlist) <- c("network","list")

    return(retlist)
}



##' backprop_evaluate function
##'
##' A function used by the train function in order to conduct backpropagation.
##'
##' @param parameters network weights and bias parameters as a vector
##' @param dat the input data, a list of vectors
##' @param truth the truth, a list of vectors to compare with output from the feed-forward network
##' @param net an object of class network, see ?network
##' @param loss the loss function, see ?Qloss and ?multinomial
##' @param batchsize optional batchsize argument for use with stochastic gradient descent
##' @param dropout optional list of dropout probabilities ?dropoutProbs
##' @return the derivative of the cost function with respect to each of the parameters
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

backprop_evaluate <- function(parameters,dat,truth,net,loss,batchsize,dropout){

    if(!is.null(dropout)){
        if(dropout$hidden != 1){
            parameters[sample(c(FALSE,TRUE),length(parameters),replace=TRUE,prob=c(dropout$hidden,1-dropout$hidden))] <- 0
        }
    }

    weights <- weights2list(parameters[1:nnetpar(net)],net$dims)
    bias <- bias2list(parameters[(nnetpar(net)+1):length(parameters)],net$dims)

    GRAD <- gradInit(net$dims)

    forwardBackward <- function(input,tru,net,weights,bias,dims,nlayers,activ,loss,regulariser){
        feval <- net$forward_pass(input,weights=weights,bias=bias,dims=dims,nlayers=nlayers,activ=activ,regulariser=net$regulariser)
        beval <- net$backward_pass(feval,loss=loss,truth=tru)
        GRAD <<- addGrad(GRAD,beval)
    }

    N <- length(dat)
    if(is.null(batchsize)){
        samp <- 1:N
    }
    else{
        samp <- sample(1:N,batchsize)
    }

    input <- dat[samp]
    truth <- truth[samp]

    if(!is.null(dropout)){
        if(dropout$input != 1){
            dropfun <- function(x){
                x[sample(c(FALSE,TRUE),length(x),replace=TRUE,prob=c(dropout$input,1-dropout$input))] <- 0
                return(x)
            }
        }

        input <- lapply(input,dropfun)
    }

    grad_eval <- mapply(forwardBackward,input,truth,
                        MoreArgs=list(  net=net,
                                        weights=weights,
                                        bias=bias,
                                        dims=net$dims,
                                        nlayers=net$nlayers,
                                        activ=net$activ,
                                        loss=loss,
                                        regulariser=net$regulariser))

    return(GRAD)
}

##' train function
##'
##' A function to train a neural network defined using the network function.
##'
##' @param dat the input data, a list of vectors
##' @param truth the truth, a list of vectors to compare with output from the feed-forward network
##' @param net an object of class network, see ?network
##' @param loss the loss function, see ?Qloss and ?multinomial
##' @param tol stopping criteria for training. Current method monitors the quality of randomly chosen predictions from the data,
##' terminates when the mean predictive probabilities of the last 20 randomly chosen points exceeds tol, default is 0.95
##' @param eps stepsize scaling constant in gradient descent, or stochastic gradient descent
##' @param batchsize size of minibatches to be used with stochastic gradient descent
##' @param dropout optional list of dropout probabilities ?dropoutProbs
##' @param parinit a function of a single parameter returning the initial distribution of the weights, default is uniform on (-0.01,0.01)
##' @param monitor logical, whether to produce learning/convergence diagnostic plots
##' @param stopping method for stopping computation default, 'default', calls the function stopping.default
##' @param update and default for meth is 'classification', which calls updateStopping.classification
##' @return optimal cost and parameters from the trained network; at present, diagnostic plots are produced illustrating the parameters
##' of the model, the gradient and stopping criteria trace.
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
##' # Example in context:
##'
##' \donttest{
##' download_mnist("mnist.RData") # only need to download once
##' load("mnist.RData") # loads objects train_set, truth, test_set and test_truth
##'
##' net <- network( dims = c(784,16,16,10),
##'                 activ=list(ReLU(),ReLU(),softmax()))
##'
##' netwts <- train(dat=train_set,
##'                 truth=truth,
##'                 net=net,
##'                 eps=0.001,
##'                 tol=0.8, # normally would use a higher tol here e.g. 0.95
##'                 loss=multinomial(),
##'                 batchsize=100)
##'
##' pred <- NNpredict(  net=net,
##'                     param=netwts$opt,
##'                     newdata=test_set,
##'                     newtruth=test_truth,
##'                     record=TRUE,
##'                     plot=TRUE)
##' }
##'
##' # Example 2
##'
##' N <- 1000
##' d <- matrix(rnorm(5*N),ncol=5)
##'
##' fun <- function(x){
##'     lp <- 2*x[2]
##'     pr <- exp(lp) / (1 + exp(lp))
##'     ret <- c(0,0)
##'     ret[1+rbinom(1,1,pr)] <- 1
##'     return(ret)
##' }
##'
##' d <- lapply(1:N,function(i){return(d[i,])})
##'
##' truth <- lapply(d,fun)
##'
##' net <- network( dims = c(5,10,2),
##'                 activ=list(ReLU(),softmax()))
##'
##' netwts <- train( dat=d,
##'                  truth=truth,
##'                  net=net,
##'                  eps=0.01,
##'                  tol=100,            # run for 100 iterations
##'                  batchsize=10,       # note this is not enough
##'                  loss=multinomial(), # for convergence
##'                  stopping="maxit")
##'
##' pred <- NNpredict(  net=net,
##'                     param=netwts$opt,
##'                     newdata=d,
##'                     newtruth=truth,
##'                     record=TRUE,
##'                     plot=TRUE)
##'
##' @export

train <- function(dat,truth,net,loss=Qloss(),tol=0.95,eps=0.001,batchsize=NULL,dropout=dropoutProbs(),parinit=function(n){return(runif(n,-0.01,0.01))},monitor=TRUE,stopping="default",update="classification"){

    if(any(sapply(dat,length)!=net$dims[1])){
        ch <- which(sapply(dat,length)!=net$dims[1])
        stop("Input dimensions for data point(s)",paste(ch,collapse=", "),"are not correct.")
    }
    if(any(sapply(truth,length)!=rev(net$dims)[1])){
        ch <- which(sapply(truth,length)!=rev(net$dims)[1])
        stop("Output dimensions for data point(s)",paste(ch,collapse=", "),"are not correct.")
    }

    npar <- nnetpar(net) + nbiaspar(net)

    wrap <- function(params,dat,truth,net,loss,batchsize,dropout){
        bp <- backprop_evaluate(params,dat,truth,net,loss,batchsize,dropout)
        ans <- bp$cost
        attr(ans,"gradient") <- c(unlist(bp$dCost_dW),unlist(bp$dCost_db))
        return(ans)
    }

    cost <- NULL
    if(update=="classification"){
        curcost <- 0
    }
    else if(update=="regression"){
        curcost <- Inf
    }

    parms <- parinit(npar) #runif(npar,-0.01,0.01) #rep(0,npar) #rnorm(npar,0,0.1)
    del <- Inf
    count <- 1
    temp <- NA

    testoutput <- NULL

    mx <- curcost

    STOPPING <- getS3method("stopping",stopping)
    UPDATE <- getS3method("updateStopping",update)

    while(STOPPING(cost,curcost,count,tol)){
        wtest <- wrap(parms,dat=dat,truth=truth,net=net,loss=loss,batchsize=batchsize,dropout=dropout)

        #browser()

        if(monitor){
            par(mfrow=c(2,2))
            plot(parms,pch=".",cex=2,main="Network Weights")
            plot(attr(wtest,"gradient"),pch=".",cex=2,main="Gradient")
        }

        udate <- UPDATE(dat,parms,net,truth,testoutput,count,monitor,mx,curcost)
        curcost <- udate$curcost
        testoutput <- udate$testoutput
        mx <- udate$mx

        parms <- parms - eps*attr(wtest,"gradient")
        cost[count] <- wtest
        if(count>=2){
            del <- abs(cost[count] - cost[count-1])
        }
        print(c(cost[count],curcost))

        count <- count + 1
    }

    return(list(cost=cost,opt=parms))
}
