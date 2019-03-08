##' NNpredict function
##'
##' A function to produce predictions from a trained network
##'
##' @param net an object of class network, see ?network
##' @param param vector of trained parameters from the network, see ?train
##' @param newdata input data to be predicted, a list of vectors (i.e. ragged array)
##' @param newtruth the truth, a list of vectors to compare with output from the feed-forward network
##' @param freq frequency to print progress updates to the console, default is every 1000th training point
##' @param record logical, whether to record details of the prediction. Default is FALSE
##' @param plot locical, whether to produce diagnostic plots. Default is FALSE
##' @return if record is FALSE, the output of the neural network is returned. Otherwise a list of objects is returned including: rec, the predicted probabilities; err, the L1 error between truth and prediction; pred, the predicted categories based on maximum probability; pred_MC, the predicted categories based on maximum probability; truth, the object newtruth, turned into an integer class number
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

NNpredict <- function(net,param,newdata,newtruth=NULL,freq=1000,record=FALSE,plot=FALSE){

    if(record){
        err <- c()
        rec <- c()
    }
    for(i in 1:length(newdata)){

        if(i%%freq==0){cat(i,"\n")}

        testdat <- newdata[[i]]
        w <- weights2list(param[1:nnetpar(net)],net$dims)
        b <- bias2list(param[(nnetpar(net)+1):length(param)],net$dims)
        cls <- net$forward_pass(testdat,
                                weights=w,
                                bias=b,
                                dims=net$dims,
                                nlayers=net$nlayers,
                                activ=net$activ,
                                back=FALSE,
                                regulariser=net$regulariser)

        if(record){
            err[i] <- sum(abs(cls$output-newtruth[[i]]))
            rec <- rbind(rec,cls$output)
        }
    }

    if(record){
        prednos_MC <- apply(rec,1,function(p){return(sample(1:length(newtruth[[1]]),1,prob=p))})
        prednos <- apply(rec,1,function(p){x<-which(p==max(p));return(x[sample(1:length(x),1)])})
        tru <- sapply(newtruth,function(x){return(which(x==1))})
    }

    if(plot){
        par(mfrow=c(2,2))
        plot(err)
        lines(lowess(err),col="red",main="")

        plot(jitter(tru),jitter(prednos),main="Prediction Vs Truth ABS")
        plot(jitter(tru),jitter(prednos_MC),main="Prediction Vs Truth MC")
    }

    if(record){
        return(list(rec=rec,err=err,pred=prednos,pred_MC=prednos_MC,truth=tru))
    }
    else{
        return(cls$output)
    }
}
