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
##' @seealso \link{NNpredict.regression}, \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
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
##' # Example 1 - mnist data
##'
##' # See example at mnist repository under user bentaylor1 on githib
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

        # browser()

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



##' NNpredict.regression function
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
##' @seealso \link{NNpredict}, \link{network}, \link{train}, \link{backprop_evaluate}, \link{MLP_net}, \link{backpropagation_MLP},
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
##' @export

NNpredict.regression <- function(net,param,newdata,newtruth=NULL,freq=1000,record=FALSE,plot=FALSE){

    if(record){
        err <- list()
        rec <- list()
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
            err[[i]] <- cls$output-newtruth[[i]]
            rec[[i]] <- cls$output
        }
    }

    err <- matrix(unlist(err),ncol=length(err))

    # if(record){
    #     prednos_MC <- apply(rec,1,function(p){return(sample(1:length(newtruth[[1]]),1,prob=p))})
    #     prednos <- apply(rec,1,function(p){x<-which(p==max(p));return(x[sample(1:length(x),1)])})
    #     tru <- sapply(newtruth,function(x){return(which(x==1))})
    # }

    if(plot){
        par(mfrow=c(2,2))

        newtruth <- matrix(unlist(newtruth),ncol=length(newtruth))
        rectmp <- matrix(unlist(rec),ncol=length(rec))
        tru <- matrix(unlist(newtruth),ncol=length(newtruth))

        plot(as.vector(rectmp),as.vector(err),xlab="Fitted",ylab="Residuals")
        lines(lowess(as.vector(rectmp),as.vector(err)),col="red")
        abline(h=0,col="green")

        hist(as.vector(err),main="Residuals")
        # plot(err)
        # lines(lowess(err),col="red",main="")

        plot(as.vector(tru),as.vector(rectmp),xlab="Truth",ylab="Fitted")
        lines(lowess(as.vector(tru),as.vector(rectmp)),col="red")
        abline(0,1,col="green")

        #matplot(err,type="l",col=rgb(0,0,1,alpha=0.25),lty="solid",xlab="Index",ylab="Residual")



        #matplot(newtruth,type="l",col=rgb(0,0,1,alpha=0.25),lty="solid",xlab="Index",ylab="Truth",ylim=rg)
        #matplot(rectmp,type="l",col=rgb(0,0,1,alpha=0.25),lty="solid",xlab="Index",ylab="Fitted",ylim=rg)

        #plot(jitter(tru),jitter(prednos),main="Prediction Vs Truth ABS")
        #plot(jitter(tru),jitter(prednos_MC),main="Prediction Vs Truth MC")


    }

    if(record){
        return(list(rec=rec,err=err,truth=tru))
    }
    else{
        return(cls$output)
    }
}
