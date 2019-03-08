##' stopping function
##'
##' Generic function for implementing stopping methods
##'
##' @param ... additional arguments
##' @seealso \link{stopping.default}, \link{stopping.maxit}
##' @return method stopping


stopping <- function(...){
    UseMethod("stopping")
}


##' stopping.default function
##'
##' A function to halt computation when curcost < tol
##'
##' @method stopping default
##' @param cost the value of the loss function passed in
##' @param curcost current measure of cost, can be different to the parameter 'cost' above e.g. may consider smoothed cost over the last k iterations
##' @param count iteration count
##' @param tol tolerance, or limit
##' @param ... additional arguments
##' @seealso \link{stopping.maxit}
##' @return ...

stopping.default <- function(cost,curcost,count,tol,...){
    return(curcost<tol)
}

##' stopping.maxit function
##'
##' A function to halt computation when the number of iterations reaches a given threshold, tol
##'
##' @method stopping maxit
##' @param cost the value of the loss function passed in
##' @param curcost current measure of cost, can be different to the parameter 'cost' above e.g. may consider smoothed cost over the last k iterations
##' @param count iteration count
##' @param tol tolerance, or limit
##' @param ... additional arguments
##' @return ...

stopping.maxit <- function(cost,curcost,count,tol,...){
    return(count<tol)
}



##' updateStopping function
##'
##' Generic function for updating stopping criteria
##'
##' @param ... additional arguments
##' @seealso \link{updateStopping.classification}, \link{updateStopping.regression}
##' @return method updateStopping


updateStopping <- function(...){
    UseMethod("updateStopping")
}




##' updateStopping.classification function
##'
##' A function to update the stopping criteria for a classification problem.
##'
##' @method updateStopping classification
##' @param dat data object
##' @param parms model parameters
##' @param net an object of class network
##' @param truth the truth, to be compared with network outputs
##' @param testoutput a vector, the history of the stopping criteria
##' @param count iteration number
##' @param monitor logical, whether to produce a diagnostic plot
##' @param mx a number to be monitored e.g. the cost of the best performing paramerer configuration to date
##' @param curcost current measure of cost, can be different to the value of the loss function e.g. may consider smoothed cost (i.e. loss) over the last k iterations
##' @param ... additional arguments
##' @return curcost, testoutput and mx, used for iterating the maximisation process

updateStopping.classification <- function(dat,parms,net,truth,testoutput,count,monitor,mx,curcost,...){
    tstidx <- sample(1:length(dat),1)
    w <- weights2list(parms[1:nnetpar(net)],net$dims)
    b <- bias2list(parms[(nnetpar(net)+1):length(parms)],net$dims)
    cls <- net$forward_pass(dat[[tstidx]],
                            weights=w,
                            bias=b,
                            dims=net$dims,
                            nlayers=net$nlayers,
                            activ=net$activ,
                            back=TRUE,
                            regulariser=net$regulariser)

    idx <- which(truth[[tstidx]]==1)
    TPR <- cls$output[idx]
    #FPR <- 1-max(cls$output[-idx])
    #TST <- TPR #min(TPR,FPR)
    #TST <- mean((truth[[tstidx]]-cls$output)^2)
    testoutput <- c(testoutput,TPR)

    if(count>20){
        curcost <- mean(testoutput[(count-20):count])
    }

    #browser()

    if(monitor){
        if(count<200){
            plot(testoutput,type="l",ylim=c(0,1),main="Monitor")
        }
        else{
            plot(testoutput[(count-199):count],type="l",ylim=c(0,1),sub="Monitor",main=c(count,mx))
        }
    }

    if(curcost>mx){
        mx <- curcost
    }

    return(list(curcost=curcost,testoutput=testoutput,mx=mx))

}


##' updateStopping.regression function
##'
##' A function to update the stopping criteria for a classification problem.
##'
##' @method updateStopping regression
##' @param dat data object
##' @param parms model parameters
##' @param net an object of class network
##' @param truth the truth, to be compared with network outputs
##' @param testoutput a vector, the history of the stopping criteria
##' @param count iteration number
##' @param monitor logical, whether to produce a diagnostic plot
##' @param mx a number to be monitored e.g. the cost of the best performing paramerer configuration to date
##' @param curcost current measure of cost, can be different to the value of the loss function e.g. may consider smoothed cost (i.e. loss) over the last k iterations
##' @param ... additional arguments
##' @return curcost, testoutput and mx, used for iterating the maximisation process

updateStopping.regression <- function(dat,parms,net,truth,testoutput,count,monitor,mx,curcost,...){

    stop("UNDER DEVELOPMENT")

    tstidx <- sample(1:length(dat),1)
    w <- weights2list(parms[1:nnetpar(net)],net$dims)
    b <- bias2list(parms[(nnetpar(net)+1):length(parms)],net$dims)
    cls <- net$forward_pass(dat[[tstidx]],
                            weights=w,
                            bias=b,
                            dims=net$dims,
                            nlayers=net$nlayers,
                            activ=net$activ,
                            back=TRUE,
                            regulariser=net$regulariser)


    TST <- mean((truth[[tstidx]]-cls$output)^2)
    testoutput <- c(testoutput,TST)

    if(count>20){
        curcost <- mean(testoutput[(count-20):count])
    }

    if(monitor){
        if(count<200){
            plot(testoutput,type="l",ylim=c(0,1),main="Monitor")
        }
        else{
            plot(testoutput[(count-199):count],type="l",ylim=c(0,1),sub="Monitor",main=c(count,mx))
        }
    }

    if(curcost>mx){
        mx <- curcost
    }

    return(list(curcost=curcost,testoutput=testoutput,mx=mx))

}
