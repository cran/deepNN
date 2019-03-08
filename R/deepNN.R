##' deepNN
##'
##' Teaching resources (yet to be added) and implementation of some Deep Learning methods. Includes multilayer perceptron, different activation functions, regularisation strategies, stochastic gradient descent and dropout.
##'
##' \tabular{ll}{
##' Package: \tab deepNN\cr
##' Version: \tab 0.1\cr
##' Date: \tab 2019-01-11\cr
##' License: \tab GPL-3 \cr
##' }
##'
##'
##'
##' section{Dependencies}{
##' The package \code{deepNN} depends upon some other important contributions to CRAN in order to operate; their uses here are indicated:\cr\cr
##'     stats, graphics.
##' }
##'
##' section{Citation}{
##' deepNN: Deep Learning. Benjamin M. Taylor
##' }
##'
##' references{
##' \enumerate{
##'     \item Ian Goodfellow, Yoshua Bengio, Aaron Courville, Francis Bach. Deep Learning. (2016)
##'     \item Terrence J. Sejnowski. The Deep Learning Revolution (The MIT Press). (2018)
##'     \item Neural Networks YouTube playlist by 3brown1blue: \url{https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi}
##'     \item{http://neuralnetworksanddeeplearning.com/}
##' }
##' }
##'
##' @docType package
##' @name deepNN-package
##' @author Benjamin Taylor, Department of Medicine, Lancaster University
##' @keywords package
##'
##'

## @import sp
##' @import stats
##' @import graphics
##' @import utils
##' @import methods
##' @importFrom Matrix Matrix sparseMatrix
## @import gradDescent
## @import stringr
## @import lubridate
## @import ncdf

# instead import functions individually
## @importFrom sp bbox proj4string<- proj4string SpatialPixelsDataFrame SpatialGridDataFrame Polygon Polygons SpatialPolygons coordinates CRS geometry GridTopology over proj4string SpatialGrid SpatialPixels SpatialPoints SpatialPolygonsDataFrame split spTransform
## @importFrom raster raster crop
## @importFrom stringr str_split str_match str_trim
## @importFrom lubridate ymd year month week day
## @importFrom ncdf open.ncdf close.ncdf sync.ncdf get.var.ncdf dim.def.ncdf var.def.ncdf create.ncdf put.var.ncdf



### @importFrom raster raster crop
### @importFrom sp bbox proj4string<- proj4string SpatialPixelsDataFrame SpatialGridDataFrame Polygon Polygons SpatialPolygons coordinates CRS geometry GridTopology over proj4string SpatialGrid SpatialPixels SpatialPoints SpatialPolygonsDataFrame split spTransform
### @importFrom RColorBrewer brewer.pal
### @importFrom stringr str_count str_detect
### @importFrom Matrix Matrix sparseMatrix
### @importFrom rgl abclines3d aspect3d axes3d planes3d points3d segments3d text3d title3d
### @importFrom fields image.plot
### @importFrom RandomFields CovarianceFct
### @importFrom rgeos gBuffer
### @importFrom iterators icount iter nextElem
### @importFrom sp bbox proj4string<- proj4string SpatialPixelsDataFrame SpatialGridDataFrame Polygon Polygons SpatialPolygons coordinates CRS geometry GridTopology over proj4string SpatialGrid SpatialPixels SpatialPoints SpatialPolygonsDataFrame split spTransform
### @importFrom spatstat rpoint progressreport
### @importFrom survival Surv survfit
### @importFrom geostatsp asImRaster
### @importFrom raster crop
### @importFrom stats acf coefficients deriv dexp dist dnorm end fft fitted formula Gamma integrate knots lm model.matrix optim optimise poly quantile rbinom rexp rnorm runif sd start update var
### @importFrom graphics hist legend lines matplot par plot points title






`deepNN` = NA
