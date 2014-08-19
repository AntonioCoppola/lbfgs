\name{Leukemia}
\alias{Leukemia}
\docType{data}
\title{
Data from Golub et al. 1999
}
\description{
Data from Golub, Todd R., et al. "Molecular classification of cancer: 
class discovery and class prediction by gene expression monitoring." 
Science 286.5439 (1999): 531-537. The study uses microarray data to perform
cancer classification based on gene expression monitoring.}

\value{
\item{y}{A vector of binary values specifying the cancer class for 72 leukemia patients. A value of 0 corresponds to patients with acute lymphoblastic leukemia (ALL), and 1 corresponds to patients with acute myeloid leukemia (AML).}
\item{x}{A 72-by-3571 matrix specifying the levels of expressions of 3571 genes for the 72 different patients.}
}

\usage{data(Leukemia)}

\keyword{datasets}
