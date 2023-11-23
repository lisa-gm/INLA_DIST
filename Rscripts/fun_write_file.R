# --------- DIFFERENT FUNCTIONS TO WRITE TO FILE --------------------#
# disable scientific notation !!
options(scipen=999)

library(data.table)

# write DENSE matrix to file
mat_to_file.fun <- function(M, file_name, file_path, append = FALSE, no_obs = NULL, ns=NULL, nt=NULL){
  
  dim_M <- dim(M)

    if(is.null(dim_M)){
    dim_M <- c(length(M), 1)
  }

  if(any(is.na(M))){
     stop("NANs in matrix! Remove.")
  }
  
  M.df <- data.frame(M)

  if(is.null(no_obs) != TRUE){
    input_file = file.path(file_path, paste(file_name, "_", no_obs, "_", toString(dim_M[2]), ".dat", sep=""))    
  } else if( is.null(ns) != TRUE & is.null(nt) != TRUE){
    input_file = file.path(file_path, paste(file_name, "_", toString(dim_M[1]), "_", toString(dim_M[2]), "_ns", toString(ns), "_nt", toString(nt), ".dat", sep=""))
  } else{
    input_file = file.path(file_path, paste(file_name, "_", toString(dim_M[1]), "_", toString(dim_M[2]), ".dat", sep=""))    
  }

  fwrite(M.df, input_file, append = append, sep = " ", dec = ".", row.names = FALSE, col.names = FALSE)
  print(paste("wrote matrix to file under :", input_file))

}

mat_to_file_wNA.fun <- function(M, file_name, file_path, append = FALSE, no_obs = NULL, ns=NULL, nt=NULL){
  
  dim_M <- dim(M)

    if(is.null(dim_M)){
    dim_M <- c(length(M), 1)
  }
  
  M.df <- data.frame(M)

  if(is.null(no_obs) != TRUE){
    input_file = file.path(file_path, paste(file_name, "_", no_obs, "_", toString(dim_M[2]), ".dat", sep=""))    
  } else if( is.null(ns) != TRUE & is.null(nt) != TRUE){
    input_file = file.path(file_path, paste(file_name, "_", toString(dim_M[1]), "_", toString(dim_M[2]), "_ns", toString(ns), "_nt", toString(nt), ".dat", sep=""))
  } else{
    input_file = file.path(file_path, paste(file_name, "_", toString(dim_M[1]), "_", toString(dim_M[2]), ".dat", sep=""))    
  }

  fwrite(M.df, input_file, append = append, sep = " ", dec = ".", na = "NaN", row.names = FALSE, col.names = FALSE)
  print(paste("wrote matrix to file under :", input_file))

}

# matrix has to be in sparse (dgC format), assumed to be quadratic
mat_to_file_sp.fun <- function(M, file_name, file_path, no_obs=NULL, ns=NULL, nt=NULL){
  
  if(any(is.na(M@x))){
     stop("NANs in matrix! Remove.")
  }	

  dim_M <- M@Dim
  nnz <- nnzero(M) + sum(M@x==0) # have to also count diagonal entries
  M.df <- data.frame(c(dim_M[1], dim_M[2], nnz, M@i, M@p, M@x))
  
if(is.null(no_obs) != TRUE){
    input_file = file.path(file_path, paste(file_name, "_", no_obs, "_", toString(dim_M[2]), ".dat", sep=""))    

} else if( is.null(ns) != TRUE & is.null(nt) != TRUE){
    input_file = file.path(file_path, paste(file_name, "_", toString(dim_M[1]), "_", toString(dim_M[2]), "_ns", toString(ns), "_nt", toString(nt), ".dat", sep=""))

} else{
    input_file = file.path(file_path, paste(file_name, "_", toString(dim_M[1]), "_", toString(dim_M[2]), ".dat", sep=""))    
}

  fwrite(M.df, input_file, append = FALSE, sep = " ", dec = ".", row.names = FALSE, col.names = FALSE)
  print(paste("wrote matrix to file under :", input_file))
  
}

# matrix has to be in sparse (dgC format) & symmetric
mat_to_file_sym.fun <- function(M, file_name, file_path){
  
  M_lower <- tril(M)
  n <- M_lower@Dim[1]
  nnz <- nnzero(M_lower) + sum(M_lower@x==0) # have to also count diagonal entries
  M_lower.df <- data.frame(c(n, n, nnz, M_lower@i, M_lower@p, M_lower@x))
  
  input_file = file.path(file_path, paste(file_name, "_", toString(n), ".dat", sep=""))
  fwrite(M_lower.df, input_file, append = FALSE, sep = " ", dec = ".", row.names = FALSE, col.names = FALSE)
  print(paste("wrote matrix to file under :", input_file))
  
  return(n)
}

# write other constants to file
const_to_file.fun <- function(const, file_name, file_path){
  input_file = file.path(file_path, file_name)
  const.df <- data.frame((const))
  
  fwrite(const.df, input_file, sep = " ", dec = ".",row.names = FALSE, col.names = FALSE)
  print(paste("wrote constants to file under :", input_file))
}
