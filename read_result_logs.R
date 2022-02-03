# read c++ mpi log files to create dataframes / graphs

# later : write loop over file names (different sizes)
library(readr)

getValue = function(filepath, pattern){
  full_pattern = paste0("^", pattern, ".*")
  #print(full_pattern)
  data_file <- readLines(file_name)
  #print(data_file)
  file_sub = scan(text = data_file[grep(full_pattern, data_file)], what = "")
  #print(file_sub)
  value = as.numeric(file_sub[2:length(file_sub)])
  #print(value)
  return(value)
}


#############################################################

ns = 492
nb = 6

t_list = c(50,100)
res_t_list = t_list # overwrite later, just same length

pattern = "time_bfgs"

for(i in 1:length(t_list)){

	num_obs = 2*ns*t_list[i]

	base_path = paste0("/home/x_gaedkelb/b_INLA/data/synthetic/ns492_nt", t_list[i], "_nb", nb)
	file_name = file.path(base_path, paste0("log_RGF_ns492_nt", t_list[i], "_nb", nb, "_no", num_obs,".dat"))

	# pattern can e.g. be "t_total", "num_obs", ... 
	res_t_list[i] = getValue(file_name, pattern)

}

print(t_list)
print(res_t_list)