## runscript for nsys

n=5
num_ranks=1

#nsys profile -o nsys_output_${n} ./main ${n}
srun -n ${num_ranks} compute-sanitizer ./main ${n} #>output_${n}.txt

