#module load cuda/12.1
#make
#echo "Compiling CUDA code.."
#crun.cuda nvcc -o pso parallel_pso.cu
#N=335544320
set -euo pipefail

OPT=(1024)   # 2^17 for 10d
#N=1048576  # 2^19 for 32d
#N=8388608  # 2^23
#N=16777216 # 2^24 
#PSO=(0 1 2 3 4 5 10 30) # 100 300 1000)
PSO=(5)
#PSO=(100 300 1000 3000 10000)
#PSO=(0)
DIM=5
for i in {1..1000}
do
    echo -e "\n\n\t\t=== Run $i ===="
    for ITER in "${OPT[@]}"; do
        echo "=== Running with PSO ITER=$ITER ==="
	    for FUNC in rosenbrock; do

        # pick the threshold by function
        case "$FUNC" in
          rosenbrock) THRESHOLD="6e-4" ;;
          rastrigin)  THRESHOLD="5e-4" ;;
          ackley)     THRESHOLD="1.5"   ;;
          *) 
            echo "Unknown function '$FUNC'" >&2
            exit 1
            ;;
        esac
	      {
	         echo $FUNC  # Function selector
	         echo $DIM      # Dimension of the problem
	         echo "n"      # Exit loop
	       } | ./main -5.12 5.12 10000 5 100 "$ITER" "$THRESHOLD" "$(($i*2)))" "$i"
	    done
    done
done