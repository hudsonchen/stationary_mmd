for seed in {0}
do
  for particle_num in 10 30 100
    do
    python sp.py --seed $seed --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --step_num 10000
    done
  for particle_num in 300 1000
    do
        python sp.py --seed $seed --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --step_num 30000
    done
done

for seed in {0}
do
  for particle_num in 10 30 100
    do
    python sp.py --seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0 --step_num 10000
    done
  for particle_num in 300
    do
        python sp.py --seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0 --step_num 30000
    done
  for particle_num in 1000
    do
        python sp.py --seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0 --step_num 30000
    done
done

for seed in {0}
do
  for particle_num in 10 30 100
    do
    python sp.py --seed $seed --particle_num $particle_num --dataset elevators --step_size 1.0 --bandwidth 1.0 --step_num 10000
    done
  for particle_num in 300
    do
        python sp.py --seed $seed --particle_num $particle_num --dataset elevators --step_size 1.0 --bandwidth 1.0 --step_num 30000
    done
  for particle_num in 1000
    do
        python sp.py --seed $seed --particle_num $particle_num --dataset elevators --step_size 1.0 --bandwidth 1.0 --step_num 30000
    done
done