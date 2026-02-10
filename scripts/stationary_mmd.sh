## mixture of Gaussians
particle_num=10
python main.py --seed 0 --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 0.0 --step_num 10000

for particle_num in 30 100
do
python main.py --seed 0 --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 0.0 --step_num 100000
done

particle_num=300
python main.py --seed $seed --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 0.0 --step_num 100000

particle_num=1000
python main.py --seed $seed --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 0.0 --step_num 300000

## Elevators

for seed in {0}
do
  for particle_num in 10 30 100
  do
    python main.py --seed $seed --particle_num $particle_num --dataset elevators --step_size 0.3 --bandwidth 1.0 --inject_noise_scale 1.0 --step_num 100000
  done
  for particle_num in 300
  do
    python main.py --seed $seed --particle_num $particle_num --dataset elevators --step_size 0.3 --bandwidth 1.0 --inject_noise_scale 1.0 --step_num 100000
  done
  for particle_num in 1000
  do
    python main.py --seed $seed --particle_num $particle_num --dataset elevators --step_size 0.3 --bandwidth 1.0 --inject_noise_scale 1.0 --step_num 100000
  done
done

## House
for seed in {0}
do
  for particle_num in 10 30 100
  do
    python main.py --seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 1.0 --step_num 100000
  done
  for particle_num in 300
  do
    python main.py --seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 1.0 --step_num 100000
  done
  for particle_num in 1000
  do
    python main.py --seed $seed --particle_num $particle_num --dataset house_8L --step_size 0.3 --bandwidth 1.0 --inject_noise_scale 1.0 --step_num 100000
  done
done
