for seed in {0}
do
  for m in 4 5 6 7 8 9
  do
    python kt.py --seed $seed --m $m --dataset mog --step_size 1.0 --bandwidth 1.0
  done
done

for seed in {0}
do
  for m in 4 5 6 7 8 9
  do
    python kt.py --seed $seed --m $m --dataset house_8L --step_size 1.0 --bandwidth 1.0
  done
done

for seed in {0}
do
  for m in 4 5 6 7 8 9
  do
    python kt.py --seed $seed --m $m --dataset elevators --step_size 1.0 --bandwidth 1.0
  done
done


